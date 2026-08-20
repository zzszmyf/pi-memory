/**
 * Distillation subagent — "organize the notebook while you sleep".
 *
 * Reads the passive message mirror (entries table), batches recent messages
 * through an LLM, and distills them into structured notebook entries:
 * preferences, decisions, conventions, gotchas, progress. Also performs
 * lifecycle maintenance: marks which existing notes should be merged or
 * deleted because they were superseded.
 *
 * Runs as an independent process/command so it never disturbs the main
 * agent session. Trigger manually, on session end, or on a schedule.
 */

import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { Notebook } from "./notebook.ts";

export interface DistillOptions {
	/** OpenAI-compatible chat completions endpoint (base URL + model + key). */
	llm: {
		baseUrl: string;
		apiKey: string;
		model: string;
	};
	/** Batch size of source messages per distillation call. */
	batchSize?: number;
}

interface DistillResponse {
	notes: Array<{ category: string; content: string }>;
	merge?: number[];
	delete?: number[];
}

const DISTILL_PROMPT = `You are a memory distillation assistant. Below are raw conversation messages from a coding agent session. Distill them into durable, structured memory notes.

Rules:
- One fact per note. Be concise and self-contained (no pronouns without referents).
- Categories: preference (user taste/style), decision (architecture/technical choice), convention (project rule), gotcha (pitfall/fix), progress (work state).
- Skip transient content (tool chatter, test output, greetings).
- If an existing note in the "Existing notes" section is clearly superseded by these messages, list its index in "merge" or "delete" (merge = fold into a new note, delete = no longer true).
- Respond with JSON only: {"notes":[{"category":"...","content":"..."}],"merge":[1],"delete":[2]}

Existing notes:
{existing}

Raw messages:
{raw}`;

export class Distiller {
	private readonly db: DatabaseSync;
	private readonly notebook: Notebook;
	private readonly options: DistillOptions;

	constructor(dbPath: string, options: DistillOptions) {
		this.db = new DatabaseSync(dbPath);
		this.db.exec("PRAGMA journal_mode = WAL;");
		this.db.exec("ALTER TABLE entries ADD COLUMN distilled_at INTEGER"); // no-op if exists? handled below
		this.notebook = new Notebook(dbPath);
		this.options = { batchSize: 50, ...options };
	}

	private ensureColumn(): void {
		try {
			this.db.exec("ALTER TABLE entries ADD COLUMN distilled_at INTEGER");
		} catch {
			// column already exists
		}
	}

	/** Source messages not yet distilled (oldest first, batched). */
	private pendingBatch(limit: number): Array<{ id: string; role: string; content: string }> {
		this.ensureColumn();
		return this.db
			.prepare("SELECT id, role, content FROM entries WHERE distilled_at IS NULL ORDER BY created_at ASC LIMIT ?")
			.all(limit) as Array<{ id: string; role: string; content: string }>;
	}

	private markDistilled(ids: string[]): void {
		if (ids.length === 0) return;
		const placeholders = ids.map(() => "?").join(",");
		this.db.prepare(`UPDATE entries SET distilled_at = ? WHERE id IN (${placeholders})`).run(Date.now(), ...ids);
	}

	/** Existing notes snapshot for lifecycle hints in the prompt. */
	private existingNotes(limit: number): string {
		const notes = this.notebook.readRecent("", limit);
		return notes.map((n, i) => `${i + 1}. [${n.category ?? "note"}] ${n.content}`).join("\n");
	}

	/**
	 * Run one distillation round over up to batchSize undigested messages.
	 * Returns counts of created / merged / deleted notes.
	 */
	async runOnce(): Promise<{ processed: number; created: number; merged: number; deleted: number }> {
		const batch = this.pendingBatch(this.options.batchSize ?? 50);
		if (batch.length === 0) {
			return { processed: 0, created: 0, merged: 0, deleted: 0 };
		}

		const raw = batch.map((m) => `[${m.role}] ${m.content}`).join("\n");
		const prompt = DISTILL_PROMPT.replace("{existing}", this.existingNotes(20)).replace("{raw}", raw);

		const response = await fetch(`${this.options.llm.baseUrl}/chat/completions`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${this.options.llm.apiKey}`,
			},
			body: JSON.stringify({
				model: this.options.llm.model,
				messages: [
					{ role: "system", content: "You are a memory distillation subagent. Respond with JSON only." },
					{ role: "user", content: prompt },
				],
				temperature: 0,
			}),
		});
		if (!response.ok) {
			throw new Error(`Distillation failed: ${response.status} ${await response.text()}`);
		}
		const body = (await response.json()) as {
			choices: Array<{ message: { content: string } }>;
		};
		const parsed = this.parseResponse(body.choices[0]?.message?.content ?? "{}");

		// Create new notes.
		let created = 0;
		for (const n of parsed.notes) {
			if (!n.content?.trim()) continue;
			const category = n.category || "note";
			this.notebook.write("distill", n.content, category);
			created++;
		}

		// Lifecycle: delete superseded notes (merge is handled by creating the
		// new note; the old one is deleted as well for simplicity).
		let deleted = 0;
		const existing = this.notebook.readRecent("", 20);
		for (const idx of [...(parsed.merge ?? []), ...(parsed.delete ?? [])]) {
			if (!Number.isInteger(idx) || idx < 1) continue;
			const note = existing[idx - 1];
			if (note) {
				this.notebook.delete(note.id);
				deleted++;
			}
		}

		this.markDistilled(batch.map((b) => b.id));
		return { processed: batch.length, created, merged: parsed.merge?.length ?? 0, deleted };
	}

	private parseResponse(content: string): DistillResponse {
		// Strip markdown fences if present.
		const cleaned = content
			.replace(/^```(?:json)?\s*/i, "")
			.replace(/```\s*$/, "")
			.trim();
		const start = cleaned.indexOf("{");
		const end = cleaned.lastIndexOf("}");
		if (start === -1 || end === -1) return { notes: [] };
		try {
			return JSON.parse(cleaned.slice(start, end + 1)) as DistillResponse;
		} catch {
			return { notes: [] };
		}
	}

	/** Run rounds until no undigested messages remain. */
	async runAll(): Promise<{ rounds: number; processed: number; created: number; deleted: number }> {
		let rounds = 0;
		let processed = 0;
		let created = 0;
		let deleted = 0;
		for (;;) {
			const result = await this.runOnce();
			if (result.processed === 0) break;
			rounds++;
			processed += result.processed;
			created += result.created;
			deleted += result.deleted;
		}
		return { rounds, processed, created, deleted };
	}
}

// Deterministic id helper for notebook deletion (reuse hash scheme).
export function noteId(sessionId: string, content: string): string {
	return createHash("sha1").update(`note\0${sessionId}\0${content}`).digest("hex").slice(0, 32);
}
