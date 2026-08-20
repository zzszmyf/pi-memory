/**
 * Memory tools — the agent's notebook interface.
 *
 * Exposes write_note / search_notes / read_recent_notes as Pi tool
 * definitions so the agent decides when to write and when to look things up.
 * Vectorization is best-effort background work (same pattern as the passive
 * mirror); keyword search works even when the embedding API is down.
 */

import type { AgentToolResult, AgentToolUpdateCallback } from "@earendil-works/pi-agent-core";
import type { ExtensionContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { type Static, Type } from "typebox";
import type { EmbeddingClientOptions } from "./embedding.ts";
import { EmbeddingClient } from "./embedding.ts";
import { Notebook } from "./notebook.ts";

const writeNoteSchema = Type.Object({
	content: Type.String({ description: "What to remember (a fact, decision, convention, gotcha, or progress)" }),
	category: Type.Optional(
		Type.String({ description: "Optional label: preference / decision / convention / gotcha / progress" }),
	),
});

const searchNotesSchema = Type.Object({
	query: Type.String({ description: "What to look for (keywords or a natural-language question)" }),
	topK: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
});

const readRecentSchema = Type.Object({
	n: Type.Optional(Type.Number({ description: "Number of most recent notes (default 10)" })),
});

export interface MemoryToolsOptions {
	/** Lazy context resolver — evaluated on every tool call so the extension
	 * can initialize cwd/session/embedding after the first context event. */
	getContext: () => {
		sessionId: string;
		dbPath: string;
		embedding?: EmbeddingClientOptions;
	};
}

export function createMemoryToolDefinitions(options: MemoryToolsOptions): ToolDefinition[] {
	// Lazy singletons, resolved from getContext on first use.
	let notebook: Notebook | null = null;
	let embedder: EmbeddingClient | null = null;

	const resolve = () => {
		const ctx = options.getContext();
		if (!notebook) notebook = new Notebook(ctx.dbPath);
		if (!embedder && ctx.embedding) embedder = new EmbeddingClient(ctx.embedding);
		return { ...ctx, notebook, embedder };
	};

	const vectorizePending = async () => {
		const { notebook: nb, embedder: emb, sessionId } = resolve();
		if (!emb || !nb) return;
		try {
			const pending = nb.pendingVectorIds(sessionId, 32);
			if (pending.length === 0) return;
			const vecs = await emb.embed(pending.map((p) => p.content));
			for (let i = 0; i < pending.length; i++) {
				nb.storeVector(pending[i].id, vecs[i]);
			}
		} catch (error) {
			console.error("[memory] note vectorization failed:", error);
		}
	};

	const text = (s: string): AgentToolResult<unknown> => ({
		content: [{ type: "text", text: s }],
		details: undefined,
	});

	const writeNoteTool: ToolDefinition = {
		name: "write_note",
		label: "Write note",
		description:
			"Write a note to your memory notebook. Use this for facts, decisions, user preferences, conventions, " +
			"gotchas and progress you want to remember beyond the current conversation. Be concise — one fact per note.",
		promptSnippet: "Write a memory note",
		promptGuidelines: [
			"Take notes when you learn something you'll need later (user preferences, decisions, conventions, gotchas).",
			"Before starting a task, search your notes for relevant prior knowledge.",
		],
		parameters: writeNoteSchema,
		async execute(
			_toolCallId: string,
			params: Static<typeof writeNoteSchema>,
			_signal: AbortSignal | undefined,
			_onUpdate: AgentToolUpdateCallback<unknown> | undefined,
			_ctx: ExtensionContext,
		) {
			const { notebook: nb, sessionId } = resolve();
			const note = nb.write(sessionId, params.content, params.category);
			void vectorizePending();
			return text(`Noted${note.category ? ` (${note.category})` : ""}.`);
		},
	};

	const searchNotesTool: ToolDefinition = {
		name: "search_notes",
		label: "Search notes",
		description:
			"Search your memory notebook. Combines keyword matching with semantic similarity when available. " +
			"Use this before starting a task to recall prior decisions, preferences and gotchas.",
		promptSnippet: "Search memory notes",
		parameters: searchNotesSchema,
		async execute(
			_toolCallId: string,
			params: Static<typeof searchNotesSchema>,
			_signal: AbortSignal | undefined,
			_onUpdate: AgentToolUpdateCallback<unknown> | undefined,
			_ctx: ExtensionContext,
		) {
			const topK = Math.min(20, Math.max(1, Math.floor(params.topK ?? 5)));
			await vectorizePending();
			const { notebook: nb2, sessionId: sid2 } = resolve();
			let notes = nb2.searchKeyword(sid2, params.query, topK * 2);
			const { embedder: emb2 } = resolve();
			if (emb2) {
				try {
					const [queryVec] = await emb2.embed([params.query]);
					const semantic = nb2.searchVector(sid2, queryVec, topK * 2);
					const seen = new Set(notes.map((n) => n.id));
					for (const n of semantic) {
						if (!seen.has(n.id)) {
							notes.push(n);
							seen.add(n.id);
						}
					}
					notes.sort((a, b) => b.createdAt - a.createdAt);
				} catch (error) {
					console.error("[memory] semantic note search failed, keyword-only:", error);
				}
			}
			notes = notes.slice(0, topK);
			if (notes.length === 0) {
				return text("No notes match.");
			}
			const lines = notes.map((n, i) => `${i + 1}. [${n.category ?? "note"}] ${n.content}`);
			return text(lines.join("\n"));
		},
	};

	const readRecentTool: ToolDefinition = {
		name: "read_recent_notes",
		label: "Recent notes",
		description: "Read your most recent memory notes in chronological order (newest first).",
		promptSnippet: "Read recent memory notes",
		parameters: readRecentSchema,
		async execute(
			_toolCallId: string,
			params: Static<typeof readRecentSchema>,
			_signal: AbortSignal | undefined,
			_onUpdate: AgentToolUpdateCallback<unknown> | undefined,
			_ctx: ExtensionContext,
		) {
			const n = Math.min(50, Math.max(1, Math.floor(params.n ?? 10)));
			const { notebook: nb3 } = resolve();
			const notes = nb3.readRecent("", n);
			if (notes.length === 0) {
				return text("Notebook is empty.");
			}
			const lines = notes.map((n, i) => `${i + 1}. [${n.category ?? "note"}] ${n.content}`);
			return text(lines.join("\n"));
		},
	};

	return [writeNoteTool, searchNotesTool, readRecentTool];
}
