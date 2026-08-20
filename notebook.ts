/**
 * The "notebook beside the agent" — active note-taking layer on top of the
 * passive message mirror.
 *
 * Unlike the mirror (which records everything that happened), the notebook
 * records what the agent decides is worth remembering. The agent writes notes
 * through tools, reads them back through tools, and a background subagent can
 * later distill them into structured memory.
 *
 * Three tools:
 * - write_note:       agent decides something is worth remembering
 * - search_notes:     hybrid BM25 + vector retrieval over notes
 * - read_recent_notes: chronological window (temporal locality)
 *
 * Notes live in the same SQLite database as the passive mirror, in separate
 * tables so the two layers never pollute each other.
 */

import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { cosineSimilarity } from "./embedding.ts";

export interface Note {
	id: string;
	category: string | null;
	content: string;
	createdAt: number;
}

const NOTES_SCHEMA = `
CREATE TABLE IF NOT EXISTS notes (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	category TEXT,
	content TEXT NOT NULL,
	created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
	content,
	content='notes',
	content_rowid='rowid',
	tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
	INSERT INTO note_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
	INSERT INTO note_fts(note_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TABLE IF NOT EXISTS note_vec (
	id TEXT PRIMARY KEY,
	vec BLOB NOT NULL
);
`;

export class Notebook {
	private readonly db: DatabaseSync;

	constructor(dbPath: string) {
		mkdirSync(dirname(dbPath), { recursive: true });
		this.db = new DatabaseSync(dbPath);
		this.db.exec("PRAGMA journal_mode = WAL;");
		this.db.exec(NOTES_SCHEMA);
	}

	close(): void {
		this.db.close();
	}

	/** Write a note (idempotent via content hash within the session). */
	write(sessionId: string, content: string, category?: string): Note {
		const id = createHash("sha1").update(`note\0${sessionId}\0${content}`).digest("hex").slice(0, 32);
		const now = Date.now();
		this.db
			.prepare("INSERT OR REPLACE INTO notes (id, session_id, category, content, created_at) VALUES (?, ?, ?, ?, ?)")
			.run(id, sessionId, category ?? null, content, now);
		return { id, category: category ?? null, content, createdAt: now };
	}

	/** Delete a note (lifecycle maintenance from the distillation subagent). */
	delete(id: string): void {
		this.db.prepare("DELETE FROM notes WHERE id = ?").run(id);
	}

	/** Recent N notes, newest first (temporal locality — "flip the notebook"). */
	readRecent(sessionId: string, n: number): Note[] {
		const rows = this.db
			.prepare("SELECT id, category, content, created_at FROM notes ORDER BY created_at DESC LIMIT ?")
			.all(n) as Array<{ id: string; category: string | null; content: string; created_at: number }>;
		return rows.map((r) => ({
			id: r.id,
			category: r.category,
			content: r.content,
			createdAt: r.created_at,
		}));
	}

	/** Keyword search over notes (FTS5 BM25), newest-ranked within score ties. */
	searchKeyword(sessionId: string, query: string, topK: number): Note[] {
		const rows = this.db
			.prepare(
				`SELECT n.id, n.category, n.content, n.created_at, bm25(note_fts) AS score
				 FROM note_fts
				 JOIN notes n ON n.rowid = note_fts.rowid
				 WHERE note_fts MATCH ?
				 ORDER BY score, n.created_at DESC
				 LIMIT ?`,
			)
			.all(this.#ftsQuery(query), topK) as Array<{
			id: string;
			category: string | null;
			content: string;
			created_at: number;
			score: number;
		}>;
		return rows.map((r) => ({ id: r.id, category: r.category, content: r.content, createdAt: r.created_at }));
	}

	/** Vector search over notes (cosine similarity). */
	searchVector(sessionId: string, queryVector: Float32Array, topK: number): Note[] {
		const rows = this.db
			.prepare(
				`SELECT v.id, v.vec, n.category, n.content, n.created_at
				 FROM note_vec v
				 JOIN notes n ON n.id = v.id
				 `,
			)
			.all(sessionId) as Array<{
			id: string;
			vec: Uint8Array;
			category: string | null;
			content: string;
			created_at: number;
		}>;
		const scored = rows.map((r) => {
			const vec = new Float32Array(r.vec.buffer, r.vec.byteOffset, r.vec.byteLength / 4);
			return { ...r, cosine: cosineSimilarity(queryVector, vec) };
		});
		scored.sort((a, b) => b.cosine - a.cosine);
		return scored
			.slice(0, topK)
			.map((r) => ({ id: r.id, category: r.category, content: r.content, createdAt: r.created_at }));
	}

	storeVector(id: string, vec: Float32Array): void {
		const blob = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
		this.db.prepare("INSERT OR REPLACE INTO note_vec (id, vec) VALUES (?, ?)").run(id, blob);
	}

	pendingVectorIds(sessionId: string, limit: number): Array<{ id: string; content: string }> {
		return this.db
			.prepare(
				`SELECT n.id, n.content FROM notes n
				 LEFT JOIN note_vec v ON v.id = n.id
				 WHERE v.id IS NULL
				 ORDER BY n.created_at ASC
				 LIMIT ?`,
			)
			.all(limit) as Array<{ id: string; content: string }>;
	}

	#ftsQuery(query: string): string {
		const tokens = query
			.split(/[^\p{L}\p{N}]+/u)
			.filter((t) => t.length > 1)
			.slice(0, 32);
		if (tokens.length === 0) {
			return '""';
		}
		return tokens.map((t) => `"${t.replaceAll('"', '""')}"`).join(" OR ");
	}
}
