/**
 * Retrieval-augmented session memory backed by SQLite (node:sqlite).
 *
 * Every session message (user / assistant / tool results) is mirrored into a
 * per-project SQLite database with an FTS5 index. When a new request is built,
 * the latest user message is used as the query to pull the most relevant
 * historical messages back into context — replacing the "send the entire
 * history every turn" behavior with retrieval, so context usage stays flat
 * no matter how long the session runs.
 *
 * Design notes:
 * - node:sqlite is built into Node 22.5+, no native deps.
 * - FTS5 BM25 scoring provides keyword relevance; recency is applied as a
 *   multiplicative decay so old-but-relevant results can still surface.
 * - The database is per-project-directory (same granularity as Pi sessions),
 *   stored under the project dir as .pi/memory.sqlite.
 */

import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { cosineSimilarity } from "./embedding.ts";

export interface MemoryEntry {
	id: string;
	role: string;
	content: string;
	createdAt: number;
}

export interface RetrievalMemorySettings {
	/** Top-K results injected per request. */
	topK: number;
	/** Number of most recent messages always kept verbatim. */
	keepRecent: number;
	/** Half-life (ms) of the recency decay applied to BM25 scores. */
	halfLifeMs: number;
}

export const DEFAULT_RETRIEVAL_MEMORY_SETTINGS: RetrievalMemorySettings = {
	topK: 8,
	keepRecent: 12,
	halfLifeMs: 10 * 60 * 1000,
};

const SCHEMA = `
CREATE TABLE IF NOT EXISTS entries (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	role TEXT NOT NULL,
	content TEXT NOT NULL,
	created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS entry_fts USING fts5(
	content,
	content='entries',
	content_rowid='rowid',
	tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
	INSERT INTO entry_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
	INSERT INTO entry_fts(entry_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TABLE IF NOT EXISTS entries_vec (
	id TEXT PRIMARY KEY,
	vec BLOB NOT NULL
);
`;

export class RetrievalMemory {
	private readonly db: DatabaseSync;
	private readonly _settings: RetrievalMemorySettings;

	constructor(dbPath: string, settings?: Partial<RetrievalMemorySettings>) {
		mkdirSync(dirname(dbPath), { recursive: true });
		this.db = new DatabaseSync(dbPath);
		this._settings = { ...DEFAULT_RETRIEVAL_MEMORY_SETTINGS, ...settings };
		this.db.exec("PRAGMA journal_mode = WAL;");
		this.db.exec(SCHEMA);
	}

	close(): void {
		this.db.close();
	}

	get settings(): RetrievalMemorySettings {
		return this._settings;
	}

	has(id: string): boolean {
		const row = this.db.prepare("SELECT 1 FROM entries WHERE id = ?").get(id);
		return row !== undefined;
	}

	store(sessionId: string, role: string, content: string, createdAt = Date.now()): void {
		if (!content.trim()) {
			return;
		}
		// Deterministic content-hash id: re-sent history (the harness resends
		// the full context every turn) becomes an INSERT OR REPLACE no-op
		// instead of duplicating rows.
		const id = createHash("sha1").update(`${role}\0${content}`).digest("hex").slice(0, 32);
		this.db
			.prepare("INSERT OR REPLACE INTO entries (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)")
			.run(id, sessionId, role, content, createdAt);
	}

	/**
	 * FTS5 BM25 retrieval with recency decay. Returns entries sorted by score.
	 */
	retrieve(sessionId: string, query: string, topK = this._settings.topK): MemoryEntry[] {
		const now = Date.now();
		const halfLife = this._settings.halfLifeMs;
		// FTS5 BM25: bm25(entry_fts) ranks by keyword relevance.
		const rows = this.db
			.prepare(
				`SELECT e.id, e.role, e.content, e.created_at,
						bm25(entry_fts) AS score
				 FROM entry_fts
				 JOIN entries e ON e.rowid = entry_fts.rowid
				 WHERE entry_fts MATCH ? AND e.session_id = ?
				 ORDER BY score
				 LIMIT 200`,
			)
			.all(this.#ftsQuery(query), sessionId) as Array<{
			id: string;
			role: string;
			content: string;
			created_at: number;
			score: number;
		}>;

		// Decay: relevance = bm25 * 0.5^(age/halfLife). BM25 is negative
		// (lower is better), so convert to a positive relevance first.
		const entries = rows.map((r) => ({
			id: r.id,
			role: r.role,
			content: r.content,
			createdAt: r.created_at,
			relevance: -r.score * 0.5 ** ((now - r.created_at) / halfLife),
		}));

		entries.sort((a, b) => b.relevance - a.relevance);
		return entries.slice(0, topK).map(({ id, role, content, createdAt }) => ({
			id,
			role,
			content,
			createdAt,
		}));
	}

	/**
	 * Store an embedding vector for an entry (float32 little-endian blob).
	 * Returns false if the entry does not exist.
	 */
	storeVector(id: string, vec: Float32Array): boolean {
		const row = this.db.prepare("SELECT 1 FROM entries WHERE id = ?").get(id);
		if (!row) return false;
		const blob = Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
		this.db.prepare("INSERT OR REPLACE INTO entries_vec (id, vec) VALUES (?, ?)").run(id, blob);
		return true;
	}

	/** Ids of entries that still lack a stored vector. */
	pendingVectorIds(sessionId: string, limit: number): Array<{ id: string; content: string }> {
		const rows = this.db
			.prepare(
				`SELECT e.id, e.content FROM entries e
				 LEFT JOIN entries_vec v ON v.id = e.id
				 WHERE e.session_id = ? AND v.id IS NULL
				 ORDER BY e.created_at ASC
				 LIMIT ?`,
			)
			.all(sessionId, limit) as Array<{ id: string; content: string }>;
		return rows;
	}

	/**
	 * Hybrid retrieval: BM25 + cosine similarity fused with RRF.
	 * When no vectors are stored yet, degrades to pure BM25.
	 */
	async retrieveHybrid(
		sessionId: string,
		query: string,
		queryVector: Float32Array,
		topK = this._settings.topK,
	): Promise<MemoryEntry[]> {
		const now = Date.now();
		const halfLife = this._settings.halfLifeMs;

		// Keyword candidates.
		const keywordRows = this.db
			.prepare(
				`SELECT e.id, e.role, e.content, e.created_at,
						bm25(entry_fts) AS score
				 FROM entry_fts
				 JOIN entries e ON e.rowid = entry_fts.rowid
				 WHERE entry_fts MATCH ? AND e.session_id = ?
				 ORDER BY score
				 LIMIT 200`,
			)
			.all(this.#ftsQuery(query), sessionId) as Array<{
			id: string;
			role: string;
			content: string;
			created_at: number;
			score: number;
		}>;

		const keywordRank = new Map<string, number>();
		keywordRows.forEach((r, i) => keywordRank.set(r.id, i + 1));

		// Vector candidates: cosine over stored vectors.
		const vecRows = this.db
			.prepare(
				`SELECT v.id, v.vec, e.role, e.content, e.created_at
				 FROM entries_vec v
				 JOIN entries e ON e.id = v.id
				 WHERE e.session_id = ?`,
			)
			.all(sessionId) as Array<{
			id: string;
			vec: Uint8Array;
			role: string;
			content: string;
			created_at: number;
		}>;

		const vecScored = vecRows.map((r) => {
			const vec = new Float32Array(r.vec.buffer, r.vec.byteOffset, r.vec.byteLength / 4);
			return { ...r, cosine: cosineSimilarity(queryVector, vec) };
		});
		vecScored.sort((a, b) => b.cosine - a.cosine);
		const vecRank = new Map<string, number>();
		vecScored.forEach((r, i) => vecRank.set(r.id, i + 1));

		// RRF fusion with recency decay.
		const all = new Map<string, { id: string; role: string; content: string; createdAt: number }>();
		for (const r of keywordRows) {
			all.set(r.id, { id: r.id, role: r.role, content: r.content, createdAt: r.created_at });
		}
		for (const r of vecScored) {
			if (!all.has(r.id)) {
				all.set(r.id, { id: r.id, role: r.role, content: r.content, createdAt: r.created_at });
			}
		}

		const RRF_K = 60;
		const fused = [...all.values()].map((e) => {
			const kr = keywordRank.get(e.id);
			const vr = vecRank.get(e.id);
			let score = 0;
			if (kr !== undefined) score += 1 / (RRF_K + kr);
			if (vr !== undefined) score += 1 / (RRF_K + vr);
			score *= 0.5 ** ((now - e.createdAt) / halfLife);
			return { ...e, score };
		});
		fused.sort((a, b) => b.score - a.score);
		return fused.slice(0, topK).map(({ id, role, content, createdAt }) => ({ id, role, content, createdAt }));
	}

	#ftsQuery(query: string): string {
		// Tokenize the raw query into double-quoted phrases for FTS5.
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
