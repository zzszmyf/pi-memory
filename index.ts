/**
 * Retrieval Memory Extension — three cooperating memory layers:
 *
 * 1. Passive mirror: every message is stored in a per-project SQLite db
 *    (node:sqlite + FTS5). The "context" event replaces the full-history
 *    resend with hybrid retrieval (BM25 + optional embeddings, RRF fusion,
 *    recency decay), so per-request context stays flat no matter how long
 *    the session runs.
 * 2. Active notebook: write_note / search_notes / read_recent_notes tools —
 *    the agent decides what to remember and when to look things up. Notes
 *    persist across sessions.
 * 3. Distillation subagent: a standalone command batches undigested messages
 *    through an LLM into structured notes with merge/delete lifecycle.
 *
 * Configuration (optional): `.pi/memory.json` in the project directory:
 * {
 *   "enabled": true,
 *   "topK": 8,
 *   "keepRecent": 12,
 *   "embedding": {
 *     "enabled": true,
 *     "baseUrl": "https://api.siliconflow.com/v1",
 *     "apiKey": "...",
 *     "model": "Qwen/Qwen3-Embedding-0.6B",
 *     "dimensions": 512
 *   }
 * }
 *
 * Usage: copy this directory to ~/.pi/agent/extensions/retrieval-memory/
 * or a project's .pi/extensions/retrieval-memory/.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { EmbeddingClientOptions } from "./embedding.ts";
import { createMemoryToolDefinitions } from "./memory-tools.ts";
import { createRetrievalTransform } from "./transform-context.ts";

interface MemoryConfig {
	enabled?: boolean;
	topK?: number;
	keepRecent?: number;
	embedding?: EmbeddingClientOptions & { enabled?: boolean };
}

export default function retrievalMemoryExtension(pi: ExtensionAPI) {
	// Extension factories receive no context up front, so initialization is
	// lazy: the first "context" event supplies cwd + session id.
	let initialized = false;
	let transform: ReturnType<typeof createRetrievalTransform> | null = null;
	let sessionId = "";
	let dbPath = "";
	let lastEmbedding: EmbeddingClientOptions | undefined;

	const resolveEmbedding = (config: MemoryConfig): EmbeddingClientOptions | undefined =>
		config.embedding?.enabled !== false && config.embedding?.apiKey
			? {
					baseUrl: config.embedding.baseUrl,
					apiKey: config.embedding.apiKey,
					model: config.embedding.model,
					dimensions: config.embedding.dimensions,
				}
			: undefined;

	const loadConfig = (ctx: ExtensionContext): MemoryConfig => {
		try {
			const { readFileSync } = require("node:fs") as typeof import("node:fs");
			return JSON.parse(readFileSync(`${ctx.cwd}/.pi/memory.json`, "utf8")) as MemoryConfig;
		} catch {
			return {};
		}
	};

	const ensureInitialized = (ctx: ExtensionContext) => {
		if (initialized) return;
		const config = loadConfig(ctx);
		if (config.enabled === false) {
			// Disabled: leave the hook inert.
			initialized = true;
			return;
		}
		sessionId = ctx.sessionManager.getSessionId();
		dbPath = `${ctx.cwd}/.pi/memory.sqlite`;
		lastEmbedding = resolveEmbedding(config);
		transform = createRetrievalTransform({
			sessionId,
			dbPath,
			settings: {
				topK: config.topK ?? 8,
				keepRecent: config.keepRecent ?? 12,
			},
			embedding: lastEmbedding,
		});
		initialized = true;
	};

	// Layer 1: passive mirror + hybrid retrieval on the context event.
	pi.on("context", async (event, ctx) => {
		ensureInitialized(ctx);
		if (!transform) return {};
		return { messages: await transform(event.messages) };
	});

	// Layer 2: active notebook tools (lazily resolved per call).
	const toolDefs = createMemoryToolDefinitions({
		getContext: () => {
			// Resolve embedding lazily from the last loaded config.
			return { sessionId, dbPath, embedding: lastEmbedding };
		},
	});
	for (const tool of toolDefs) {
		pi.registerTool(tool);
	}
}
