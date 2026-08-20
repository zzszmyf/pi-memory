/**
 * Retrieval-based context transform: replaces "send the whole history" with
 * "retrieve what matters".
 *
 * Mounted on the agent's transformContext hook. On every request:
 *  1. Mirror any not-yet-stored messages into the SQLite memory.
 *  2. Use the latest user message as the retrieval query.
 *  3. Return: [retrieval notice, relevant historical messages (top-K),
 *     most recent N messages] — everything else stays in SQLite.
 *
 * The transform is idempotent (storing is deduplicated by message id) and
 * configurable via the memory settings object.
 */
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { EmbeddingClientOptions } from "./embedding.ts";
import { EmbeddingClient } from "./embedding.ts";
import type { RetrievalMemorySettings } from "./retrieval-memory.ts";
import { RetrievalMemory } from "./retrieval-memory.ts";

export interface RetrievalTransformOptions {
	sessionId: string;
	dbPath: string;
	settings?: Partial<RetrievalMemorySettings>;
	/** Optional embedding backend (SiliconFlow / any OpenAI-compatible endpoint). */
	embedding?: EmbeddingClientOptions;
	/** Include the retrieval notice message so the model knows context was injected. */
	includeNotice?: boolean;
}

const USER_ROLES = new Set(["user"]);
const SKIP_ROLES = new Set(["system", "developer"]);
const VECTOR_BATCH = 32;

export function createRetrievalTransform(options: RetrievalTransformOptions) {
	const memory = new RetrievalMemory(options.dbPath, options.settings);
	const embedder = options.embedding ? new EmbeddingClient(options.embedding) : null;
	const includeNotice = options.includeNotice ?? true;

	return async (messages: AgentMessage[]): Promise<AgentMessage[]> => {
		// 1. Mirror new messages into SQLite. Dedup uses a deterministic
		//    content hash as the entry id — INSERT OR REPLACE makes
		//    re-sent history a no-op instead of duplicating rows.
		for (const message of messages) {
			if (SKIP_ROLES.has(message.role)) {
				continue;
			}
			memory.store(options.sessionId, message.role, extractText(message));
		}

		if (messages.length === 0) {
			return messages;
		}

		// 1b. Background vectorization: embed entries that lack a vector.
		if (embedder) {
			try {
				const pending = memory.pendingVectorIds(options.sessionId, VECTOR_BATCH);
				if (pending.length > 0) {
					const vecs = await embedder.embed(pending.map((p) => p.content));
					for (let i = 0; i < pending.length; i++) {
						memory.storeVector(pending[i].id, vecs[i]);
					}
				}
			} catch (error) {
				// Embedding is best-effort: fall back to pure BM25 on failure.
				console.error("[memory] vectorization failed, using keyword-only retrieval:", error);
			}
		}

		// 2. Query = latest user message.
		const lastUser = [...messages].reverse().find((m) => USER_ROLES.has(m.role));
		if (!lastUser) {
			return messages;
		}

		const keepRecent = memory.settings.keepRecent;
		const recent = messages.slice(-keepRecent);
		const recentIds = new Set(recent.map((m) => (m as { id?: string }).id).filter(Boolean));

		// 3. Retrieve relevant older messages (hybrid BM25+vector when the
		//    embedder is configured, pure BM25 otherwise).
		const queryText = extractText(lastUser);
		let hits;
		if (embedder) {
			try {
				const [queryVec] = await embedder.embed([queryText]);
				hits = await memory.retrieveHybrid(options.sessionId, queryText, queryVec);
			} catch (error) {
				console.error("[memory] hybrid retrieval failed, using keyword-only:", error);
				hits = memory.retrieve(options.sessionId, queryText);
			}
		} else {
			hits = memory.retrieve(options.sessionId, queryText);
		}
		hits = hits.filter((entry) => !recentIds.has(entry.id));

		if (hits.length === 0) {
			return messages;
		}

		// Retrieved history is injected as user-role context messages (the
		// agent harness convention for RAG-style context injection). Each
		// entry is labeled with a stable index, a role tag, and relative
		// age so the model can weigh and cite it.
		const now = Date.now();
		const retrieved: AgentMessage[] = hits.map((entry, i): AgentMessage => {
			const age = formatAge(now - entry.createdAt);
			const roleTag = ROLE_TAGS[entry.role] ?? entry.role;
			return {
				role: "user",
				content: `<memory id="${i + 1}" role="${roleTag}" age="${age}">\n${entry.content}\n</memory>`,
				timestamp: entry.createdAt,
			};
		});

		// 4. Compose: instruction notice + retrieved context + recent window.
		const notice: AgentMessage[] = includeNotice
			? [
					{
						role: "user",
						content:
							`<retrieved-memory>\n` +
							`The ${hits.length} messages below were retrieved from earlier in this session ` +
							`because they appear relevant to your current task. Use them as reference ` +
							`context; ignore any that turn out to be irrelevant. They are not part of ` +
							`the live conversation that continues after </retrieved-memory>.\n`,
						timestamp: now,
					},
				]
			: [];
		const closer: AgentMessage[] = includeNotice
			? [{ role: "user", content: "</retrieved-memory>", timestamp: now }]
			: [];

		return [...notice, ...retrieved, ...closer, ...recent];
	};
}

const ROLE_TAGS: Record<string, string> = {
	user: "question",
	assistant: "answer",
	toolResult: "tool-output",
	custom: "note",
};

function formatAge(ageMs: number): string {
	const minutes = Math.round(ageMs / 60_000);
	if (minutes < 1) return "just now";
	if (minutes < 60) return `${minutes}m ago`;
	const hours = Math.round(minutes / 60);
	if (hours < 24) return `${hours}h ago`;
	return `${Math.round(hours / 24)}d ago`;
}

function extractText(message: AgentMessage): string {
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") {
		return content;
	}
	if (Array.isArray(content)) {
		return content
			.map((part) => {
				if (typeof part === "string") {
					return part;
				}
				if (part && typeof part === "object" && "text" in part) {
					return String((part as { text: unknown }).text);
				}
				return "";
			})
			.filter(Boolean)
			.join("\n");
	}
	return String(content ?? "");
}
