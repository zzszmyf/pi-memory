/**
 * OpenAI-compatible embedding client (SiliconFlow / any /v1/embeddings endpoint).
 *
 * Features:
 * - Batch embedding with an in-memory cache (content-hash keyed) so repeated
 *   history resends don't re-bill the API.
 * - Optional dimension trimming (Matryoshka models) by slicing the vector.
 * - Float32Array output for cheap cosine similarity in the retrieval layer.
 */

import { createHash } from "node:crypto";

export interface EmbeddingClientOptions {
	baseUrl: string;
	apiKey: string;
	model: string;
	/** Optional output dimensions (Matryoshka). Slices the returned vector. */
	dimensions?: number;
}

interface EmbeddingsResponse {
	data: Array<{ embedding: number[] }>;
}

export class EmbeddingClient {
	private readonly options: EmbeddingClientOptions;
	private readonly cache = new Map<string, Float32Array>();

	constructor(options: EmbeddingClientOptions) {
		this.options = options;
	}

	async embed(texts: string[]): Promise<Float32Array[]> {
		const unique: string[] = [];
		const indexOf = new Map<string, number>();
		for (const text of texts) {
			const key = this.#cacheKey(text);
			if (!indexOf.has(key)) {
				indexOf.set(key, unique.length);
				unique.push(text);
			}
		}

		const resolved: Float32Array[] = new Array(texts.length);
		const pending: Array<{ text: string; key: string; originalIndex: number }> = [];
		for (let i = 0; i < texts.length; i++) {
			const key = this.#cacheKey(texts[i]);
			const hit = this.cache.get(key);
			if (hit) {
				resolved[i] = hit;
			} else {
				pending.push({ text: texts[i], key, originalIndex: i });
			}
		}

		if (pending.length > 0) {
			const response = await fetch(`${this.options.baseUrl}/embeddings`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${this.options.apiKey}`,
				},
				body: JSON.stringify({
					model: this.options.model,
					input: pending.map((p) => p.text),
					encoding_format: "float",
					...(this.options.dimensions ? { dimensions: this.options.dimensions } : {}),
				}),
			});
			if (!response.ok) {
				throw new Error(`Embedding request failed: ${response.status} ${await response.text()}`);
			}
			const body = (await response.json()) as EmbeddingsResponse;
			if (body.data.length !== pending.length) {
				throw new Error("Embedding response count mismatch");
			}
			for (let i = 0; i < pending.length; i++) {
				const vec = new Float32Array(body.data[i].embedding);
				this.cache.set(pending[i].key, vec);
				resolved[pending[i].originalIndex] = vec;
			}
		}

		return resolved;
	}

	#cacheKey(text: string): string {
		return createHash("sha1").update(text).digest("hex").slice(0, 40);
	}
}

/** Cosine similarity between two float vectors. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	const n = Math.min(a.length, b.length);
	for (let i = 0; i < n; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (na === 0 || nb === 0) return 0;
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
