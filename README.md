# pi-memory

Retrieval memory extension for [pi](https://github.com/earendil-works/pi) — three cooperating memory layers that keep per-request context flat no matter how long a session runs.

## Layers

1. **Passive mirror** — every message is stored in a per-project SQLite database (built-in `node:sqlite` + FTS5, zero native deps). The `context` event replaces the full-history resend with hybrid retrieval: BM25 keywords + optional embeddings (any OpenAI-compatible endpoint, e.g. SiliconFlow Qwen3-Embedding), RRF fusion, recency decay. Embedding failures degrade gracefully to keyword-only.

2. **Active notebook** — three tools so the agent decides what to remember and when to look things up:
   - `write_note` — record a fact, decision, convention, gotcha, or progress
   - `search_notes` — hybrid retrieval over notes
   - `read_recent_notes` — chronological window (temporal locality)

3. **Distillation subagent** — an independent command that batches undigested messages through an LLM into structured notes (preference / decision / convention / gotcha / progress) with merge/delete lifecycle.

## Install

```bash
pi install github:zzszmyf/pi-memory
```

Or copy this directory to `~/.pi/agent/extensions/pi-memory/`.

## Configure

Create `.pi/memory.json` in your project directory:

```json
{
  "enabled": true,
  "topK": 8,
  "keepRecent": 12,
  "embedding": {
    "enabled": true,
    "baseUrl": "https://api.siliconflow.com/v1",
    "apiKey": "sk-...",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "dimensions": 512
  }
}
```

Everything is optional; without `embedding` the extension runs keyword-only.

## Distillation

```bash
node --experimental-strip-types distill-cli.mts <db-path> <llm-base-url> <model> [api-key]
```

(Standalone CLI is in progress — the Distiller class in `distill.ts` is ready to use.)

## Validation

- 20-round stress session: 2/2 cross-round recall tests, context stayed ~12k tokens while the store grew to 343 entries
- 10-round vLLM learning session: recall produced incremental reasoning (new derivations), not repetition
- Distillation on the learning session: 343 raw messages → 267 structured notes, 7 superseded notes deleted
