# AI Incident Copilot

A production-grade AI system for investigating production incidents. Ingest logs, deployment events, and alerts — then query them in natural language to get grounded, LLM-powered root cause analysis.

**Stack:** FastAPI · PostgreSQL + pgvector · sentence-transformers · OpenAI GPT-4o-mini · SQLAlchemy · Alembic

---

## Problem Statement

When production systems fail, engineers face a flood of signals: thousands of log lines, deployment records, and alerts spanning multiple services and time windows. Answering "what happened in the last hour?" requires:

- Manually correlating timestamps across sources
- Filtering noise from signal
- Identifying which events are causally related

Traditional log search is keyword-based — it finds matches, not answers. Vector search alone ignores the time dimension entirely. And handing raw logs directly to an LLM is impractical: context windows are finite and the model has no way to distinguish a CRITICAL error from an INFO log.

This system solves that by combining **temporal filtering**, **semantic retrieval**, and **structured LLM reasoning** into a single deterministic pipeline.

---

## Solution Overview

```
Ingest           →   incident_events table (logs, deployments, alerts, metadata)
                     Normalised embeddings, SHA-256 deduplication, pgvector HNSW index

Query            →   "What happened in payment-service in the last hour?"
                     Time window + service extracted automatically from the query

Hybrid Retrieval →   SQL time-window filter narrows the candidate set
                     pgvector cosine ANN search finds the most semantically relevant events

Investigation    →   Cross-encoder reranker re-scores candidates with full context
                     Quality gate: drops events with rerank_score < threshold
                     Key Signals header: highlights CRITICAL/ERROR events first

LLM Reasoning    →   Structured, grouped prompt sent to GPT-4o-mini
                     LLM is grounded — it answers only from retrieved evidence
                     If no events pass the quality gate, the LLM call is skipped entirely

Answer           →   Grounded explanation + source events + time window + latency breakdown
```

---

## Architecture

### Incident query flow

```
User query: "Why did payment-service fail in the last 2 hours?"
  │
  ▼
extract_time_window(query)          heuristic: "last 2 hours" → (now-2h, now)
  │
  ▼
extract_service(query, db)          substring match against live service registry
  │
  ▼
rewrite_query(query)                expand short keyword queries for better bi-encoder recall
  │
  ▼
search_incidents()                  SQL: WHERE occurred_at BETWEEN start AND end
  │                                      AND service = 'payment-service'
  │                                 pgvector: ORDER BY embedding <=> query_embedding
  │                                 → top 20 candidates (RERANK_TOP_K)
  ▼
rerank_incidents()                  cross-encoder/ms-marco-MiniLM-L-6-v2
  │                                 scores each [query, "[SEVERITY] event content"] pair
  │                                 quality gate: drop if score < RERANK_THRESHOLD (-2.0)
  │                                 → top 5 events (FINAL_TOP_K)
  ▼
build_incident_prompt()             ### Key Signals  (CRITICAL + ERROR events only)
  │                                 ## Errors / Alerts
  │                                 ## Deployments / Changes
  │                                 ## Timeline
  │                                 chronological within each section
  ▼
generate_answer()                   OpenAI GPT-4o-mini, temperature=0.2
  │                                 timeout=30s, max_retries=2, fallback on failure
  ▼
{ answer, sources, time_window, latency }
```

### Two-stage retrieval: why it matters

The bi-encoder (pgvector ANN) encodes query and events independently — fast, but coarse. The cross-encoder reads query and event together in a single forward pass, producing much more accurate relevance scores. Running the cross-encoder over only 20 candidates (not the full table) keeps latency acceptable on CPU.

### Hybrid SQL + vector search

Doing a vector search across all historical events is both slow and wrong — a query about "last hour" should never surface events from last month. Time-window SQL filtering runs first, reducing the candidate set before the vector index is consulted. The embedding then ranks within the window, not across all history.

---

## Project Structure

```
app/
├── main.py                    FastAPI app, logging config, middleware, exception handler
├── api/
│   ├── incidents.py           POST /incidents/ingest/batch  POST /incidents/ask
│   ├── feedback.py            POST /feedback
│   └── analytics.py           GET /analytics/summary
├── core/
│   ├── config.py              pydantic-settings — single source of truth
│   └── request_id.py          ContextVar request ID, middleware, logging filter
├── db/
│   ├── session.py             SQLAlchemy engine + get_db() dependency
│   └── models.py              IncidentEvent, QueryLog, Feedback ORM models
└── services/
    ├── incidents.py           extract_time_window, extract_service, search_incidents, rerank_incidents
    ├── rag.py                 answer_incident_query — pipeline orchestrator
    ├── prompt.py              build_incident_prompt — Key Signals, grouped sections
    ├── ingestion.py           ingest_logs, ingest_events, ingest_pipeline_metadata
    ├── embedding.py           generate_embedding() — all-MiniLM-L6-v2, lru_cache
    ├── reranker.py            get_reranker() — ms-marco cross-encoder, lru_cache
    ├── llm.py                 generate_answer() — OpenAI client, retry, fallback
    ├── evaluator.py           LLM-as-judge: groundedness + relevance + correctness
    └── query_logger.py        background analytics logging to query_logs

alembic/versions/              schema migrations (incident_events, HNSW index, etc.)
tests/                         233 tests — all mocked, no external services required
frontend/index.html            browser UI served at GET /ui
Dockerfile                     python:3.11-slim, CPU-only torch, non-root appuser
docker-compose.yml             PostgreSQL + pgvector + FastAPI
.github/workflows/ci.yml       GitHub Actions: test → deploy to Render
```

---

## Key Features

### Time-aware retrieval
Queries like "what happened in the last hour?" automatically extract a time window. The window bounds a SQL filter that runs before the vector search, so the ANN never scores events outside the relevant time range.

Supported patterns: `last hour`, `last 2 hours`, `last 6 hours`, `last 24 hours / yesterday`, `last week`. Default: 1 hour.

### Hybrid search (SQL + vector)
The `incident_events` table is filtered by `occurred_at`, `service`, and `event_type` at the SQL layer. The pgvector HNSW index then ranks by cosine distance within that filtered set. The combination gives both temporal precision and semantic relevance.

### Deterministic investigation pipeline
No agents, no planning loops. Every query follows the same fixed stages: extract → retrieve → rerank → prompt → generate. The pipeline is auditable, fast, and predictable. The prompt structure is deterministic — grouping into Errors, Deployments, and Timeline is applied consistently regardless of input.

### Reranking + quality gate
The cross-encoder re-scores all 20 candidates with full context awareness. Events with `rerank_score < RERANK_THRESHOLD` are discarded before prompt assembly. If the quality gate removes all candidates, the LLM call is skipped entirely and a canned "no data" response is returned — preventing hallucination on empty context.

### Structured prompt with Key Signals
When CRITICAL or ERROR events are present, a `### Key Signals` header is prepended to the prompt with the top 1–2 severity events and the top semantic match. This directs the LLM's attention before it reads the full chronological context.

### Observability
Every request emits structured log lines with a per-request ID (propagated via `ContextVar` through middleware):
- `incident_retrieval: window=... retrieved=N reranked=M passed_threshold=P avg_sim=...`
- `investigation: timeline=N errors=M deployments=K`
- `llm_usage: model=... prompt=N completion=M total=T`
- `incident_latency: retrieval=...s rerank=...s llm=...s eval=...s total=...s`

### Production resilience
- LLM calls: `timeout=30s`, `max_retries=2`, fallback string on all error paths (no unhandled exceptions reaching the client)
- Global exception handler returns structured JSON `{"error": "...", "request_id": "..."}` with HTTP 500
- Non-root Docker user (`appuser`, uid 1001)
- Deduplication: ingestion uses SHA-256 content hashes — re-submitting the same events is idempotent

---

## System Design Decisions

| Decision | Why |
|---|---|
| **SQL time filter before vector search** | Querying across all historical events is slow and semantically wrong for incident analysis. Time-bounding the candidate set before ANN search gives correct results and scales better. |
| **No pure vector search** | Cosine similarity alone ignores time. An event from last month might be semantically similar to a current query but completely irrelevant. Hybrid filtering is required. |
| **No agents or planning loops** | Agents add non-determinism, latency, and debugging complexity. For incident analysis the investigation steps are known upfront: time → service → retrieve → rerank → reason. A deterministic pipeline is faster and easier to trust under pressure. |
| **Deterministic pipeline first** | The rewriter, time extractor, and service extractor are all heuristic functions — fast, free, testable. An LLM-based planner is a future improvement, not a requirement for the core use case. |
| **Cross-encoder reranker** | Bi-encoders are fast but score query and event independently. The cross-encoder reads both together, producing far more accurate relevance scores at the cost of O(k) inference — acceptable when k=20. |
| **Quality gate** | Discarding low-confidence events before prompt assembly is safer than letting the LLM reason over noisy signal. The LLM safety check (skip if final_hits empty) prevents hallucination on empty context. |
| **Severity prefix for reranker** | The cross-encoder only sees text. Prepending `[CRITICAL]` or `[DEPLOYMENT]` to each candidate before scoring gives the model signal to distinguish high-severity events from verbose INFO logs with similar vocabulary. |
| **`@lru_cache` on both models** | Bi-encoder (~90 MB) and cross-encoder are loaded once per process. Model load is the slow path — cached load makes per-request inference cost negligible. |

---

## Example Flow

**Query:** `"What caused the payment-service outage in the last 2 hours?"`

**Step 1 — Time + service extraction:**
```
window: (now - 2h) → now
service: "payment-service"
```

**Step 2 — Hybrid retrieval:**
```sql
SELECT * FROM incident_events
WHERE occurred_at BETWEEN :start AND :end
  AND service = 'payment-service'
ORDER BY embedding <=> :query_embedding
LIMIT 20
```
Returns 8 candidates (avg_sim=0.7231).

**Step 3 — Rerank + quality gate:**
Cross-encoder scores all 8. 5 pass threshold (-2.0). Top scores: 7.82, 6.41, 4.15, 2.88, -0.54.

**Step 4 — Prompt assembly:**
```
### Key Signals
- [CRITICAL] payment-service @ 2024-01-15T14:23:00Z: Connection pool exhausted...
- [ERROR] payment-service @ 2024-01-15T14:19:00Z: DB query timeout after 30s...

## Errors / Alerts
[Error 1 | 2024-01-15T14:15:00Z | payment-service | log | CRITICAL | score: 7.82]
DB connection pool exhausted: max_connections=20 active=20 waiting=47

## Deployments / Changes
[Deploy 1 | 2024-01-15T14:10:00Z | payment-service | deployment | score: 6.41]
Deployed v2.4.1 — increased DB query concurrency limit

## Timeline
[Event 1 | 2024-01-15T14:12:00Z | payment-service | log | WARNING | score: 4.15]
Slow query detected: avg=1.8s p99=4.2s
```

**Step 5 — LLM answer:**
```
The payment-service outage was caused by connection pool exhaustion, triggered
by a deployment at 14:10 that increased query concurrency without a corresponding
increase in the connection pool limit. DB queries began timing out at 14:19,
and the pool was fully exhausted by 14:23 with 47 requests waiting.
```

**Response also includes:** source events with timestamps and scores, `time_window`, per-stage latency.

---

## API Endpoints

### `POST /incidents/ask`

Query the incident knowledge base with temporal and semantic search.

```bash
curl -X POST http://localhost:8000/incidents/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened in payment-service in the last hour?"}'
```

```json
{
  "answer": "At 14:23 UTC, payment-service experienced connection pool exhaustion...",
  "sources": [
    {
      "id": 42,
      "content": "Connection pool exhausted: max_connections=20 active=20 waiting=47",
      "occurred_at": "2024-01-15T14:23:00Z",
      "service": "payment-service",
      "event_type": "log",
      "severity": "CRITICAL",
      "similarity_score": 0.8821,
      "rerank_score": 7.8241
    }
  ],
  "time_window": {
    "start": "2024-01-15T13:30:00Z",
    "end": "2024-01-15T14:30:00Z"
  }
}
```

**With `"debug": true`** — adds full pipeline internals:

```json
{
  "answer": "...",
  "sources": [...],
  "time_window": {...},
  "rewritten_query": "Explain payment-service failure in detail...",
  "prompt": "### Key Signals\n- [CRITICAL] ...",
  "latency": {
    "retrieve": 0.082,
    "rerank": 0.431,
    "llm": 1.203,
    "eval": 0.0,
    "total": 1.718
  }
}
```

**Optional request fields:**

| Field | Type | Description |
|---|---|---|
| `query` | string | Natural-language question (required) |
| `debug` | bool | Returns prompt, latency, rewritten query |
| `evaluate` | bool | Runs LLM-as-judge (groundedness + relevance) |
| `start_time` / `end_time` | datetime | Override auto-extracted time window |
| `service` | string | Override auto-extracted service filter |
| `event_types` | list[string] | Filter to specific event types: `log`, `deployment`, `alert`, `metadata` |

---

### `POST /incidents/ingest/batch`

Ingest logs, deployment events, and pipeline metadata in one request. Idempotent — re-submitting the same payload returns `ingested=0, skipped=N`.

```bash
curl -X POST http://localhost:8000/incidents/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "service": "payment-service",
        "occurred_at": "2024-01-15T14:23:00Z",
        "content": "Connection pool exhausted: max_connections=20 active=20 waiting=47",
        "severity": "CRITICAL"
      }
    ],
    "events": [
      {
        "service": "payment-service",
        "occurred_at": "2024-01-15T14:10:00Z",
        "content": "Deployed v2.4.1 — increased DB query concurrency limit",
        "event_type": "deployment"
      }
    ],
    "metadata": []
  }'
```

```json
{
  "logs":     {"ingested": 1, "skipped": 0, "total": 1},
  "events":   {"ingested": 1, "skipped": 0, "total": 1},
  "metadata": {"ingested": 0, "skipped": 0, "total": 0},
  "total_ingested": 2,
  "total_skipped": 0,
  "total_received": 2
}
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "db": "connected"}
```

Returns `"db": "failed"` if the database is unreachable (SELECT 1 probe). Used by Render health checks and monitoring.

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3/M4):** Use the ARM64 Homebrew Python to avoid Rosetta + PyTorch crashes:
> `arch -arm64 /opt/homebrew/bin/brew install python@3.12`
> then create the venv with `/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv`

### 2. Configure environment

Create `.env` in the project root:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/incident_db
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
RERANK_TOP_K=20
FINAL_TOP_K=5
RERANK_THRESHOLD=-2.0
MAX_CONTEXT_CHARS=2000
```

### 3. Start PostgreSQL

```bash
docker compose up -d db
```

Starts `pgvector/pgvector:pg16` on port **5433**.

### 4. Run migrations

```bash
alembic upgrade head
```

Creates `incident_events`, `query_logs`, `feedback` tables and the HNSW vector index on `incident_events.embedding`.

> Migrations also create a legacy `documents` table from the original document-RAG iteration. It is no longer used by any code; it remains in migrations so applied histories stay linear. Safe to ignore.

### 5. Seed synthetic data (optional)

```bash
python scripts/seed.py
```

Ingests ~18 synthetic incident events so `/incidents/ask` returns grounded answers on a fresh DB.

### 6. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
```

> On first start, bi-encoder (~90 MB) and cross-encoder models are downloaded from HuggingFace and cached. Subsequent starts are instant.

**Interactive docs:** http://localhost:8000/docs

**UI:** http://localhost:8000/ui

![alt text](<screenshots/Screenshot 1.png>)


### Full stack with Docker

```bash
docker compose up --build
```

Runs migrations automatically and starts the API. Ready when you see:

```
api  | INFO  AI Incident Copilot v2.0.0 ready
```

---

## Deployment on Render

### 1. Provision Postgres

New → PostgreSQL. Copy the **Internal Database URL**.

### 2. Create Web Service

New → Web Service → connect your GitHub repo.

- **Environment:** Docker (auto-detected from `Dockerfile`)
- **Instance type:** Standard (1 GB RAM minimum — models require ~500 MB)
- **Region:** Same as your Postgres instance

### 3. Environment Variables

| Variable | Value |
|---|---|
| `DATABASE_URL` | Internal Database URL from step 1 |
| `OPENAI_API_KEY` | Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `RERANK_THRESHOLD` | `-2.0` |

### 4. Start Command

```
sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info"
```

Migrations run on every deploy — idempotent, already-applied versions are skipped.

### 5. Verify

```bash
curl https://<your-service>.onrender.com/health
# {"status":"ok","db":"connected"}

curl -X POST https://<your-service>.onrender.com/incidents/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what happened in the last hour?"}'
```

![alt text](<screenshots/Screenshot 2.png>)
![alt text](<screenshots/Screenshot 3.png>)
![alt text](<screenshots/Screenshot 4.png>)
![alt text](<screenshots/Screenshot 5.png>)

---

## CI/CD

GitHub Actions at [.github/workflows/ci.yml](.github/workflows/ci.yml):

```
push to main
  │
  ├── test job: checkout → Python 3.11 → install deps (cached) → pytest
  │
  └── deploy job (only if test passes AND push to main):
        curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
```

- Tests run on every push and PR — no external services required (all API calls mocked)
- Deploy triggers only on direct pushes to `main`, never on PRs
- Add `RENDER_DEPLOY_HOOK_URL` to GitHub repo secrets (Settings → Secrets → Actions)

**Local test run:**

```bash
pytest tests/ -v
# 233 passed
```

---

## Testing

All 233 tests run without a real database or OpenAI key. The test suite covers:

| Module | What's tested |
|---|---|
| `test_api.py` | `/health` endpoint |
| `test_rag.py` | `rewrite_query` |
| `test_incidents.py` | `search_incidents`, `rerank_incidents`, `extract_time_window`, `extract_service`, `build_incident_prompt`, `answer_incident_query` |
| `test_ingestion.py` | `ingest_logs`, `ingest_events`, `ingest_pipeline_metadata`, log normalisation |
| `test_query_processor.py` | `classify_query`, `process_query`, time-window and service extraction |
| `test_evaluator.py` | `evaluate_answer`, `_parse_eval_response` |
| `test_feedback_analytics.py` | `/feedback`, `/analytics/summary` |

CI environment variables (`OPENAI_API_KEY=test-key-for-ci`, `DATABASE_URL=postgresql://test:test@localhost:5432/testdb`) satisfy pydantic-settings validation without making real requests.

---

## Future Improvements

| Area | Idea |
|---|---|
| **Service extraction** | Replace substring matching with an LLM-based NER step for multi-service queries and aliases |
| **Time window extraction** | Support absolute timestamps ("on January 15th"), relative calendar references, and timezone-aware parsing |
| **Async ingestion** | Stream log lines directly from Kafka or a webhook rather than batch HTTP ingest |
| **Cross-event correlation** | Use `correlation_id` to group causally related events and surface full incident threads |
| **Embedding quality** | Fine-tune the bi-encoder on incident/log vocabulary for higher recall on noisy log text |
| **Persistent model cache** | Mount a Render Persistent Disk at `/home/appuser/.cache` to avoid re-downloading models on every deploy |
| **Multi-window analysis** | Allow queries that span and compare two time windows, e.g. "compare last hour to same time yesterday" |
