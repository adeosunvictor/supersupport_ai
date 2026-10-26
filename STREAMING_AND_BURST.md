# Superspace Support AI v2.2, streaming and burst optimization

## Scope
- `POST /api/v1/support/chat` remains backward-compatible, non-streaming JSON. Public first-turn answers now use a bounded evidence-fingerprinted cache.
- `POST /api/v1/support/chat/stream` is the new POST **SSE** API. Send the same JSON request as the standard chat endpoint. The client must parse event names `meta`, `delta`, `done`, and `error`.
- The stream sends real Cloudflare text fragments with `stream: true`. It doesn't invent a TTFT measurement on the nonstreaming endpoint.
- Repeated concurrent *public, stateless* questions are single-flighted so only one reaches Cloudflare. Followers get the full answer when the first call completes.
- The cache key covers question, article URLs/titles/**text**, model and output cap. An article change invalidates the relevant cached entry automatically. Cache is deliberately disabled for conversation history, user-specific questions and private details.
- In-memory cache and single-flight registry are **per process**. For multi-replica deployment, use shared Redis before claiming cross-replica deduplication. Restart clears the cache; it is not persistent.

## Local commands
1. Copy `.env.example` to `.env`, fill Cloudflare credentials. Use your already built v2.2 FastEmbed index in `data/index`.
2. `python -m pytest -q tests`
3. `python -m uvicorn main:app --host 127.0.0.1 --port 8000`
4. In a new terminal: `python -m evals.burst_test --users 100 --mode repeat`
5. To test a mix of questions: `python -m evals.burst_test --users 100 --mode mixed`
6. **Real uncached expensive test**: `python -m evals.burst_test --users 100 --mode unique` (uses Cloudflare quota, may generate 503 under free-tier capacity and will not meet 5s without more inference throughput).
7. Older JSON endpoint: `python -m evals.load_test --users 10 --requests-per-user 1`

## Expected SSE event example
```
event: meta
data: {"session_id":"...","sources":[...]}

event: delta
data: {"text":"Open My Spaces"}

event: done
data: {"session_id":"...","sources":[...],"timings_ms":{"retrieval":10,"queue":0,"provider":2000,"total":2100,"cache_hit":false}}
```
The event `error` can arrive after HTTP 200 because SSE headers are already sent. The frontend must treat `done` as success and `error` as failure. Stream readers should use `fetch` POST, not native `EventSource` (GET-only).

## Capacity / cost interpretation
- A test with 100 repeated questions is a **cache and single-flight test**, NOT a 100-unique-inference provider benchmark.
- In cold unique load, only `MAX_CONCURRENT_INFERENCE` concurrent provider requests are executed; up to `MAX_INFERENCE_INFLIGHT` calls are admitted. The remainder fail fast and should be retried with backoff. Scaling inference to 100 distinct questions within 5-10 seconds requires provider throughput and budget evidence.
- Railway Free memory must be tested with the ONNX embedding runtime before production.
- Cloudflare free Workers AI neuron allocation is limited. `MAX_AI_REQUESTS_PER_DAY` is a local request guard, not a neuron meter.
- Rate limiting is IP-based and process-local. Tests on localhost sharing one IP should set `RATE_LIMIT_PER_MINUTE=1000` in **development only**; restore a safe production setting afterward.
- Source-grounded evaluation, Recall@30 and real-provider P95 have NOT been verified by the mocked tests. Those remain deployment release gates.

## Suggested local settings for repeated burst testing
```
MAX_OUTPUT_TOKENS=300
MAX_CONCURRENT_INFERENCE=3
MAX_INFERENCE_INFLIGHT=12
INFERENCE_QUEUE_TIMEOUT_SECONDS=45
ANSWER_CACHE_TTL_SECONDS=3600
ANSWER_CACHE_MAX_ITEMS=512
RATE_LIMIT_PER_MINUTE=1000
MAX_AI_REQUESTS_PER_DAY=60
```
Do not assume a higher app request limit increases Cloudflare's external allowance.
