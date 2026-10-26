# v2.2 Validation Report

## Scope
- Code-only automated tests exercise BM25 ordering, vector ranking, immutable snapshots, incremental re-embedding, malformed crawl protection, no demo route, API contract, five simulated concurrent requests, 429 behavior.
- No real Cloudflare credentials were supplied, and the current execution environment cannot download FastEmbed or reach Superspace's support article pages. Real answer correctness, Recall@30 and deployed P95 **cannot be confirmed** here.
- The prior v2.0 benchmark (10 local users, 50 requests) observed 20 success / 30 HTTP 429; P95 16.84 seconds among successful requests. **That was v2.0, not v2.2.**

## Release gates that must be measured on your machine
- Crawl count at least 100 and ideally all 122 expected articles.
- `python -m scripts.verify --offline` passes.
- `python -m scripts.verify --live` returns a real cited answer.
- Manually validate a labeled dataset (>=100 questions) and run `python -m evals.evaluate evals/golden.jsonl --enforce`.
- Load test 5 independent sessions, observe 0 unexpected HTTP errors and track P50/P95 against targets.
- Verify Railway `/ready`, memory usage, and Cloudflare neuron allocation.

This is a deployable implementation, not a measured guarantee of five-second P95 or 95% Recall@30.
