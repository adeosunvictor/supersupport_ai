# Superspace v2.2: 10-user capacity patch

This update preserves the `/api/v1/support/chat` request and response schema and adds two optional `timings_ms` fields: `queue` and `provider`.

## Settings for a first 10-user local load test

```env
MAX_OUTPUT_TOKENS=300
MAX_CONCURRENT_RETRIEVAL=2
MAX_CONCURRENT_INFERENCE=3
MAX_INFERENCE_INFLIGHT=12
INFERENCE_QUEUE_TIMEOUT_SECONDS=45
RATE_LIMIT_PER_MINUTE=100
```

These are **test settings**, not guarantees. Use a suitable `MAX_AI_REQUESTS_PER_DAY` based on available Cloudflare neurons. Larger values do not increase Cloudflare's quota. For realistic testing, use 10 distinct messages and keep the Cloudflare model the same.

With three provider slots, ten simultaneous uncached questions may require roughly four consecutive inference waves. If each wave takes 3 seconds, later responses can take around 12 seconds even when all succeed. The five-second P95 target cannot be guaranteed with this free provider and 10 simultaneous LLM calls.

`MAX_INFERENCE_INFLIGHT` is the maximum admitted provider requests including queued ones. Additional traffic returns 503 quickly with `Retry-After: 10`. Admitted requests wait up to `INFERENCE_QUEUE_TIMEOUT_SECONDS` for a provider slot, then receive 503. This bounds memory pressure instead of allowing an unlimited queue.

## Verify

1. Copy your actual `.env` credentials and generated `data/index` from the previous v2.2 folder. Neither is included in this archive.
2. `python -m pytest -q tests`
3. `python -m uvicorn main:app --reload`
4. In a second terminal: `python -m evals.load_test --users 10 --requests-per-user 1 --message mixed`

Observe `queue_p95_ms`, `provider_p95_ms`, `response_p95_ms`, and non-200 statuses. If provider times balloon or 503s persist, reduce concurrent Cloudflare calls; don't increase parallelism blindly. To test 10 customers conversationally, keep session IDs separate and use multiple turns per user.

## Deployment caution

The rate limiter, sessions, and AI budget are process-local. This design assumes one Railway worker and one replica. Enforce budget and abuse controls centrally before deploying multiple replicas. Local tests simulate Cloudflare responses and do not certify live inference quality, 10-user latency, or free-tier sustainability. Keep credentials out of git.
