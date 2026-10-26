# Superspace Support AI v2.2

API-only support assistant, designed for small Railway allocations and Cloudflare Workers AI free usage.

## Architecture

- Requests: `POST /api/v1/support/chat` -> hybrid BM25 + ONNX MiniLM embeddings -> source-grounded Cloudflare generation.
- Background knowledge sync: crawl official Superspace help pages, compare article hashes, reuse unchanged chunk embeddings, verify new index, and publish immutable deployment artifacts.
- GitHub Actions syncs every six hours and on `workflow_dispatch`. It commits only changed index artifacts; Railway deploys from the updated GitHub default branch.
- No live website network requests occur during chat. No `demo.html`, web widget or database is needed.
- One Uvicorn worker on Railway. Sessions, quotas and caches are local to this worker. Do not enable multiple replicas without shared storage and distributed rate limiting.

## First-time setup on Windows (PowerShell)

```powershell
Copy-Item .env.example .env
# Add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
python -m pip install -r requirements.txt
python -m scripts.sync_knowledge
python -m scripts.verify --offline
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In **another terminal**:

```powershell
python -m scripts.verify --live
```

`--live` uses the real Cloudflare account and spends free neuron capacity. `--offline` does not call Cloudflare. For just the endpoint: `Invoke-RestMethod http://127.0.0.1:8000/ready`.

## Refreshing content

`python -m scripts.sync_knowledge` will fetch the website, detect additions/edits/removals, reuse unchanged embeddings and replace the index only after validation. It blocks unexpected severe crawl shrinkage. `--skip-crawl` rebuilds from an existing corpus. Changes become visible to the running API **only after redeployment/restart**. GitHub Actions schedules sync every 6 hours, which automatically publishes changed index files to the branch used by Railway. Critical policy edits should trigger a manual workflow run.

## API contract

POST `/api/v1/support/chat` with `{"message":"How do I publish my space?","session_id":null}`. Response includes `answer`, `session_id`, `sources` (title and URL), `status`, `timings_ms`. GET `/health` and `/ready`. No anonymous admin/update endpoint.

## Evaluation

Create `evals/golden.jsonl` from independently reviewed factual help-center answers and URL labels. Start with `evals/golden.example.jsonl` to test the evaluator code, **not** as an accuracy proof. Run:

`python -m evals.evaluate evals/golden.jsonl --enforce`

Target Recall@30 >=95%, Recall@5 >=90%, retrieval P95 <=150 ms, completed answer P95 <=5 s under a representative user mix, subject to Cloudflare availability. TTFT is not measured on this non-streaming API. For load tests run the included `evals/load_test.py`, with realistic rates and protected quota settings.

## Security and ops

- Rotate any previously exposed provider tokens; `.env` never belongs in Git.
- Allow only your production widget origins using `ALLOWED_ORIGINS`.
- Default daily AI cap is a local-process protection, **not** a precise neuron-metering or billing guarantee. Observe daily neuron spend in the Cloudflare dashboard.
- Return 503/429 on overload and budget exhaustion; no fallback to ungrounded answers.
- See `DEPLOYMENT.md` and `RELEASE_REPORT.md` for caveats and Railway setup.
