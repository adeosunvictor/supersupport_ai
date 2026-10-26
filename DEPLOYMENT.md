# Railway deployment, Superspace AI v2.2

1. Build the new FastEmbed index locally with `python -m scripts.sync_knowledge`. The old v2.0 sentence-transformer vectors are incompatible with the new runtime.
2. Commit `data/corpus.json` and `data/index/{chunks.json,vectors.npy,manifest.json}` with the source code to a **private** GitHub repository, without `.env` or credentials.
3. Create one Railway service from the repository; Railway uses `Dockerfile` and `railway.json`. Add environment values `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_MODEL`, and `ALLOWED_ORIGINS`. Keep one replica/worker.
4. Ensure `/ready` returns 200 on the Railway HTTPS domain. Run `python -m scripts.verify --live --url https://YOUR-SERVICE.up.railway.app` on your local PC.
5. GitHub repository Settings > Actions > General: allow workflows write access to contents so scheduled `sync_knowledge.yml` can commit changed index artifacts. Configure Railway to deploy branch updates. Use Actions > Sync Superspace knowledge > Run workflow to force sync after critical edits.
6. Keep the last deployment available for rollback. Review Cloudflare neuron usage, Railway RAM, 429 counts, HTTP P95 and error rate.

**Important**: Railway Trial can allow 1 GB RAM but the ongoing Free plan is capped at 512 MB. ONNX is chosen to reduce memory, but peak RAM and 5-user concurrent P95 need testing on your actual plan. The $5 trial is a credit, not a reserved unlimited server. Heavy inference remains on Cloudflare.

**Known tradeoffs**: sessions and quota counters are process-local; a restart forgets conversation memory and quota state. No multi-replica scaling. Scheduled GitHub Actions are not exact-time guarantees, and failed crawls retain the last verified index. Safe publication requires the index update and deploy to complete. To meet strict multi-instance SLAs, add shared state and a durable budget counter.
