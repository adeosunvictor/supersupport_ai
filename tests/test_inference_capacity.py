"""Offline admission/queue tests; these do not benchmark real Cloudflare."""
import asyncio
import concurrent.futures
import time

import pytest
from fastapi.testclient import TestClient
import main


class StaticRetriever:
    bm25 = True
    records = [{'title': 'Create a space', 'url': 'https://superspaceapp.com/support/article/create-your-first-space', 'text': 'Click New Space'}]

    async def retrieve_many(self, queries, top_k=5):
        return self.records


@pytest.fixture
def test_app(monkeypatch):
    monkeypatch.setattr(main, 'retriever', StaticRetriever())
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE', '1000')
    monkeypatch.setenv('MAX_AI_REQUESTS_PER_DAY', '1000')
    monkeypatch.setenv('INFERENCE_QUEUE_TIMEOUT_SECONDS', '3')
    main.max_concurrent = asyncio.Semaphore(3)
    main.inference_capacity = asyncio.Semaphore(12)
    main.sessions.clear()
    main.locks.clear()
    main.limits.clear()
    main.ai_budget.update(day=None, used=0)
    main.app.router.on_startup.clear()
    yield main.app


def test_ten_simultaneous_users_complete(test_app, monkeypatch):
    async def mock_cloudflare(*args, **kwargs):
        await asyncio.sleep(0.09)
        return 'To create a space, select New Space.'

    monkeypatch.setattr(main, 'answer', mock_cloudflare)
    with TestClient(test_app) as client:
        def request(i):
            response = client.post('/api/v1/support/chat', json={'message': f'How do I create space {i}?'})
            return response.status_code, response.json()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(request, range(10)))
    assert all(status == 200 for status, _ in results)
    assert any(payload['timings_ms']['queue'] > 30 for _, payload in results)
    assert all('provider' in payload['timings_ms'] for _, payload in results)


def test_budget_charged_only_for_admitted_requests(test_app, monkeypatch):
    monkeypatch.setenv('MAX_AI_REQUESTS_PER_DAY', '1')

    async def mock_cloudflare(*args, **kwargs):
        return 'Verified answer.'

    monkeypatch.setattr(main, 'answer', mock_cloudflare)
    with TestClient(test_app) as client:
        a = client.post('/api/v1/support/chat', json={'message': 'Create a space?'})
        b = client.post('/api/v1/support/chat', json={'message': 'Publish a space?'})
    assert a.status_code == 200
    assert b.status_code == 429
    assert main.ai_budget['used'] == 1
