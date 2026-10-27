import os
import pytest
from fastapi.testclient import TestClient
import main

class DummyRetriever:
    records=[{'url':'https://superspaceapp.com/support/article/publish','title':'Publish a space','text':'Click Publish'}]
    bm25=True
    async def retrieve_many(self,queries,top_k=5):return self.records

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main,'retriever',DummyRetriever())
    monkeypatch.setattr(main,'answer',fake_answer)
    main.sessions.clear();main.locks.clear();main.limits.clear();main.ai_budget.update(day=None,used=0)
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE','1000')
    monkeypatch.setenv('MAX_AI_REQUESTS_PER_DAY','1000')
    # Avoid startup actual index loading because mock has no load method.
    main.app.router.on_startup.clear()
    with TestClient(main.app) as c:yield c

async def fake_answer(message,sources,history,**kwargs):
    return 'You can publish a space from the Superspace dashboard.'

def test_chat_contract_and_citations(client):
    response=client.post('/api/v1/support/chat',json={'message':'How do I publish a space?'})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['answer'] and result['session_id']
    assert result['sources'][0]['url'].startswith('https://superspaceapp.com/support/')
    assert result['timings_ms']['total']>=0

def test_api_only_no_demo(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Superspace Support AI"

    assert client.get("/ready").status_code == 200

def test_five_parallel_chats(client):
    import concurrent.futures
    def one(i):return client.post('/api/v1/support/chat',json={'message':f'How do I publish space number {i}?'}).status_code
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        assert list(pool.map(one,range(5)))==[200]*5

def test_rate_limit(client,monkeypatch):
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE','1')
    assert client.post('/api/v1/support/chat',json={'message':'publish a space'}).status_code==200
    assert client.post('/api/v1/support/chat',json={'message':'publish another space'}).status_code==429
