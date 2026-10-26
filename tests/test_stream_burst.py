import asyncio
import json
import pytest
import httpx
import main
from app.answer_cache import answer_cache, AnswerCache
from llm import _stream_fragment

class FakeRetriever:
    bm25 = True
    records = [{'title': 'Create your first space', 'url':'https://superspaceapp.com/support/article/create-your-first-space', 'text':'Open My Spaces, then choose New Space.'}]
    async def retrieve_many(self, queries):
        await asyncio.sleep(.001)
        return list(self.records)


def parse_sse(raw):
    events = []
    for block in raw.split('\n\n'):
        if not block.strip():
            continue
        parts = dict(line.split(': ',1) for line in block.splitlines() if ': ' in line)
        if 'event' in parts and 'data' in parts:
            events.append((parts['event'], json.loads(parts['data'])))
    return events

@pytest.mark.asyncio
async def test_hundred_identical_streaming_requests_single_flight(monkeypatch):
    monkeypatch.setattr(main, 'retriever', FakeRetriever())
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE','1000')
    monkeypatch.setenv('MAX_AI_REQUESTS_PER_DAY','1000')
    main.sessions.clear(); main.locks.clear(); main.limits.clear()
    main.stream_flights.clear(); answer_cache.clear(); main.ai_budget.update(day=None,used=0)
    calls = 0
    async def mock_stream(*args,**kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(.03)
        yield 'Click New Space.'
        await asyncio.sleep(.01)
        yield ' Add your details.'
    monkeypatch.setattr(main, 'stream_answer', mock_stream)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport,base_url='http://test') as client:
        async def ask():
            r = await client.post('/api/v1/support/chat/stream', json={'message':'How do I create a space?'})
            return r.status_code, parse_sse(r.text)
        results = await asyncio.gather(*(ask() for _ in range(100)))
    assert all(status == 200 for status,_ in results)
    assert all(any(event == 'done' for event,_ in events) for _,events in results)
    assert calls == 1
    assert main.ai_budget['used'] == 1
    assert not main.stream_flights

@pytest.mark.asyncio
async def test_cache_invalidation_and_private_question():
    cache=AnswerCache()
    sources=[{'url':'https://superspaceapp.com/support/article/x','title':'Guide','text':'Original policy'}]
    key=cache.key('How do I create a space?',sources,[],{})
    assert key
    cache.put(key,'Original answer')
    assert cache.get(key)=='Original answer'
    sources[0]['text']='Changed policy'
    assert cache.get(cache.key('How do I create a space?',sources,[],{})) is None
    assert cache.key('What is my account number?',sources,[],{}) is None
    assert cache.key('How do I create a space?',sources,[{'role':'user','content':'x'}],{}) is None


def test_cloudflare_stream_event_parser():
    assert _stream_fragment({'response':'Hello'})=='Hello'
    assert _stream_fragment({'result':{'response':'Hi'}})=='Hi'
    assert _stream_fragment({'choices':[{'delta':{'content':'There'}}]})=='There'
