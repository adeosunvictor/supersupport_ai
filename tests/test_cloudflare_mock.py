import asyncio
import json
import httpx
import pytest
import llm

@pytest.mark.asyncio
async def test_cloudflare_mocked_provider(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-token')
    captured=[]
    async def respond(request):
        payload=json.loads(request.content)
        captured.append(payload)
        assert payload['stream'] is False
        assert payload['max_tokens'] <= 500
        return httpx.Response(200, json={'success': True,'result':{'response':'Create your space in the dashboard.'}})
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(llm, 'client', lambda: client)
    answer=await llm.answer('How do I publish?', [{'title':'Create a space','url':'https://superspaceapp.com/support/article/example','text':'Open dashboard and create a space.'}], [])
    assert 'dashboard' in answer
    assert len(captured)==1
    await client.aclose()

@pytest.mark.asyncio
async def test_20_parallel_mocked_provider_calls(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-token')
    async def respond(request):
        await asyncio.sleep(0.02)
        return httpx.Response(200,json={'success':True,'result':{'response':'Published.'}})
    client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(llm, 'client', lambda: client)
    calls=await asyncio.gather(*(llm.answer('Publish space?', [{'title':'Publish','url':'https://example.com','text':'Publish it'}],[]) for _ in range(20)))
    assert len(calls)==20 and set(calls)=={'Published.'}
    await client.aclose()
