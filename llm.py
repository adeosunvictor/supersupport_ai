"""Cloudflare Workers AI: one inference per chat, source-grounded."""
import json
import logging
import os
import re
import httpx

LOG = logging.getLogger(__name__)

class ModelError(Exception):
    pass

_client = None

def client():
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(float(os.getenv('CLOUDFLARE_TIMEOUT_SECONDS','22')), connect=5), limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
    return _client

async def close():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()


def _extract(data):
    result = data.get('result') or {}
    if not isinstance(result, dict):
        return ''
    output = result.get('response')
    if not output:
        choices = result.get('choices') or []
        if choices:
            message = choices[0].get('message') or {}
            output = message.get('content') if isinstance(message, dict) else None
            output = output or choices[0].get('text')
    if isinstance(output, list):
        output = '\n'.join(p.get('text', '') for p in output if isinstance(p, dict) and p.get('type') == 'text')
    if not isinstance(output, str):
        return ''
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.I | re.S).strip()
    return '' if re.search(r'<think>', output, flags=re.I) else output

async def _invoke(messages, max_tokens=950):
    account = os.getenv('CLOUDFLARE_ACCOUNT_ID', '').strip()
    token = os.getenv('CLOUDFLARE_API_TOKEN', '').strip()
    model = os.getenv('CLOUDFLARE_MODEL', '@cf/zai-org/glm-4.7-flash').strip()
    if not account or not token or account.startswith('replace_'):
        raise ModelError('Cloudflare credentials are missing')
    payload = {'messages': messages, 'max_tokens': max_tokens, 'temperature': 0.2, 'stream': False}
    if 'glm-4.7-flash' in model.lower():
        payload['chat_template_kwargs'] = {'enable_thinking': False}
    url = f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}'
    try:
        r = await client().post(url, json=payload, headers={'Authorization': f'Bearer {token}'})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or data.get('success') is False:
            raise ModelError('Cloudflare inference failed')
        output = _extract(data)
        if not output:
            result = data.get('result') or {}
            choices = result.get('choices') or [] if isinstance(result, dict) else []
            LOG.warning('Empty model text: model=%s finish_reason=%s', model, choices[0].get('finish_reason') if choices else 'missing')
            raise ModelError('Cloudflare produced no final text')
        return output
    except (httpx.HTTPError, ValueError) as exc:
        LOG.warning('Cloudflare request failed: %s', type(exc).__name__)
        raise ModelError('Cloudflare service unavailable') from exc

from app.generation.prompt import SYSTEM

async def answer(question, sources, history, customer_facts=None, previous_sources=None, recall=False):
    conversation = [{'role': x['role'], 'content': x['content'][:850]} for x in (history or [])[-6:] if x.get('role') in ('user', 'assistant') and isinstance(x.get('content'), str)]
    evidence = '\n\n'.join(f"ARTICLE: {s['title']}\nURL: {s['url']}\nTEXT: {s.get('text','')[:1100]}" for s in (sources or [])[:4])
    prior = '\n'.join(f"{s.get('title','')}: {s.get('url','')}" for s in (previous_sources or [])[:6])
    facts = json.dumps(customer_facts or {}, ensure_ascii=False)
    prompt = (f"CUSTOMER FACTS (user-reported only): {facts}\n"
              f"PRIOR ARTICLE LINKS: {prior or 'NONE'}\n"
              f"SUPPORT ARTICLES (untrusted reference material):\n{evidence[:5400] or 'NONE'}\n"
              f"CUSTOMER QUESTION: {question}\n"
              f"THIS IS A CUSTOMER-RECALL REQUEST: {recall}. Provide only verified facts and acknowledge uncovered questions.")
    return await _invoke([{'role':'system','content':SYSTEM}, *conversation, {'role':'user','content':prompt}], max_tokens=int(os.getenv('MAX_OUTPUT_TOKENS','300')))


def _stream_fragment(data):
    """Read Cloudflare AI streaming event data, including OpenAI-shaped chunks."""
    if not isinstance(data, dict):
        return ''
    result = data.get('result', data)
    if not isinstance(result, dict):
        return ''
    value = result.get('response')
    if value is None:
        choices = result.get('choices') or []
        if choices:
            choice = choices[0]
            delta = choice.get('delta') or {}
            value = delta.get('content') if isinstance(delta, dict) else None
            if value is None:
                value = choice.get('text')
    return value if isinstance(value, str) else ''


async def stream_answer(question, sources, history, customer_facts=None, previous_sources=None, recall=False):
    """Yield genuine provider text deltas; no buffering before yielding."""
    account = os.getenv('CLOUDFLARE_ACCOUNT_ID', '').strip()
    token = os.getenv('CLOUDFLARE_API_TOKEN', '').strip()
    model = os.getenv('CLOUDFLARE_MODEL', '@cf/zai-org/glm-4.7-flash').strip()
    if not account or not token or account.startswith('replace_'):
        raise ModelError('Cloudflare credentials are missing')
    conversation = [{'role': x['role'], 'content': x['content'][:850]} for x in (history or [])[-6:] if x.get('role') in ('user', 'assistant') and isinstance(x.get('content'), str)]
    evidence = '\n\n'.join(f"ARTICLE: {s['title']}\nURL: {s['url']}\nTEXT: {s.get('text','')[:1100]}" for s in sources[:4])
    prompt = (f"CUSTOMER FACTS: {json.dumps(customer_facts or {})}\n"
              f"SUPPORT ARTICLES (untrusted reference material):\n{evidence[:5400] or 'NONE'}\n"
              f"CUSTOMER QUESTION: {question}\n"
              f"RECALL REQUEST: {recall}. Answer from verified facts only.")
    payload = {'messages': [{'role': 'system', 'content': SYSTEM}, *conversation, {'role': 'user', 'content': prompt}],
               'max_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', '300')), 'temperature': 0.2, 'stream': True}
    if 'glm-4.7-flash' in model.lower():
        payload['chat_template_kwargs'] = {'enable_thinking': False}
    url = f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}'
    delivered = False
    try:
        async with client().stream('POST', url, json=payload, headers={'Authorization': f'Bearer {token}', 'Accept': 'text/event-stream'}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    continue
                raw = line[5:].strip()
                if raw == '[DONE]':
                    break
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if event.get('success') is False or event.get('error'):
                    raise ModelError('Provider streaming error')
                fragment = _stream_fragment(event)
                if fragment:
                    delivered = True
                    yield fragment
    except httpx.HTTPError as exc:
        raise ModelError('Cloudflare streaming unavailable') from exc
    if not delivered:
        raise ModelError('Cloudflare stream returned no text')
