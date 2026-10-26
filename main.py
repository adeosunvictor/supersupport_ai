"""Anonymous Superspace support API, v1.24: evidence-led and latency-conscious."""
import asyncio
import logging
import os
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import json
from app.answer_cache import answer_cache
from llm import stream_answer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm import ModelError, answer, close as close_llm
from app.retrieval.hybrid import HybridRetriever, IndexNotReady

load_dotenv()
LOG = logging.getLogger(__name__)
app = FastAPI(title='Superspace Support AI', version='2.2.0')
@app.get("/")
async def root():
    return {
        "service": "Superspace Support AI",
        "version": "2.2",
        "status": "online",
        "documentation": "/docs"
    }
origins = [s.strip() for s in os.getenv('ALLOWED_ORIGINS','http://localhost:8000').split(',') if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET','POST'], allow_headers=['Content-Type'])
retriever = HybridRetriever(os.getenv('INDEX_DIR','data/index'))
sessions, locks = {}, {}
limits = defaultdict(deque)
from datetime import datetime, timezone
ai_budget = {'day': None, 'used': 0}
budget_lock = asyncio.Lock()

# Keep provider concurrency low, while admitting a bounded burst of customers.
# A capacity slot includes both calls in progress and requests waiting for inference.
max_concurrent = asyncio.Semaphore(max(1, int(os.getenv('MAX_CONCURRENT_INFERENCE', '3'))))
inference_capacity = asyncio.Semaphore(max(1, int(os.getenv('MAX_INFERENCE_INFLIGHT', '12'))))
stream_flights = {}
stream_flights_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    session_id: str | None = Field(default=None, max_length=80)

@app.on_event('startup')
async def startup():
    # Fail readiness rather than silently falling back to live crawling.
    await asyncio.to_thread(retriever.load)

@app.on_event('shutdown')
async def shutdown():
    await close_llm()

@app.get('/health')
def health():
    return {'status':'ok','version':'2.2.0'}

@app.get('/ready')
def ready():
    if retriever.bm25 is None: raise HTTPException(503,detail='Index not loaded')
    return {'status':'ready','version':'2.2.0','chunks':len(retriever.records)}

def prohibited_secret_request(message):
    # Narrow short-circuit: security discussions about API keys are allowed.
    wants = r'(?:show|reveal|print|give|share|expose|disclose|dump|tell me|output)'
    secret = r'(?:system\s*prompt|hidden\s*instructions?|developer\s*instructions?|cloudflare\s*(?:api\s*)?(?:key|token)|api\s*(?:key|token)|secret\s*key|credentials?|\.env\b)'
    return bool(re.search(rf'(?i)\b{wants}\b.{{0,130}}\b{secret}', message) or re.search(rf'(?i)\b{secret}\b.{{0,80}}\b{wants}\b', message))

def smalltalk(message):
    return bool(re.fullmatch(r'(?i)\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you|okay thanks)[!. ]*\s*', message))

def recall_request(message):
    return bool(re.search(r'(?i)\b(?:remember|remind me|earlier|we discussed|i mentioned|what did (?:i|you) say|before we finish)\b', message))

def source_request(message):
    return bool(re.search(r'(?i)(?:where did you get|show me (?:the )?(?:official )?(?:sources|articles)|official articles|your sources|where is that documented)',message))

def extract_customer_facts(message, previous):
    """Conservative, bounded facts: only explicit user reports, never model-produced claims."""
    facts = dict(previous or {})
    match = re.search(r'(?i)\b(?:i run|i own|my business is)\s+((?:one|two|three|four|five|\d+)\s+)?([\w-]+(?:\s+[\w-]+){0,2})\s+(studios?|spaces?|venues?|shops?|salons?)\b',message)
    if match:
        facts['business_description'] = match.group(0)[:120]
        if match.group(1): facts['business_count'] = match.group(1).strip()
        facts['business_type'] = (match.group(2) + ' ' + match.group(3)).strip()
    elif re.search(r'(?i)\bi (?:run|own)\s+(?:a|an)\s+([\w-]+\s+){0,3}(studio|shop|salon|venue|space)\b',message):
        facts['business_description'] = re.search(r'(?i)\bi (?:run|own)\s+(?:a|an)\s+([\w-]+\s+){0,3}(studio|shop|salon|venue|space)\b',message).group(0)
    if re.search(r'(?i)\b(?:i (?:currently )?manage bookings|my bookings (?:are|were) managed)\s+(?:through|via|on)\s+whatsapp\b',message):
        facts['booking_method'] = 'WhatsApp'
    if re.search(r'(?i)\bcustomers? sometimes forget to pay\b',message):
        facts['challenge_payment'] = 'Customers sometimes forget to pay'
    if re.search(r'(?i)\bchecking availability manually\b',message):
        facts['challenge_availability'] = 'Manual availability checking'
    return facts

def search_queries(message, facts, history):
    """Break compound customer questions into focused, fast retrieval queries."""
    text = re.sub(r'\s+',' ',message).strip()
    topics = {
        'pricing': r'\b(pricing|plans?|subscription|monthly|fees?|costs?)\b',
        'payments': r'\b(paid|payments?|refund|unapplied|booking (?:not|isn.t) paid)\b',
        'payout': r'\b(payout|withdraw|settlement|earnings|five minutes)\b',
        'setup': r'\b(create (?:my|a|first)|set up|publish|drafts?)\b',
        'photos': r'\b(photos?|pictures?|images?)\b',
        'availability': r'\b(availability|operating hours|calendar|schedule)\b',
        'equipment': r'\b(equipment|add-ons?|inventory)\b',
        'website': r'\b(own website|booking widget|embed|showcase|booking page)\b',
        'spaces': r'\b(multiple studios|two studios|both studios|multiple spaces)\b',
    }
    topic_queries = {
        'pricing':'Superspace plans features pricing limits number of spaces team members',
        'payments':'Superspace booking payment status unapplied payments help',
        'payout':'Superspace payout earnings withdraw timing payout account',
        'setup':'Superspace create your first space draft publish',
        'photos':'Superspace upload manage photos space',
        'availability':'Superspace operating hours availability settings',
        'equipment':'Superspace attach equipment addons to spaces',
        'website':'Superspace showcase link own website booking widget',
        'spaces':'Superspace manage multiple spaces locations plans',
    }
    detected = [topic_queries[k] for k, pattern in topics.items() if re.search(pattern,text,re.I)]
    if detected:
        return detected[:5]
    if len(text.split()) < 16 and history:
        last = next((x['content'] for x in reversed(history) if x['role']=='user' and not recall_request(x['content'])), '')
        if last and re.search(r'(?i)\b(it|that|those|this|again|later)\b',text):
            return [(last[:95] + ' ' + text)[:180]]
    return [text[:180]]

@app.post('/api/v1/support/chat')
async def chat(data: ChatRequest, request: Request):
    start = time.monotonic()
    now = time.monotonic()
    ip = request.client.host if request.client else 'unknown'
    hits = limits[ip]
    while hits and hits[0] < now - 60:
        hits.popleft()
    if len(hits) >= max(1,int(os.getenv('RATE_LIMIT_PER_MINUTE','20'))):
        raise HTTPException(429,detail='Too many requests. Please try again shortly.')
    hits.append(now)
    key = data.session_id if data.session_id in sessions and sessions[data.session_id]['expires'] > now else str(uuid.uuid4())
    async with locks.setdefault(key, asyncio.Lock()):
        saved = sessions.get(key,{})
        history = list(saved.get('history',[]))
        previous_sources = list(saved.get('sources',[]))
        facts = extract_customer_facts(data.message, saved.get('facts',{}))
        message = data.message.strip()
        sources = []
        retrieval_ms = 0.0
        inference_ms = 0.0
        queue_ms = 0.0
        provider_ms = 0.0
        cache_hit = False
        try:
            if prohibited_secret_request(message):
                response = 'I can help with Superspace support, but I can’t share private instructions, API tokens or credentials.'
                status = 'answered'
            else:
                is_recall = recall_request(message) and not re.search(r'(?i)\b(prices?|plans?|payout|refund|guarantee)\b',message)
                is_casual = smalltalk(message)
                wants_sources = source_request(message)
                # A source request often includes a new factual question. Retrieve for both.
                if not is_casual and not is_recall:
                    queries = search_queries(message, facts, history)
                    t_retrieval=time.monotonic()
                    sources = await retriever.retrieve_many(queries)
                    retrieval_ms=(time.monotonic()-t_retrieval)*1000
                if wants_sources and previous_sources:
                    by_url = {s['url']:s for s in previous_sources}
                    for s in sources: by_url[s['url']] = s
                    sources = list(by_url.values())[:7]
                # No relevant evidence means no expensive, ungrounded inference.
                if not sources and not is_casual and not is_recall:
                    response = ('I couldn’t verify that from Superspace’s published information. '
                                'Please contact support@superspaceapp.com for confirmation.')
                    status = 'unverified'
                    skip_inference = True
                else:
                    skip_inference = False
                # Customer-recall questions do not require support documentation.
                cache_key = answer_cache.key(message, sources, history, facts, recall=is_recall) if not skip_inference else None
                cached_response = answer_cache.get(cache_key)
                if cached_response is not None:
                    response = cached_response
                    cache_hit = True
                    skip_inference = True
                if not skip_inference:
                    # Reject excess traffic quickly instead of accumulating an unlimited queue.
                    # acquire_nowait is not available on asyncio.Semaphore; wait_for(1ms)
                    # permits immediate ready permits without introducing a long queue.
                    try:
                        await asyncio.wait_for(inference_capacity.acquire(), timeout=0.1)
                    except asyncio.TimeoutError:
                        raise HTTPException(503, detail='Support is at capacity. Please retry shortly.',
                                            headers={'Retry-After': '10'})
                    try:
                        queue_start = time.monotonic()
                        try:
                            await asyncio.wait_for(max_concurrent.acquire(), timeout=float(os.getenv('INFERENCE_QUEUE_TIMEOUT_SECONDS','45')))
                        except asyncio.TimeoutError:
                            raise HTTPException(503, detail='Support is busy. Please retry shortly.',
                                                headers={'Retry-After': '10'})
                        queue_ms = (time.monotonic()-queue_start)*1000
                        try:
                            # Consume the local budget only after admission. These limits
                            # are process-local; a persistent shared budget is required for replicas.
                            day = datetime.now(timezone.utc).date().isoformat()
                            async with budget_lock:
                                if ai_budget['day'] != day:
                                    ai_budget.update(day=day, used=0)
                                if ai_budget['used'] >= int(os.getenv('MAX_AI_REQUESTS_PER_DAY','60')):
                                    raise HTTPException(429, detail='Daily AI request budget reached. Please try tomorrow.')
                                ai_budget['used'] += 1
                            provider_start = time.monotonic()
                            try:
                                response = await answer(message,sources,history,customer_facts=facts,
                                                        previous_sources=previous_sources,recall=is_recall)
                            finally:
                                provider_ms = (time.monotonic()-provider_start)*1000
                        finally:
                            max_concurrent.release()
                    finally:
                        inference_capacity.release()
                    inference_ms = queue_ms + provider_ms
                    answer_cache.put(cache_key, response)
                response = re.sub(r'\s*\[\d+\]', '', response).strip()
                if not response:
                    raise ModelError('Empty answer')
                status = 'answered' if sources or is_casual or is_recall else 'unverified'
                if status == 'unverified':
                    response = ('I couldn’t verify that from Superspace’s published information. '
                                'Please contact support@superspaceapp.com for confirmation.')
            cited = [{'title':s['title'],'url':s['url']} for s in sources[:4]]
            history.extend([{'role':'user','content':message},{'role':'assistant','content':response}])
            sessions[key] = {'expires':time.monotonic()+3600,'history':history[-24:],
                             'sources':sources or previous_sources,'facts':facts}
            # Expire stale sessions and locks; bounded in-memory store for single-worker demo.
            if len(sessions) > 500:
                expired = [sid for sid,val in sessions.items() if val['expires'] <= time.monotonic()]
                for sid in expired:
                    sessions.pop(sid,None); locks.pop(sid,None)
                while len(sessions) > 500:
                    oldest = next(iter(sessions))
                    sessions.pop(oldest,None); locks.pop(oldest,None)
            if len(limits) > 5000: limits.clear()
            LOG.info('chat status=%s sources=%d duration_ms=%d retrieval_ms=%.2f inference_ms=%.2f queue_ms=%.2f provider_ms=%.2f',status,len(sources),int((time.monotonic()-start)*1000),retrieval_ms,inference_ms,queue_ms,provider_ms)
            return {'answer':response,'session_id':key,'sources':cited,'status':status,'timings_ms':{'retrieval':round(retrieval_ms,2),'inference':round(inference_ms,2),'queue':round(queue_ms,2),'provider':round(provider_ms,2),'cache_hit':cache_hit,'total':round((time.monotonic()-start)*1000,2)}}
        except HTTPException:
            raise
        except ModelError as exc:
            LOG.warning('Cloudflare inference failed: %s',type(exc).__name__)
            raise HTTPException(503,detail='Our support assistant is temporarily unavailable. Please try again shortly.') from exc
        except Exception as exc:
            LOG.exception('Support request failed')
            raise HTTPException(503,detail='Our support assistant is temporarily unavailable. Please try again shortly.') from exc


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post('/api/v1/support/chat/stream')
async def chat_stream(data: ChatRequest, request: Request):
    """POST SSE API: meta -> delta* -> done, or error. Browser clients use fetch()."""
    ip = request.client.host if request.client else 'unknown'
    now = time.monotonic()
    hits = limits[ip]
    while hits and hits[0] < now - 60:
        hits.popleft()
    if len(hits) >= max(1, int(os.getenv('RATE_LIMIT_PER_MINUTE', '20'))):
        raise HTTPException(429, detail='Too many requests')
    hits.append(now)
    key = data.session_id if data.session_id in sessions and sessions[data.session_id]['expires'] > now else str(uuid.uuid4())
    message = data.message.strip()

    async def events():
        started = time.monotonic()
        async with locks.setdefault(key, asyncio.Lock()):
            saved = sessions.get(key, {})
            history = list(saved.get('history', []))
            facts = extract_customer_facts(message, saved.get('facts', {}))
            sources = []
            queue_ms = provider_ms = retrieval_ms = 0.0
            cache_hit = False
            response_text = ''
            admitted = False
            slot = False
            leader = False
            flight = None
            flight_key = None
            try:
                if prohibited_secret_request(message):
                    response_text = 'I cannot share private instructions or credentials.'
                elif smalltalk(message):
                    response_text = 'Hello! How can I help with Superspace?'
                else:
                    t = time.monotonic()
                    sources = await retriever.retrieve_many(search_queries(message, facts, history))
                    retrieval_ms = (time.monotonic() - t) * 1000
                    if not sources:
                        response_text = 'I could not verify that from the Superspace Help Center. Please contact support@superspaceapp.com.'
                citations = [{'title': s['title'], 'url': s['url']} for s in sources[:4]]
                yield _sse('meta', {'session_id': key, 'sources': citations})
                cache_key = answer_cache.key(message, sources, history, facts)
                if not response_text:
                    response_text = answer_cache.get(cache_key) or ''
                    cache_hit = bool(response_text)
                if response_text:
                    yield _sse('delta', {'text': response_text})
                else:
                    # Single-flight coalescing: 100 identical public questions use
                    # one provider call, even when they arrive simultaneously.
                    if cache_key:
                        async with stream_flights_lock:
                            flight = stream_flights.get(cache_key)
                            if flight is None:
                                flight = asyncio.get_running_loop().create_future()
                                stream_flights[cache_key] = flight
                                leader = True
                                flight_key = cache_key
                        if not leader:
                            try:
                                shared = await asyncio.wait_for(asyncio.shield(flight), timeout=float(os.getenv('INFERENCE_QUEUE_TIMEOUT_SECONDS', '45')) + float(os.getenv('CLOUDFLARE_TIMEOUT_SECONDS', '22')))
                            except asyncio.TimeoutError:
                                shared = None
                            if shared:
                                response_text = shared
                                cache_hit = True
                                yield _sse('delta', {'text': response_text})
                            else:
                                yield _sse('error', {'code': 'upstream', 'message': 'Support temporarily unavailable.'})
                                return
                    if not response_text:
                        try:
                            await asyncio.wait_for(inference_capacity.acquire(), timeout=0.1)
                            admitted = True
                        except asyncio.TimeoutError:
                            yield _sse('error', {'code': 'capacity', 'message': 'Support is busy. Retry shortly.'})
                            return
                        t = time.monotonic()
                        try:
                            await asyncio.wait_for(max_concurrent.acquire(), timeout=float(os.getenv('INFERENCE_QUEUE_TIMEOUT_SECONDS', '45')))
                            slot = True
                        except asyncio.TimeoutError:
                            yield _sse('error', {'code': 'timeout', 'message': 'Support is busy. Retry shortly.'})
                            return
                        queue_ms = (time.monotonic() - t) * 1000
                        day = datetime.now(timezone.utc).date().isoformat()
                        async with budget_lock:
                            if ai_budget['day'] != day:
                                ai_budget.update(day=day, used=0)
                            if ai_budget['used'] >= int(os.getenv('MAX_AI_REQUESTS_PER_DAY', '60')):
                                yield _sse('error', {'code': 'budget', 'message': 'Daily AI budget reached.'})
                                return
                            ai_budget['used'] += 1
                        t = time.monotonic()
                        async for fragment in stream_answer(message, sources, history, customer_facts=facts):
                            response_text += fragment
                            yield _sse('delta', {'text': fragment})
                        provider_ms = (time.monotonic() - t) * 1000
                        answer_cache.put(cache_key, response_text)
                    if leader and flight is not None and not flight.done():
                        flight.set_result(response_text or None)
                if response_text:
                    history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': response_text}])
                    sessions[key] = {'expires': time.monotonic()+3600, 'history': history[-24:],
                                     'sources': sources or saved.get('sources', []), 'facts': facts}
                yield _sse('done', {'session_id': key, 'sources': citations,
                    'timings_ms': {'retrieval': round(retrieval_ms,2), 'queue': round(queue_ms,2),
                                   'provider': round(provider_ms,2), 'total': round((time.monotonic()-started)*1000,2),
                                   'cache_hit': cache_hit}})
            except (ModelError, Exception):
                LOG.exception('SSE stream failed')
                yield _sse('error', {'code': 'inference', 'message': 'Support temporarily unavailable.'})
            finally:
                if leader and flight is not None:
                    if not flight.done():
                        flight.set_result(None)
                    async with stream_flights_lock:
                        if stream_flights.get(flight_key) is flight:
                            stream_flights.pop(flight_key, None)
                if slot:
                    max_concurrent.release()
                if admitted:
                    inference_capacity.release()
    return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no'})
