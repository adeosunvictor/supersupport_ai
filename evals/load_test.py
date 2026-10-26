"""HTTP concurrency load test: real inference calls, does not fake SLA results."""
import argparse
import asyncio
import json
import statistics
import time
import uuid
import httpx

def percentile(values,p):
    ordered=sorted(values)
    return round(ordered[min(len(ordered)-1,int((len(ordered)-1)*p))],2) if ordered else None

QUESTIONS = ['How do I publish a space?', 'How can I create my first space?', 'Can I manage availability?', 'How do I update my space photos?', 'Where are the booking instructions?']

async def run(url,users,requests_per_user,message,timeout):
    durations=[];first=[]; errors=[]; retrieval=[]; inference=[]; queue=[]; provider=[]
    semaphore=asyncio.Semaphore(users)
    async with httpx.AsyncClient(timeout=timeout,limits=httpx.Limits(max_connections=users+10)) as client:
        async def user_task(user_index):
            session_id=str(uuid.uuid4())
            for turn in range(requests_per_user):
                async with semaphore:
                    started=time.perf_counter()
                    try:
                        response=await client.post(url,json={'message':QUESTIONS[(user_index + turn) % len(QUESTIONS)] if message == 'mixed' else message,'session_id':session_id})
                        ms=(time.perf_counter()-started)*1000
                        if response.status_code!=200:
                            errors.append(response.status_code);continue
                        data=response.json();session_id=data.get('session_id',session_id)
                        durations.append(ms)
                        measure=data.get('timings_ms',{})
                        retrieval.append(measure.get('retrieval',0))
                        inference.append(measure.get('inference',0))
                        queue.append(measure.get('queue',0))
                        provider.append(measure.get('provider',0))
                        first.append(ms) # Nonstreaming response, NOT TTFT
                    except Exception as error:
                        errors.append(type(error).__name__)
        await asyncio.gather(*(user_task(i) for i in range(users)))
    report={'concurrent_users':users,'requests':users*requests_per_user,'successes':len(durations),
            'failures':len(errors),'failure_types':{str(t):errors.count(t) for t in set(errors)},
            'response_p50_ms':percentile(durations,.5),'response_p95_ms':percentile(durations,.95),
            'response_p99_ms':percentile(durations,.99),'retrieval_p95_ms':percentile(retrieval,.95),
            'inference_p95_ms':percentile(inference,.95),
            'queue_p95_ms':percentile(queue,.95),'provider_p95_ms':percentile(provider,.95),
            'ttft_ms':None,'note':'Nonstreaming endpoint. TTFT cannot be measured until SSE streaming is implemented.'}
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000/api/v1/support/chat');p.add_argument('--users',type=int,default=10);p.add_argument('--requests-per-user',type=int,default=2);p.add_argument('--message',default='mixed');p.add_argument('--timeout',type=float,default=90)
    a=p.parse_args();asyncio.run(run(a.url,a.users,a.requests_per_user,a.message,a.timeout))
