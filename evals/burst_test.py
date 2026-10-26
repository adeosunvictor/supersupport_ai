"""Burst test of POST streaming endpoint, with realistic unique or repeated prompts.

Example: python -m evals.burst_test --users 100 --mode repeat
WARNING: --mode unique uses REAL Cloudflare calls and consumes neurons.
"""
import argparse
import asyncio
import json
import time
import httpx

QUESTIONS = [
    'How do I create my first space?', 'How do I publish my space?',
    'Where can I change availability?', 'How can I upload space photos?',
    'What are the booking rules?', 'How can I unpublish a space?',
]

def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values)-1, int((len(values)-1)*p))], 1) if values else None

async def benchmark(url, users, mode, timeout):
    results = []
    gate = asyncio.Event()
    limits = httpx.Limits(max_connections=users+10, max_keepalive_connections=users+10)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), limits=limits) as client:
        async def one(i):
            question = QUESTIONS[0] if mode == 'repeat' else (QUESTIONS[i % len(QUESTIONS)] if mode == 'mixed' else f'{QUESTIONS[i % len(QUESTIONS)]} Please explain point {i+1}.')
            await gate.wait()
            start = time.perf_counter()
            first = None
            successful = False
            error = None
            try:
                async with client.stream('POST', url, json={'message': question}) as response:
                    if response.status_code != 200:
                        error = f'HTTP {response.status_code}'
                    else:
                        event = None
                        async for line in response.aiter_lines():
                            if line.startswith('event:'):
                                event = line[6:].strip()
                            elif line.startswith('data:'):
                                if event == 'delta' and first is None:
                                    first = (time.perf_counter()-start)*1000
                                if event == 'done':
                                    successful = True
                                if event == 'error':
                                    error = line[5:].strip()[:140]
            except Exception as exc:
                error = type(exc).__name__
            results.append({'ok': successful, 'error': error, 'first': first, 'total': (time.perf_counter()-start)*1000})
        tasks = [asyncio.create_task(one(i)) for i in range(users)]
        gate.set()
        await asyncio.gather(*tasks)
    good = [x for x in results if x['ok']]
    failures = {}
    for x in results:
        if not x['ok']:
            failures[x['error']] = failures.get(x['error'],0)+1
    output = {'requests': users, 'successes': len(good), 'failures': users-len(good), 'failure_types': failures,
              'ttft_p50_ms': pct([x['first'] for x in good if x['first'] is not None],.5),
              'ttft_p95_ms': pct([x['first'] for x in good if x['first'] is not None],.95),
              'total_p50_ms': pct([x['total'] for x in good], .5), 'total_p95_ms': pct([x['total'] for x in good],.95)}
    print(json.dumps(output, indent=2))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000/api/v1/support/chat/stream')
    parser.add_argument('--users',type=int,default=100)
    parser.add_argument('--mode',choices=['repeat','mixed','unique'],default='repeat')
    parser.add_argument('--timeout',type=float,default=90)
    args = parser.parse_args()
    asyncio.run(benchmark(args.url,args.users,args.mode,args.timeout))
