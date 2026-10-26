"""Single-command release verification: python -m scripts.verify [--offline] [--live]."""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import httpx

def check_index():
    root=Path(os.getenv('INDEX_DIR','data/index'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    records=json.loads((root/'chunks.json').read_text(encoding='utf-8'))
    vectors=np.load(root/'vectors.npy',allow_pickle=False)
    assert manifest['embedding_backend']=='fastembed','Run python -m scripts.sync_knowledge to rebuild v2.2 vectors'
    assert len(records)==len(vectors)>0 and len(records)==manifest['chunks']
    assert np.isfinite(vectors).all()
    article_count=len({r['url'] for r in records})
    assert article_count>=int(os.getenv('MIN_EXPECTED_ARTICLES','100')),f'Only {article_count} unique articles'
    return {'articles':article_count,'chunks':len(records),'model':manifest['embedding_model'],'dimensions':vectors.shape[1]}

async def live_test(url):
    from uuid import uuid4
    async with httpx.AsyncClient(timeout=40.0) as client:
        ready=await client.get(url.rstrip('/')+'/ready');ready.raise_for_status()
        start=time.perf_counter()
        r=await client.post(url.rstrip('/')+'/api/v1/support/chat',json={'message':'How do I create my first space?'})
        elapsed=round((time.perf_counter()-start)*1000,1)
        if r.status_code!=200:
            raise RuntimeError(f'Live chat failed: HTTP {r.status_code} {r.text[:400]}')
        data=r.json()
        assert isinstance(data.get('answer'),str) and data['answer'].strip()
        assert isinstance(data.get('sources'),list) and data['sources']
        assert all(s['url'].startswith('https://superspaceapp.com/support/') for s in data['sources'])
        assert data.get('session_id')
        return {'http_status':r.status_code,'answer':data['answer'],'sources':data['sources'],'timings_ms':data.get('timings_ms'), 'client_total_ms':elapsed}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true',help='Skip real Cloudflare HTTP call')
    parser.add_argument('--live',action='store_true',help='Run real Cloudflare request through API')
    parser.add_argument('--url',default=os.getenv('API_BASE_URL','http://127.0.0.1:8000'))
    args=parser.parse_args()
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests'],check=False)
    if tests.returncode:raise SystemExit(tests.returncode)
    try: result=check_index()
    except Exception as e:raise SystemExit(f'INDEX NOT READY: {e}. Run python -m scripts.sync_knowledge')
    print('INDEX VALID:',json.dumps(result,indent=2))
    if args.live:
        print('LIVE API TEST:',json.dumps(asyncio.run(live_test(args.url)),ensure_ascii=False,indent=2))
    else:
        print('Live Cloudflare inference was not tested. Start Uvicorn and rerun with --live.')

if __name__=='__main__':main()
