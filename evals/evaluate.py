"""Article-level retrieval quality. Ground truth must be reviewed, not generated from ranking results."""
import argparse
import json
import statistics
import time
from pathlib import Path
from app.retrieval.hybrid import HybridRetriever

def evaluate(dataset,index_dir='data/index',min_recall=0.95,max_p95_ms=150,retriever=None):
    r=retriever or HybridRetriever(index_dir)
    if retriever is None:r.load()
    rows=[json.loads(line) for line in Path(dataset).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:raise ValueError('Golden dataset is empty')
    samples=[]
    for item in rows:
        gold=set(item['relevant_urls'])
        if not gold:raise ValueError('Each question requires human-verified relevant_urls')
        start=time.perf_counter()
        predicted=[v['url'] for v in r.search_sync(item['query'],top_k=30,candidate_k=60)]
        duration=(time.perf_counter()-start)*1000
        samples.append({'query':item['query'],'recall_30':len(gold.intersection(predicted))/len(gold),
                        'recall_5':len(gold.intersection(predicted[:5]))/len(gold),
                        'mrr_10':next((1/(i+1) for i,url in enumerate(predicted[:10]) if url in gold),0),
                        'ms':duration,'top_urls':predicted[:5]})
    ordered=sorted(x['ms'] for x in samples)
    p95=ordered[max(0,__import__('math').ceil(.95*len(ordered))-1)]
    avg=lambda name:round(statistics.mean(x[name] for x in samples),4)
    result={'questions':len(rows),'recall_at_30':avg('recall_30'),'recall_at_5':avg('recall_5'),
            'mrr_at_10':avg('mrr_10'),'retrieval_p50_ms':round(statistics.median(ordered),2),'retrieval_p95_ms':round(p95,2),
            'targets_met':avg('recall_30')>=min_recall and p95<=max_p95_ms,
            'note':'Gold URLs must be verified independently of retrieval rankings.'}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset');parser.add_argument('--index-dir',default='data/index');parser.add_argument('--enforce',action='store_true');args=parser.parse_args()
    result=evaluate(args.dataset,args.index_dir)
    if args.enforce and not result['targets_met']:raise SystemExit(1)
