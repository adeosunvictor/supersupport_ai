"""Crawl, detect changes, reuse unchanged vectors, atomically publish verified index.
Usage: python -m scripts.sync_knowledge  (off the customer-request path)
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from app.ingestion.crawler import build_corpus
from app.ingestion.chunker import chunk_articles
from app.retrieval.hybrid import Embedder

LOG=logging.getLogger('index_sync')

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def article_hash(article):
    normalized=' '.join((article['title']+'\n'+article['text']).split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def validate_corpus(new,old,minimum=1):
    if not isinstance(new,list) or len(new)<minimum: raise RuntimeError(f'Incomplete crawl: {len(new) if isinstance(new,list) else 0} articles')
    urls=[x['url'] for x in new]
    if len(set(urls))!=len(urls): raise RuntimeError('Duplicate URLs in crawl')
    if any(not x.get('title') or not x.get('text') or not x.get('url') for x in new): raise RuntimeError('Invalid article data')
    if old and len(new)<max(minimum,int(len(old)*0.8)):
        raise RuntimeError(f'Unexpected crawl shrinkage: {len(old)} -> {len(new)}. Previous index retained.')

def prepare_index(articles,folder,model_name,previous=None,embedder_factory=Embedder):
    """Only modified chunks are reembedded. Removed chunks disappear from new snapshot."""
    chunks=chunk_articles(articles)
    if not chunks: raise RuntimeError('Empty chunks')
    old_records=[];old_vectors=None;old_manifest={}
    if previous and (previous/'manifest.json').exists():
        old_manifest=json.loads((previous/'manifest.json').read_text(encoding='utf-8'))
        if old_manifest.get('embedding_model')==model_name and old_manifest.get('embedding_backend')=='fastembed':
            old_records=json.loads((previous/'chunks.json').read_text(encoding='utf-8'))
            old_vectors=np.load(previous/'vectors.npy',allow_pickle=False)
    previous_map={}
    if old_vectors is not None and len(old_records)==len(old_vectors):
        previous_map={hashlib.sha256((r['title']+'\n'+r['text']).encode()).hexdigest():old_vectors[i] for i,r in enumerate(old_records)}
    vectors=[None]*len(chunks);pending=[];pending_idx=[]; reused=0
    for i,c in enumerate(chunks):
        key=hashlib.sha256((c['title']+'\n'+c['text']).encode()).hexdigest()
        if key in previous_map:
            vectors[i]=previous_map[key];reused+=1
        else:
            pending.append(c['title']+'\n'+c['text']);pending_idx.append(i)
    if pending:
        model=embedder_factory(model_name,cache_dir=os.getenv('FASTEMBED_CACHE_PATH'))
        arr=model.encode(pending)
        if len(arr)!=len(pending):raise RuntimeError('Embedding output mismatch')
        for idx,vec in zip(pending_idx,arr):vectors[idx]=vec
    arr=np.asarray(vectors,dtype=np.float32)
    if arr.ndim!=2 or len(arr)!=len(chunks) or not np.isfinite(arr).all() or np.any(np.linalg.norm(arr,axis=1)==0):
        raise RuntimeError('Invalid embedding index')
    folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'chunks.json',chunks)
    np.save(folder/'vectors.npy',arr,allow_pickle=False)
    manifest={'version':'2.2','embedding_backend':'fastembed','embedding_model':model_name,
              'embedding_dimensions':int(arr.shape[1]),'articles':len(articles),'chunks':len(chunks),
              'source_hash':hashlib.sha256(json.dumps(sorted((a['url'],article_hash(a)) for a in articles)).encode()).hexdigest(),
              'indexed_at':datetime.now(timezone.utc).isoformat()}
    dump(folder/'manifest.json',manifest)
    return {'articles':len(articles),'chunks':len(chunks),'reused_embeddings':reused,'created_embeddings':len(pending),'source_hash':manifest['source_hash']}

async def sync(skip_crawl=False):
    corpus=Path(os.getenv('CORPUS_PATH','data/corpus.json'))
    folder=Path(os.getenv('INDEX_DIR','data/index'))
    model=os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
    old=json.loads(corpus.read_text(encoding='utf-8')) if corpus.exists() else []
    if not skip_crawl:
        # Crawler writes to isolated temporary path, never mutates the last known-good corpus on failure.
        with tempfile.TemporaryDirectory(prefix='superspace-crawl-') as tmp:
            temp=Path(tmp)/'corpus.json'
            crawl=await build_corpus(str(temp))
            new=json.loads(temp.read_text(encoding='utf-8'))
    else:
        if not corpus.exists():raise RuntimeError('Missing corpus; run without --skip-crawl first')
        new=old
    minimum=int(os.getenv('MIN_EXPECTED_ARTICLES','100'))
    validate_corpus(new,old,minimum)
    new_urls={x['url']:article_hash(x) for x in new}
    old_urls={x['url']:article_hash(x) for x in old}
    change={'added':len(new_urls.keys()-old_urls.keys()),'modified':sum(new_urls[u]!=old_urls[u] for u in new_urls.keys()&old_urls.keys()),'removed':len(old_urls.keys()-new_urls.keys())}
    current_manifest={}
    if (folder/'manifest.json').exists():current_manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    source_hash=hashlib.sha256(json.dumps(sorted(new_urls.items())).encode()).hexdigest()
    if current_manifest.get('source_hash')==source_hash and current_manifest.get('embedding_backend')=='fastembed' and current_manifest.get('embedding_model')==model:
        print(json.dumps({'status':'unchanged',**change,'articles':len(new)},indent=2));return {'status':'unchanged'}
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='superspace-index-',dir=str(folder.parent.resolve())) as tempdir:
        staging=Path(tempdir)/'index'
        result=prepare_index(new,staging,model,previous=folder if folder.exists() else None)
        # Validate the staged snapshot before publication.
        chk=json.loads((staging/'chunks.json').read_text(encoding='utf-8'))
        vec=np.load(staging/'vectors.npy',allow_pickle=False)
        if len(chk)!=len(vec):raise RuntimeError('Staging validation failed')
        backup=folder.with_name(folder.name+'.backup')
        if backup.exists():shutil.rmtree(backup)
        previous_moved=False
        try:
            if folder.exists():folder.rename(backup);previous_moved=True
            staging.rename(folder)
        except Exception:
            if previous_moved and not folder.exists() and backup.exists():backup.rename(folder)
            raise
    corpus.parent.mkdir(parents=True,exist_ok=True)
    tmp=corpus.with_suffix('.pending')
    dump(tmp,new);tmp.replace(corpus)
    outcome={'status':'updated',**change,**result}
    print(json.dumps(outcome,indent=2));return outcome

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-crawl',action='store_true')
    args=parser.parse_args()
    asyncio.run(sync(skip_crawl=args.skip_crawl))
