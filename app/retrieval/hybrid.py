"""Read-only hybrid retrieval: in-process BM25 + ONNX semantic search + RRF."""
import asyncio
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())

class IndexNotReady(RuntimeError):
    pass

class BM25:
    """Compact Okapi BM25, avoids a heavyweight database for a 122-article corpus."""
    def __init__(self, documents, k1=1.5, b=0.75):
        self.docs=[Counter(tokens(t)) for t in documents]
        self.lengths=np.asarray([sum(d.values()) for d in self.docs],dtype=np.float32)
        self.avg_len=max(float(self.lengths.mean()),1.0)
        self.k1,self.b=k1,b
        df=Counter()
        for d in self.docs: df.update(d.keys())
        self.idf={w:math.log(1+(len(self.docs)-freq+0.5)/(freq+0.5)) for w,freq in df.items()}
    def get_scores(self,query):
        scores=np.zeros(len(self.docs),dtype=np.float32)
        for word in set(query):
            if word not in self.idf: continue
            freq=np.fromiter((doc.get(word,0) for doc in self.docs),dtype=np.float32,count=len(self.docs))
            denom=freq+self.k1*(1-self.b+self.b*self.lengths/self.avg_len)
            scores+=self.idf[word]*freq*(self.k1+1)/np.maximum(denom,1e-6)
        return scores

class Embedder:
    def __init__(self,model_name, cache_dir=None):
        from fastembed import TextEmbedding
        self.model=TextEmbedding(model_name=model_name,cache_dir=cache_dir,threads=max(1,int(os.getenv('ONNX_THREADS','1'))),local_files_only=os.getenv('EMBEDDING_OFFLINE','false').lower()=='true')
    def encode(self,texts):
        data=np.asarray(list(self.model.embed(texts,batch_size=16)),dtype=np.float32)
        lengths=np.maximum(np.linalg.norm(data,axis=1,keepdims=True),1e-12)
        return data/lengths

class HybridRetriever:
    def __init__(self,index_dir='data/index',embedder_factory=Embedder):
        self.path=Path(index_dir)
        self.embedder_factory=embedder_factory
        self.records=[]
        self.vectors=None
        self.bm25=None
        self.model=None
        self._sem=asyncio.Semaphore(max(1,int(os.getenv('MAX_CONCURRENT_RETRIEVAL','2'))))
    def load(self):
        try:
            manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
            self.records=json.loads((self.path/'chunks.json').read_text(encoding='utf-8'))
            self.vectors=np.load(self.path/'vectors.npy',allow_pickle=False)
        except (OSError,ValueError,KeyError) as e: raise IndexNotReady(f'Index missing or invalid: {e}') from e
        if not self.records or self.vectors.ndim!=2 or len(self.records)!=len(self.vectors) or not np.isfinite(self.vectors).all():
            raise IndexNotReady('Corrupted or empty index')
        if manifest.get('embedding_backend')!='fastembed':
            raise IndexNotReady('Wrong embedding backend. Rebuild with python -m scripts.sync_knowledge')
        self.model=self.embedder_factory(manifest['embedding_model'],cache_dir=os.getenv('FASTEMBED_CACHE_PATH'))
        self.bm25=BM25([x['title']+' '+x['text'] for x in self.records])
        if int(manifest.get('embedding_dimensions',0))!=self.vectors.shape[1]: raise IndexNotReady('Wrong vector dimension')
    @lru_cache(maxsize=256)
    def _encode_query(self,query):
        return self.model.encode([query])[0]
    def search_sync(self,query,top_k=5,candidate_k=40):
        if self.bm25 is None: raise IndexNotReady('Index not loaded')
        n=len(self.records); candidate_k=min(n,max(top_k,candidate_k))
        lexical=self.bm25.get_scores(tokens(query))
        dense=self.vectors@self._encode_query(query)
        lexical_ids=np.argsort(-lexical,kind='stable')[:candidate_k]
        dense_ids=np.argsort(-dense,kind='stable')[:candidate_k]
        scores={}
        for ids in (lexical_ids,dense_ids):
            for pos,idx in enumerate(ids,1): scores[int(idx)]=scores.get(int(idx),0)+1/(60+pos)
        ranked=sorted(scores,key=lambda i:(-scores[i],i))
        unique=[];seen=set()
        for i in ranked:
            if self.records[i]['url'] not in seen: unique.append(i);seen.add(self.records[i]['url'])
        unique.extend(i for i in ranked if i not in unique)
        return [dict(self.records[i],score=round(scores[i],6)) for i in unique[:top_k]]
    async def retrieve_many(self,queries,top_k=5):
        q=' '.join(list(dict.fromkeys(x.strip() for x in queries if x.strip()))[:3])
        if not q: return []
        async with self._sem:
            return await asyncio.to_thread(self.search_sync,q,top_k,40)
