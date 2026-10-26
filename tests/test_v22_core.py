import asyncio
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from app.retrieval.hybrid import BM25,HybridRetriever,IndexNotReady
from scripts.sync_knowledge import prepare_index,validate_corpus,article_hash
from app.ingestion.chunker import chunk_articles

class FakeEmbedder:
    calls=[]
    def __init__(self,model_name,cache_dir=None):pass
    def encode(self,texts):
        FakeEmbedder.calls.extend(texts)
        words=['publish','refund','space','payout']
        arr=np.array([[float(t.lower().count(w))+.01 for w in words] for t in texts],dtype=np.float32)
        return arr / np.linalg.norm(arr,axis=1,keepdims=True)

def articles():
    return [{'url':'https://superspaceapp.com/support/article/publish','title':'Publish a space','text':'Publish your space using the Publish button.','sha256':'1'*64},
            {'url':'https://superspaceapp.com/support/article/refunds','title':'Refund guide','text':'Learn about the refund process.','sha256':'2'*64}]

def test_bm25_prefers_matching_article():
    scores=BM25(['publish first space','refund requests']).get_scores(['publish'])
    assert scores[0]>scores[1]

def test_retrieval_and_persistent_index():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)/'index'
        prepare_index(articles(),root,'fake',embedder_factory=FakeEmbedder)
        r=HybridRetriever(root,embedder_factory=FakeEmbedder);r.load()
        assert r.search_sync('publish a space')[0]['url'].endswith('/publish')
        assert len(r.search_sync('refund',top_k=30))==len(chunk_articles(articles()))

def test_reuse_unchanged_embeddings_and_replace_modified():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);first=root/'first';second=root/'second';third=root/'third'
        FakeEmbedder.calls=[]
        r1=prepare_index(articles(),first,'fake',embedder_factory=FakeEmbedder)
        assert r1['created_embeddings']==2
        r2=prepare_index(articles(),second,'fake',previous=first,embedder_factory=FakeEmbedder)
        assert r2['created_embeddings']==0 and r2['reused_embeddings']==2
        changed=articles();changed[0]=dict(changed[0],text='Publish button is now on your dashboard.')
        r3=prepare_index(changed,third,'fake',previous=second,embedder_factory=FakeEmbedder)
        assert r3['created_embeddings']==1 and r3['reused_embeddings']==1

def test_reject_partial_crawl():
    with pytest.raises(RuntimeError,match='Incomplete crawl'):
        validate_corpus([],articles(),minimum=1)
    with pytest.raises(RuntimeError,match='shrinkage'):
        validate_corpus(articles()[:1],articles()*5,minimum=1)

def test_missing_index_fails_closed():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(IndexNotReady): HybridRetriever(d,embedder_factory=FakeEmbedder).load()

def test_chunk_url_is_preserved():
    assert chunk_articles(articles())[0]['url'].endswith('/publish')
