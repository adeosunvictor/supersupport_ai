"""Offline-only crawl: never invoked by the chat request path."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from retrieval import WebsiteRetriever

async def build_corpus(path: str = 'data/corpus.json', max_pages: int = 600):
    retriever = WebsiteRetriever(os.getenv('SUPPORT_BASE_URL','https://superspaceapp.com/support'))
    try:
        client = retriever.client()
        catalog = await retriever.catalog(client)
        urls = [u for u in catalog if urlsplit(u).scheme == 'https' and urlsplit(u).hostname in {'superspaceapp.com','www.superspaceapp.com'} and urlsplit(u).path.startswith('/support/article/')]
        # Discover non-article support pages too, so that important guide indexes are not dropped.
        if not urls:
            urls = [u for u in catalog if urlsplit(u).path.startswith('/support') and u != retriever.support]
        urls = urls[:max_pages]
        pages = await asyncio.gather(*(retriever.read_page(client,u) for u in urls))
        records = []
        for p in pages:
            if not p or len(p['body'].strip()) < 90: continue
            records.append({'url':p['url'], 'title':p['title'], 'text':p['body'],
                            'sha256':hashlib.sha256(p['body'].encode()).hexdigest()})
        if not records: raise RuntimeError('No articles crawled. Index not replaced.')
        target = Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(target)
        return {'discovered':len(urls),'indexed_articles':len(records),'corpus_path':str(target)}
    finally:
        await retriever.close()
