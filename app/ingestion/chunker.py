"""Paragraph-aware chunking, preserving source URLs for citations."""
import re

def chunk_articles(articles, max_chars=1100, overlap_chars=180):
    chunks=[]
    for article in articles:
        text=article['text'].strip()
        paragraphs=[p.strip() for p in re.split(r'\n\s*\n|(?<=\.)\n',text) if p.strip()]
        if not paragraphs: paragraphs=[text]
        current=''
        parts=[]
        for para in paragraphs:
            if len(para)>max_chars:
                if current: parts.append(current); current=''
                step=max_chars-overlap_chars
                for i in range(0,len(para),step):
                    part=para[i:i+max_chars]
                    if part: parts.append(part)
            elif current and len(current)+len(para)+2>max_chars:
                parts.append(current)
                current=current[-overlap_chars:]+ '\n' + para
            else:
                current=(current+'\n'+para).strip()
        if current: parts.append(current)
        for n,content in enumerate(parts):
            chunks.append({'id':f"{article['sha256'][:16]}:{n}", 'url':article['url'],
                           'title':article['title'],'text':content,'position':n})
    return chunks
