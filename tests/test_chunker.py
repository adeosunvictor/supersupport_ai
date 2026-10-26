from app.ingestion.chunker import chunk_articles

def test_preserves_source_and_chunks():
    items=[{'sha256':'abcdef1234567890','url':'https://superspaceapp.com/support/article/example',
            'title':'Example','text':'Paragraph one.\n\nParagraph two about payouts.'}]
    chunks=chunk_articles(items,max_chars=80,overlap_chars=10)
    assert chunks
    assert all(x['url']==items[0]['url'] for x in chunks)
    assert 'payouts' in ' '.join(x['text'] for x in chunks)
