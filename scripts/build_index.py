"""Backward-compatible entry point for v2.2. For live sync use python -m scripts.sync_knowledge."""
import asyncio
from scripts.sync_knowledge import sync
if __name__=='__main__':
    asyncio.run(sync(skip_crawl=False))
