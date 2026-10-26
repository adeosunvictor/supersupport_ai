"""Bounded, version-aware answer cache for stateless public support questions.

No user history, identifiers, or customer facts are ever cached. A new article index
invalidates the cache via the source evidence fingerprint.
"""
import hashlib
import json
import os
import re
import time
from collections import OrderedDict

class AnswerCache:
    def __init__(self):
        self.items = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, message, sources, history, facts, *, recall=False):
        if history or facts or recall or not sources:
            return None
        cleaned = ' '.join(message.casefold().split())
        # Normalize generic onboarding questions for safe caching.
        cleaned = re.sub(
            r"\bmy first space\b",
            "a first space",
            cleaned,
        )
        # Only public help questions. Exclude PII and contextual/personal queries.
        if len(cleaned) > 180 or len(cleaned) < 10:
            return None
        if not re.match(r'^(how|what|where|when|can i|is there|does superspace|do you)\b', cleaned):
            return None
        if re.search(r'\b(my|mine|our|we|i paid|email|phone|account number|booking id|password|refund me|today|now)\b|@|\d{5,}', cleaned):
            return None
        evidence = [(s.get('url'), s.get('title'), s.get('text')) for s in sources[:4]]
        raw = json.dumps([cleaned, evidence, os.getenv('CLOUDFLARE_MODEL'), os.getenv('MAX_OUTPUT_TOKENS')], ensure_ascii=False)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def get(self, key):
        if key is None:
            return None
        item = self.items.get(key)
        if item is None or item[0] < time.monotonic():
            self.items.pop(key, None)
            self.misses += 1
            return None
        self.items.move_to_end(key)
        self.hits += 1
        return item[1]

    def put(self, key, answer):
        if key is None or not answer:
            return
        ttl = max(5, int(os.getenv('ANSWER_CACHE_TTL_SECONDS', '3600')))
        self.items[key] = (time.monotonic() + ttl, answer)
        self.items.move_to_end(key)
        max_items = max(1, int(os.getenv('ANSWER_CACHE_MAX_ITEMS', '512')))
        while len(self.items) > max_items:
            self.items.popitem(last=False)

    def clear(self):
        self.items.clear()

answer_cache = AnswerCache()
