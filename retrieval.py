"""Superspace live retrieval: support-first, multi-query RRF, passage reranking."""
import asyncio
import logging
import math
import re
import time
from collections import Counter, deque
from urllib.parse import urljoin, urlsplit, urlunsplit, urlencode
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)
STOP = set('a an and are as at be by can could do does for from how i in is it me my of on or our please should the this to what when where which who why with would you your superspace'.split())
HEADERS = {'User-Agent': 'SuperspaceSupportAI/1.22 (+support content reader)', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.8'}


def tokens(value):
    return [w for w in re.findall(r'[a-z0-9]+', value.lower()) if w not in STOP and len(w) > 1]


def relevance(query, text):
    """Lightweight lexical score for page-title and passage selection, not BM25."""
    terms = tokens(query)
    if not terms:
        return 0.0
    counts = Counter(tokens(text))
    return sum(1.0 + min(counts.get(t, 0), 4) * 0.6 for t in set(terms) if counts[t]) / len(set(terms))


class WebsiteRetriever:
    def __init__(self, base):
        p = urlsplit(base.rstrip('/'))
        if p.scheme != 'https' or p.hostname not in {'superspaceapp.com', 'www.superspaceapp.com'}:
            raise ValueError('SUPPORT_BASE_URL must be an approved Superspace HTTPS URL')
        self.origin = f'https://{p.hostname}'
        self.support = self.origin + '/support'
        self.hosts = {'superspaceapp.com', 'www.superspaceapp.com'}
        self._catalog = {}
        self._catalog_until = 0
        self._pages = {}
        self._lock = asyncio.Lock()
        self._results = {}
        self._inflight = {}
        self._client = None
        self._semaphore = asyncio.Semaphore(8)

    def safe_url(self, value, *, relative_to=None):
        try:
            p = urlsplit(urljoin(relative_to or self.origin + '/', value))
            if p.scheme != 'https' or p.hostname not in self.hosts or p.port not in (None, 443) or p.username or p.password:
                return None
            if re.search(r'\.(pdf|xml|zip|png|jpe?g|gif|svg|webp|ico|css|js|json|woff2?)$', p.path, re.I):
                return None
            return urlunsplit(('https', p.hostname, p.path.rstrip('/') or '/', '', ''))
        except (ValueError, TypeError):
            return None

    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(8, connect=3), headers=HEADERS, limits=httpx.Limits(max_connections=30, max_keepalive_connections=15))
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()

    async def _get(self, client, url):
        current = self.safe_url(url)
        if not current:
            raise ValueError('Unsafe source URL')
        seen = set()
        for _ in range(5):
            if current in seen:
                raise ValueError('Redirect loop')
            seen.add(current)
            async with self._semaphore:
                response = await client.get(current, follow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('location')
                target = self.safe_url(location or '', relative_to=current)
                if not target:
                    raise ValueError('Unsafe redirect')
                current = target
                continue
            response.raise_for_status()
            if len(response.content) > 1_500_000 or 'text/html' not in response.headers.get('content-type', '').lower():
                raise ValueError('Invalid HTML page')
            return response.text
        raise ValueError('Too many redirects')

    def _extract(self, html, url):
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.select('a[href]'):
            link = self.safe_url(a.get('href', ''), relative_to=url)
            if link and link != url:
                links.append((link, a.get_text(' ', strip=True)[:160]))
        title = (soup.title.get_text(' ', strip=True) if soup.title else url.rsplit('/', 1)[-1])[:180]
        for node in soup.select('script,style,noscript,svg,footer,nav,header,aside,form'):
            node.decompose()
        root = soup.select_one('article, main, [role="main"]') or soup.body or soup
        body = '\n'.join(line.strip() for line in root.get_text('\n').splitlines() if line.strip())
        return title, body[:35000], links

    async def _sitemap(self, client):
        discovered, todo, seen = set(), deque([self.origin + '/sitemap.xml']), set()
        while todo and len(seen) < 10 and len(discovered) < 1500:
            url = todo.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                p = urlsplit(url)
                if p.scheme != 'https' or p.hostname not in self.hosts or not p.path.endswith('.xml'):
                    continue
                r = await client.get(url, follow_redirects=False)
                r.raise_for_status()
                if len(r.content) > 1_500_000:
                    continue
                xml = ElementTree.fromstring(r.text)
                for node in xml.iter():
                    if not node.tag.endswith('loc') or not node.text:
                        continue
                    raw = node.text.strip()
                    parsed = urlsplit(raw)
                    if parsed.scheme != 'https' or parsed.hostname not in self.hosts:
                        continue
                    if parsed.path.endswith('.xml'):
                        todo.append(raw)
                    else:
                        good = self.safe_url(raw)
                        if good:
                            discovered.add(good)
            except (httpx.HTTPError, ElementTree.ParseError, ValueError):
                continue
        return discovered

    async def catalog(self, client):
        if self._catalog and time.monotonic() < self._catalog_until:
            return dict(self._catalog)
        async with self._lock:
            if self._catalog and time.monotonic() < self._catalog_until:
                return dict(self._catalog)
            pages = {self.support: 'Help Center'}
            for url in await self._sitemap(client):
                if urlsplit(url).path.startswith('/support'):
                    pages[url] = urlsplit(url).path.replace('-', ' ')
            # When sitemap contains articles, only fetch the Help Center homepage
            # for human-readable anchor titles. Avoid dozens of sequential page fetches.
            queue, visited = deque([self.support]), set()
            if sum('/support/article/' in url for url in pages) < 10:
                queue = deque([self.support])
                depth_limit = 12
            else:
                depth_limit = 1
            while queue and len(visited) < depth_limit and len(pages) < 850:
                url = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)
                try:
                    title, _, links = self._extract(await self._get(client, url), url)
                    pages[url] = title
                    for link, label in links:
                        path = urlsplit(link).path
                        if path.startswith('/support') and not path.startswith('/support/search'):
                            pages[link] = label or pages.get(link, path.replace('-', ' '))
                            if depth_limit > 1 and link not in visited and not path.startswith('/support/article/') and path.count('/') <= 3:
                                queue.append(link)
                except (httpx.HTTPError, ValueError) as exc:
                    LOG.info('Catalog page unavailable %s: %s', url, exc)
            self._catalog, self._catalog_until = pages, time.monotonic() + (1200 if len(pages) > 3 else 60)
            return dict(pages)

    async def read_page(self, client, url):
        url = self.safe_url(url)
        if not url:
            return None
        cached = self._pages.get(url)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        try:
            title, body, links = self._extract(await self._get(client, url), url)
            if len(body) < 90:
                return None
            result = {'url': url, 'title': title, 'body': body, 'links': links}
            self._pages[url] = (time.monotonic() + 300, result)
            if len(self._pages) > 500:
                self._pages.clear()
            return result
        except (httpx.HTTPError, ValueError) as exc:
            LOG.info('Page unavailable %s: %s', url, exc)
            return None

    def _passage(self, query, body):
        # Select overlapping windows, retaining neighboring sentences for context.
        lines = [x.strip() for x in body.splitlines() if len(x.strip()) > 10]
        if not lines:
            return '', 0
        windows = []
        for i in range(0, len(lines), 3):
            snippet = '\n'.join(lines[max(0, i-1):i+6])[:1800]
            windows.append((relevance(query, snippet), i, snippet))
        chosen = sorted(windows, reverse=True)[:3]
        chosen.sort(key=lambda row: row[1])
        return '\n\n'.join(x[2] for x in chosen)[:4200], max(x[0] for x in chosen)

    async def retrieve_many(self, queries):
        """Fast support-first lexical RRF; cache results and article content."""
        queries = list(dict.fromkeys(q.strip()[:180] for q in queries if isinstance(q, str) and q.strip()))[:5]
        if not queries:
            return []
        cache_key = tuple(q.lower() for q in queries)
        cached = self._results.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return list(cached[1])
        client = self.client()
        catalog = await self.catalog(client)
        rrf = Counter()
        ranked_by_query = []
        for query in queries:
            ranked = sorted(
                ((url, relevance(query, title + ' ' + urlsplit(url).path.replace('-', ' '))) for url, title in catalog.items() if urlsplit(url).path.startswith('/support/article/')),
                key=lambda item: item[1], reverse=True
            )
            ranked = [u for u, score in ranked if score > 0][:7]
            ranked_by_query.append(ranked)
            for rank, url in enumerate(ranked, 1):
                rrf[url] += 1 / (60 + rank)
        # Guarantee candidate coverage for each intent; do not let one topic dominate.
        candidates = list(dict.fromkeys(u for group in ranked_by_query for u in group[:3]))
        candidates += [u for u, _ in rrf.most_common(12) if u not in candidates]
        candidates = candidates[:12]
        pages = await asyncio.gather(*(self.read_page(client, u) for u in candidates))
        found = []
        for page in pages:
            if not page:
                continue
            values = [(self._passage(q, page['body']), relevance(q, page['title'])) for q in queries]
            (passage, passage_score), title_score = max(values, key=lambda x: x[0][1] * 2 + x[1] * 2)
            score = 2 * title_score + 2 * passage_score + rrf[page['url']] * 60
            if title_score >= 0.6 or passage_score >= 0.65:
                found.append({'url': page['url'], 'title': page['title'], 'text': passage or page['body'][:2200], 'score': round(score, 3)})
        found.sort(key=lambda item: item['score'], reverse=True)
        selected = found[:5]
        # Pin highly relevant topic guides, but fetch live content before using them.
        topic_paths = {
            'pricing': ['/support/article/plans-and-features', '/pricing'],
            'payment': ['/support/article/guest-says-they-paid', '/support/article/booking-payment-details', '/support/article/handle-unapplied-payments'],
            'payout': ['/support/article/add-a-payout-account', '/support/article/understand-your-earnings'],
            'setup': ['/support/article/create-your-first-space', '/support/article/space-drafts'],
            'equipment': ['/support/article/attach-equipment-to-spaces', '/support/article/add-ons-and-equipment-packages'],
            'website': ['/support/article/set-up-your-showcase'],
            'identity': ['/support/article/verify-your-identity'],
        }
        all_questions = ' '.join(queries).lower()
        topic_hits = []
        if re.search(r'\b(plans?|pricing|subscriptions?|monthly fee|costs?)\b', all_questions): topic_hits.append('pricing')
        if re.search(r'\b(payments?|booking (?:not|isn.t) paid|paid but|unapplied)\b', all_questions): topic_hits.append('payment')
        if re.search(r'\b(payouts?|withdraw|earnings|settlement)\b', all_questions): topic_hits.append('payout')
        if re.search(r'\b(first space|create (?:a|my) space|draft|publish)\b', all_questions): topic_hits.append('setup')
        if re.search(r'\b(equipment|add-ons?)\b', all_questions): topic_hits.append('equipment')
        if re.search(r'\b(own website|showcase|booking widget|embed)\b', all_questions): topic_hits.append('website')
        if re.search(r'\b(kyc|identity|verification)\b', all_questions): topic_hits.append('identity')
        pinned_urls = list(dict.fromkeys(self.origin + path for topic in topic_hits for path in topic_paths[topic]))[:7]
        pinned_pages = await asyncio.gather(*(self.read_page(client, url) for url in pinned_urls))
        pinned = []
        for page in pinned_pages:
            if not page: continue
            excerpt = '\n'.join(page['body'].splitlines()[:60])[:2800]
            pinned.append({'url':page['url'], 'title':page['title'], 'text':excerpt, 'score':99.0})
        # Keep topic-specific evidence even if the lexical passage threshold rejects it.
        selected = list({row['url']:row for row in pinned + selected}.values())[:7]
        # Main-site pages are a targeted fallback, especially for pricing.
        price_question = any(re.search(r'\b(pric(e|ing)|plan|subscription|cost|fee)\b', q, re.I) for q in queries)
        if price_question:
            pricing_page = await self.read_page(client, self.origin + '/pricing')
            if pricing_page:
                excerpt, _ = max((self._passage(q, pricing_page['body']) for q in queries), key=lambda x: x[1])
                selected = ([{'url': pricing_page['url'], 'title': pricing_page['title'], 'text': pricing_page['body'][:5000], 'score': 99}] + [p for p in selected if p['url'] != pricing_page['url']])[:7]
        if not selected:
            home = await self.read_page(client, self.origin + '/')
            if home:
                excerpt, score = max((self._passage(q, home['body']) for q in queries), key=lambda x: x[1])
                if score >= 0.65:
                    selected = [{'url': home['url'], 'title': home['title'], 'text': excerpt, 'score': score}]
        self._results[cache_key] = (time.monotonic() + 90, list(selected))
        if len(self._results) > 300:
            self._results.clear()
        return selected
