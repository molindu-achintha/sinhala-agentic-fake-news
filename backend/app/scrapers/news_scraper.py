"""
News Scraper Module for Sinhala News Sources
Fetches up to 100 articles from each source

Supports:
- Hiru News
- Ada Derana
- BBC Sinhala
- Lankadeepa
- Divaina
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional, Set
import re
import hashlib
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum articles per source
MAX_ARTICLES_PER_SOURCE = 100


@dataclass
class NewsArticle:
    """Represents a scraped news article."""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[str]
    scraped_at: str
    category: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class BaseScraper:
    """Base class for news scrapers."""
    
    SOURCE_NAME = "Unknown"
    BASE_URL = ""
    MAX_ARTICLES = MAX_ARTICLES_PER_SOURCE
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.seen_urls: Set[str] = set()
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def generate_id(self, url: str) -> str:
        """Generate unique ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        return text.strip()
    
    def is_sinhala_text(self, text: str) -> bool:
        """Check if text contains Sinhala characters."""
        sinhala_pattern = re.compile(r'[\u0D80-\u0DFF]')
        return bool(sinhala_pattern.search(text))
    
    def add_article(self, articles: List[NewsArticle], href: str, title: str) -> bool:
        """Add article if not duplicate. Returns True if added."""
        if href in self.seen_urls:
            return False
        if len(articles) >= self.MAX_ARTICLES:
            return False
        
        self.seen_urls.add(href)
        articles.append(NewsArticle(
            id=self.generate_id(href),
            title=self.clean_text(title[:200]),
            content="",
            url=href,
            source=self.SOURCE_NAME,
            published_date=None,
            scraped_at=datetime.now().isoformat()
        ))
        return True
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        """Override in subclasses."""
        raise NotImplementedError


class HiruNewsScraper(BaseScraper):
    """Scraper for Hiru News"""
    
    SOURCE_NAME = "Hiru News"
    BASE_URL = "https://www.hirunews.lk"
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        articles = []
        self.seen_urls.clear()
        
        # Just scrape the main Sinhala page - it has many articles
        urls = [
            "https://www.hirunews.lk/local-news.php?pageID=1",
            "https://www.hirunews.lk/local-news.php?pageID=2",
            "https://www.hirunews.lk/local-news.php?pageID=3",
            "https://www.hirunews.lk/local-news.php?pageID=4",
            "https://www.hirunews.lk/local-news.php?pageID=5",
            "https://www.hirunews.lk/world-news.php?pageID=1",
            "https://www.hirunews.lk/world-news.php?pageID=2",
            "https://www.hirunews.lk/",
        ]
        
        for page_url in urls:
            if len(articles) >= self.MAX_ARTICLES:
                break
                
            html = await self.fetch_page(session, page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                if len(articles) >= self.MAX_ARTICLES:
                    break
                try:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if not self.is_sinhala_text(title) or len(title) < 15:
                        continue
                    
                    if not href.startswith('http'):
                        href = self.BASE_URL + href
                    
                    self.add_article(articles, href, title)
                        
                except Exception as e:
                    continue
                
        return articles


class AdaDeranaScraper(BaseScraper):
    """Scraper for Ada Derana Sinhala - Multi-page"""
    
    SOURCE_NAME = "Ada Derana"
    BASE_URL = "https://sinhala.adaderana.lk"
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        articles = []
        self.seen_urls.clear()
        
        # Ada Derana - just use main page
        urls = [
            self.BASE_URL,
        ]
        
        for page_url in urls:
            if len(articles) >= self.MAX_ARTICLES:
                break
                
            html = await self.fetch_page(session, page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                if len(articles) >= self.MAX_ARTICLES:
                    break
                try:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if not self.is_sinhala_text(title) or len(title) < 15:
                        continue
                    
                    if '/news/' in href or '/hot-news' in href or href.endswith('.php'):
                        if not href.startswith('http'):
                            href = self.BASE_URL + href
                        
                        self.add_article(articles, href, title)
                        
                except Exception as e:
                    continue
                
        return articles


class BBCSinhalaScraper(BaseScraper):
    """Scraper for BBC Sinhala - Multi-page"""
    
    SOURCE_NAME = "BBC Sinhala"
    BASE_URL = "https://www.bbc.com/sinhala"
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        articles = []
        self.seen_urls.clear()
        
        # BBC Sinhala main pages only (topics may change)
        urls = [
            self.BASE_URL,
            "https://www.bbc.com/sinhala",
        ]
        
        for page_url in urls:
            if len(articles) >= self.MAX_ARTICLES:
                break
                
            html = await self.fetch_page(session, page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                if len(articles) >= self.MAX_ARTICLES:
                    break
                try:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if '/sinhala/' not in href:
                        continue
                        
                    if not self.is_sinhala_text(title) or len(title) < 10:
                        continue
                    
                    if not href.startswith('http'):
                        href = "https://www.bbc.com" + href
                    
                    # Skip topic/category pages
                    if '/topics/' in href:
                        continue
                    
                    self.add_article(articles, href, title)
                        
                except Exception as e:
                    continue
                
        return articles


class LankadeepaScraper(BaseScraper):
    """Scraper for Lankadeepa - Multi-page"""
    
    SOURCE_NAME = "Lankadeepa"
    BASE_URL = "https://www.lankadeepa.lk"
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        articles = []
        self.seen_urls.clear()
        
        # Multiple pages
        urls = [
            self.BASE_URL,
            f"{self.BASE_URL}/news",
            f"{self.BASE_URL}/latest",
            f"{self.BASE_URL}/business",
            f"{self.BASE_URL}/sports",
        ]
        
        for page in range(1, 11):
            urls.append(f"{self.BASE_URL}/news/page/{page}")
        
        for page_url in urls:
            if len(articles) >= self.MAX_ARTICLES:
                break
                
            html = await self.fetch_page(session, page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                if len(articles) >= self.MAX_ARTICLES:
                    break
                try:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if not self.is_sinhala_text(title) or len(title) < 15:
                        continue
                    
                    if not href.startswith('http'):
                        href = self.BASE_URL + href
                    
                    self.add_article(articles, href, title)
                        
                except Exception as e:
                    continue
                
        return articles


class DivainaScraper(BaseScraper):
    """Scraper for Divaina - Multi-page"""
    
    SOURCE_NAME = "Divaina"
    BASE_URL = "https://divaina.lk"
    
    async def scrape(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        articles = []
        self.seen_urls.clear()
        
        # Divaina main page only
        urls = [
            self.BASE_URL,
            "https://www.divaina.lk",
        ]
        
        for page_url in urls:
            if len(articles) >= self.MAX_ARTICLES:
                break
                
            html = await self.fetch_page(session, page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                if len(articles) >= self.MAX_ARTICLES:
                    break
                try:
                    href = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if not self.is_sinhala_text(title) or len(title) < 15:
                        continue
                    
                    if not href.startswith('http'):
                        href = self.BASE_URL + href
                    
                    self.add_article(articles, href, title)
                        
                except Exception as e:
                    continue
                
        return articles


class NewsAggregator:
    """Aggregates news from multiple sources."""
    
    def __init__(self):
        self.scrapers = [
            HiruNewsScraper(),
            AdaDeranaScraper(),
            BBCSinhalaScraper(),
            LankadeepaScraper(),
            DivainaScraper(),
        ]
        self._cache: Dict[str, NewsArticle] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes
    
    async def fetch_all_news(self, use_cache: bool = True) -> List[NewsArticle]:
        """Fetch news from all sources (up to 100 each, 500 total)."""
        if use_cache and self._cache_timestamp:
            age = (datetime.now() - self._cache_timestamp).total_seconds()
            if age < self._cache_ttl and self._cache:
                logger.info(f"Returning {len(self._cache)} cached articles")
                return list(self._cache.values())
        
        all_articles = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [scraper.scrape(session) for scraper in self.scrapers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Scraper {self.scrapers[i].SOURCE_NAME} failed: {result}")
                elif isinstance(result, list):
                    logger.info(f"{self.scrapers[i].SOURCE_NAME}: {len(result)} articles")
                    all_articles.extend(result)
        
        # Deduplicate by ID
        self._cache = {article.id: article for article in all_articles}
        self._cache_timestamp = datetime.now()
        
        logger.info(f"Total: {len(self._cache)} unique articles from {len(self.scrapers)} sources")
        return list(self._cache.values())
    
    async def fetch_by_source(self, source_name: str) -> List[NewsArticle]:
        """Fetch news from a specific source."""
        for scraper in self.scrapers:
            if scraper.SOURCE_NAME.lower() == source_name.lower():
                async with aiohttp.ClientSession() as session:
                    return await scraper.scrape(session)
        return []
    
    def get_available_sources(self) -> List[str]:
        """Get list of available news sources."""
        return [s.SOURCE_NAME for s in self.scrapers]


# Singleton
_aggregator = None

def get_news_aggregator() -> NewsAggregator:
    """Get or create the news aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = NewsAggregator()
    return _aggregator
