"""
News API Endpoint - Fetches and preprocesses current Sinhala news.
"""
from fastapi import APIRouter, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from datetime import datetime

from ...scrapers.news_scraper import get_news_aggregator, NewsArticle
from ...utils.sinhala_nlp import get_sinhala_nlp
from ...utils.text_normalize import normalize_text

router = APIRouter()


class ProcessedNewsItem(BaseModel):
    """A news article with NLP preprocessing applied."""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_date: Optional[str]
    scraped_at: str
    # NLP Features
    entities: Dict[str, List[str]]
    nouns: List[str]
    verbs: List[str]
    has_claim_indicator: bool
    token_count: int


class NewsResponse(BaseModel):
    """Response containing news articles."""
    success: bool
    count: int
    sources: List[str]
    articles: List[ProcessedNewsItem]
    timestamp: str


class SourcesResponse(BaseModel):
    """Available news sources."""
    sources: List[str]


def preprocess_article(article: NewsArticle) -> ProcessedNewsItem:
    """Apply NLP preprocessing to a news article."""
    nlp = get_sinhala_nlp()
    
    # Combine title and content for analysis
    text = f"{article.title}. {article.content}" if article.content else article.title
    text = normalize_text(text)
    
    # NLP Analysis
    tokens = nlp.tokenize(text)
    pos_tags = nlp.pos_tag(text)
    entities = nlp.extract_entities(text)
    
    # Handle nested POS tags from sinling (returns list of lists)
    nouns = []
    verbs = []
    try:
        for item in pos_tags:
            if isinstance(item, list):
                # sinling format: [[('char', 'TAG'), ...], ...]
                for subitem in item:
                    if isinstance(subitem, tuple) and len(subitem) == 2:
                        word, tag = subitem
                        if tag in ['NN', 'NNP', 'RP']:
                            nouns.append(word)
                        elif tag in ['VB', 'VFM']:
                            verbs.append(word)
            elif isinstance(item, tuple) and len(item) == 2:
                # fallback format: [('word', 'TAG'), ...]
                word, tag = item
                if tag in ['NN', 'NNP']:
                    nouns.append(word)
                elif tag == 'VB':
                    verbs.append(word)
    except Exception:
        pass
    
    claim_indicators = nlp.detect_claim_indicators(text)
    
    return ProcessedNewsItem(
        id=article.id,
        title=article.title,
        content=article.content[:500] if article.content else "",
        url=article.url,
        source=article.source,
        published_date=article.published_date,
        scraped_at=article.scraped_at,
        entities=entities,
        nouns=nouns[:10],
        verbs=verbs[:5],
        has_claim_indicator=len(claim_indicators) > 0,
        token_count=len(tokens)
    )


@router.get("/news", response_model=NewsResponse)
async def get_current_news(
    source: Optional[str] = Query(None, description="Filter by source name"),
    limit: int = Query(20, ge=1, le=100, description="Maximum articles to return"),
    preprocess: bool = Query(True, description="Apply NLP preprocessing")
):
    """
    Fetch current news from Sinhala news sources.
    
    - **source**: Optional filter by source (e.g., "Hiru News", "BBC Sinhala")
    - **limit**: Maximum number of articles to return
    - **preprocess**: Whether to apply NLP preprocessing
    
    Returns news articles with optional NLP features (entities, POS tags, etc.)
    """
    aggregator = get_news_aggregator()
    
    if source:
        articles = await aggregator.fetch_by_source(source)
    else:
        articles = await aggregator.fetch_all_news()
    
    # Limit results
    articles = articles[:limit]
    
    # Preprocess if requested
    if preprocess and articles:
        processed = []
        for article in articles:
            try:
                processed.append(preprocess_article(article))
            except Exception as e:
                # Skip articles that fail preprocessing
                continue
        articles_response = processed
    else:
        # Return raw articles as ProcessedNewsItem with empty NLP fields
        articles_response = [
            ProcessedNewsItem(
                id=a.id,
                title=a.title,
                content=a.content[:500] if a.content else "",
                url=a.url,
                source=a.source,
                published_date=a.published_date,
                scraped_at=a.scraped_at,
                entities={},
                nouns=[],
                verbs=[],
                has_claim_indicator=False,
                token_count=0
            ) for a in articles
        ]
    
    return NewsResponse(
        success=True,
        count=len(articles_response),
        sources=aggregator.get_available_sources(),
        articles=articles_response,
        timestamp=datetime.now().isoformat()
    )


@router.get("/news/sources", response_model=SourcesResponse)
async def get_news_sources():
    """
    Get list of available news sources.
    
    Returns names of all supported Sinhala news providers.
    """
    aggregator = get_news_aggregator()
    return SourcesResponse(sources=aggregator.get_available_sources())


@router.get("/news/refresh")
async def refresh_news_cache():
    """
    Force refresh the news cache.
    
    Scrapes all sources again, bypassing the cache.
    """
    aggregator = get_news_aggregator()
    articles = await aggregator.fetch_all_news(use_cache=False)
    
    return {
        "success": True,
        "message": f"Refreshed cache with {len(articles)} articles",
        "sources": aggregator.get_available_sources(),
        "timestamp": datetime.now().isoformat()
    }
