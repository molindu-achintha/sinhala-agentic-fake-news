"""
news.py - News API Endpoint

This module provides endpoints for fetching and indexing news.
It handles:
1. Fetching news from multiple Sinhala sources
2. NLP preprocessing of articles
3. Indexing news into the vector database
"""
from fastapi import APIRouter, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from datetime import datetime

from ...scrapers.news_scraper import get_news_aggregator, NewsArticle
from ...utils.sinhala_nlp import get_sinhala_nlp
from ...utils.text_normalize import normalize_text, preprocess_for_indexing, is_valid_for_indexing

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
    """
    Apply NLP preprocessing to a news article.
    
    This function extracts entities, POS tags, and other
    NLP features from the article text.
    """
    print("[news] Preprocessing article:", article.title[:30], "...")
    
    nlp = get_sinhala_nlp()
    
    # Combine title and content for analysis
    text = f"{article.title}. {article.content}" if article.content else article.title
    text = normalize_text(text)
    
    # NLP Analysis
    tokens = nlp.tokenize(text)
    pos_tags = nlp.pos_tag(text)
    entities = nlp.extract_entities(text)
    
    # Extract nouns and verbs from POS tags
    nouns = []
    verbs = []
    try:
        for item in pos_tags:
            if isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, tuple) and len(subitem) == 2:
                        word, tag = subitem
                        if tag in ['NN', 'NNP', 'RP']:
                            nouns.append(word)
                        elif tag in ['VB', 'VFM']:
                            verbs.append(word)
            elif isinstance(item, tuple) and len(item) == 2:
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
    
    Args:
        source: Optional filter by source
        limit: Maximum number of articles to return
        preprocess: Whether to apply NLP preprocessing
    
    Returns:
        NewsResponse with articles and metadata
    """
    print("[news] Fetching news, source:", source, "limit:", limit)
    
    aggregator = get_news_aggregator()
    
    if source:
        articles = await aggregator.fetch_by_source(source)
    else:
        articles = await aggregator.fetch_all_news()
    
    print("[news] Fetched", len(articles), "articles")
    
    # Limit results
    articles = articles[:limit]
    
    # Preprocess if requested
    if preprocess and articles:
        print("[news] Preprocessing articles")
        processed = []
        for article in articles:
            try:
                processed.append(preprocess_article(article))
            except Exception as e:
                print("[news] Failed to preprocess article:", str(e))
                continue
        articles_response = processed
    else:
        # Return raw articles
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
    
    print("[news] Returning", len(articles_response), "articles")
    
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
    print("[news] Getting available sources")
    aggregator = get_news_aggregator()
    sources = aggregator.get_available_sources()
    print("[news] Available sources:", sources)
    return SourcesResponse(sources=sources)


@router.get("/news/refresh")
async def refresh_news_cache():
    """
    Force refresh the news cache.
    
    Scrapes all sources again, bypassing the cache.
    Returns the number of articles fetched.
    """
    print("[news] Refreshing news cache")
    aggregator = get_news_aggregator()
    articles = await aggregator.fetch_all_news(use_cache=False)
    
    print("[news] Refreshed with", len(articles), "articles")
    
    return {
        "success": True,
        "message": f"Refreshed cache with {len(articles)} articles",
        "sources": aggregator.get_available_sources(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/news/index")
async def index_news_to_pinecone():
    """
    Index scraped news into Pinecone vector store.
    
    This makes scraped news available as evidence for claim verification.
    Generates embeddings for each article and adds them to the live_news namespace.
    """
    print("[news] Starting news indexing to Pinecone")
    
    from ...agents.langproc_agent import LangProcAgent
    from ...store.pinecone_store import get_pinecone_store
    from ...config import get_settings
    
    settings = get_settings()
    aggregator = get_news_aggregator()
    
    # Get scraped articles
    articles = await aggregator.fetch_all_news(use_cache=True)
    print("[news] Found", len(articles), "articles to index")
    
    if not articles:
        print("[news] No articles to index")
        return {
            "success": False,
            "message": "No articles to index. Try /news/refresh first.",
            "indexed_count": 0
        }
    
    # Initialize components
    lang_proc = LangProcAgent()
    
    try:
        pinecone_store = get_pinecone_store()
    except Exception as e:
        print("[news] Failed to connect to Pinecone:", str(e))
        return {
            "success": False,
            "message": f"Failed to connect to Pinecone: {str(e)}",
            "indexed_count": 0
        }
    
    indexed_count = 0
    embeddings_list = []
    docs_list = []
    
    for article in articles:
        try:
            # Preprocess article text - removes headers, footer, ads
            raw_text = f"{article.title}. {article.content}" if article.content else article.title
            text = preprocess_for_indexing(raw_text)
            
            # Validate before indexing
            if not is_valid_for_indexing(text, min_length=30):
                print("[news] Skipping article - not valid for indexing")
                continue
            
            # Truncate for embedding
            text = text[:500]
            
            embedding = lang_proc.get_embeddings(text)
            
            # Prepare document for storage
            # News from trusted sources (Hiru, BBC, etc.) are labeled as true
            doc = {
                "id": article.id,
                "text": text[:1000],
                "title": normalize_text(article.title)[:200],
                "source": article.source,
                "url": article.url,
                "label": "true",  # Trusted news sources
                "type": "live_news",
                "scraped_at": article.scraped_at
            }
            
            embeddings_list.append(embedding.tolist())
            docs_list.append(doc)
            indexed_count += 1
            
        except Exception as e:
            print("[news] Failed to process article:", str(e))
            continue
    
    # Add to Pinecone
    if embeddings_list:
        print("[news] Upserting", len(embeddings_list), "articles to Pinecone")
        pinecone_store.upsert_documents(docs_list, embeddings_list, namespace="live_news")
    
    print("[news] Indexing complete,", indexed_count, "articles indexed")
    
    return {
        "success": True,
        "message": f"Indexed {indexed_count} articles into Pinecone",
        "indexed_count": indexed_count,
        "timestamp": datetime.now().isoformat()
    }
