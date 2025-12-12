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


@router.post("/news/index")
async def index_news_to_vectorstore():
    """
    Index scraped news into the FAISS vector store.
    
    This makes scraped news available as evidence for claim verification.
    Generates embeddings for each article and adds them to the index.
    """
    from ...agents.langproc_agent import LangProcAgent
    from ...store.vector_store import VectorStore
    from ...config import get_settings
    import numpy as np
    
    settings = get_settings()
    aggregator = get_news_aggregator()
    nlp = get_sinhala_nlp()
    
    # Get scraped articles
    articles = await aggregator.fetch_all_news(use_cache=True)
    
    if not articles:
        return {
            "success": False,
            "message": "No articles to index. Try /news/refresh first.",
            "indexed_count": 0
        }
    
    # Initialize components
    lang_proc = LangProcAgent()
    vector_store = VectorStore(index_path=settings.FAISS_INDEX_PATH, dimension=1536)
    vector_store.load_index()
    
    indexed_count = 0
    embeddings_list = []
    docs_list = []
    
    for article in articles:
        try:
            # Generate embedding for article title + content
            text = f"{article.title}. {article.content}" if article.content else article.title
            text = normalize_text(text)
            
            embedding = lang_proc.get_embeddings(text)
            
            # Prepare document for storage
            doc = {
                "id": article.id,
                "text": text[:500],
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "scraped_at": article.scraped_at,
                "type": "live_news"
            }
            
            embeddings_list.append(embedding)
            docs_list.append(doc)
            indexed_count += 1
            
        except Exception as e:
            continue
    
    # Add to vector store
    if embeddings_list:
        embeddings_array = np.array(embeddings_list, dtype='float32')
        vector_store.add_documents(embeddings_array, docs_list)
        vector_store.save_index()
    
    return {
        "success": True,
        "message": f"Indexed {indexed_count} articles into vector store",
        "indexed_count": indexed_count,
        "total_in_index": vector_store.index.ntotal,
        "timestamp": datetime.now().isoformat()
    }

