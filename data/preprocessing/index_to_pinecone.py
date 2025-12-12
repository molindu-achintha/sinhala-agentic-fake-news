"""
Index data into Pinecone vector store.

This script:
1. Loads preprocessed dataset from processed.jsonl
2. Fetches latest news from scrapers
3. Applies NLP preprocessing to live news
4. Generates embeddings via OpenRouter API
5. Stores everything in Pinecone
"""

import os
import sys
import json
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

from app.store.pinecone_store import PineconeVectorStore
from app.agents.langproc_agent import LangProcAgent
from app.utils.text_normalize import normalize_text
from app.utils.sinhala_nlp import get_sinhala_nlp
from app.scrapers.news_scraper import get_news_aggregator


def load_processed_data(filepath: str = 'data/dataset/processed.jsonl', limit: int = None):
    """Load preprocessed documents from JSONL file."""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            documents.append(doc)
    return documents


def preprocess_news_article(article, nlp) -> dict:
    """
    Apply NLP preprocessing to a news article.
    
    Args:
        article: NewsArticle object from scraper
        nlp: SinhalaNLP instance
    
    Returns:
        Preprocessed document dict
    """
    text = article.title
    normalized_text = normalize_text(text)
    
    # Apply NLP preprocessing
    try:
        # Tokenize
        tokens = nlp.tokenize(normalized_text)
        
        # POS Tagging
        pos_tags = nlp.pos_tag(normalized_text)
        
        # Named Entity Recognition
        entities = nlp.extract_entities(normalized_text)
        
        # Claim detection
        claim_indicators = nlp.detect_claim_indicators(normalized_text)
        has_claim = len(claim_indicators) > 0
        
        # Negation detection
        has_negation = nlp.detect_negation(normalized_text)
        
        # Extract nouns and verbs (handle nested structure from sinling)
        nouns = []
        verbs = []
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
        
    except Exception as e:
        # Fallback if NLP processing fails
        tokens = normalized_text.split()
        entities = {}
        nouns = []
        verbs = []
        has_claim = False
        has_negation = False
    
    return {
        "id": article.id,
        "text": normalized_text,
        "title": article.title,
        "source": article.source,
        "url": article.url,
        "type": "live_news",
        "label": "",  # Unknown for live news
        # NLP features
        "token_count": len(tokens),
        "entities": entities,
        "nouns": nouns[:10], 
        "verbs": verbs[:5],
        "has_claim_indicator": has_claim,
        "has_negation": has_negation
    }


async def fetch_and_preprocess_news(limit: int = 1000):
    """Fetch latest news from all sources and apply preprocessing."""
    print("Initializing NLP processor...")
    nlp = get_sinhala_nlp()
    
    print("Fetching news from scrapers...")
    aggregator = get_news_aggregator()
    articles = await aggregator.fetch_all_news(use_cache=False)
    
    print(f"Fetched {len(articles)} raw articles")
    print("Applying NLP preprocessing to each article...")
    
    # Preprocess each article
    preprocessed_docs = []
    for article in tqdm(articles[:limit], desc="Preprocessing news"):
        try:
            doc = preprocess_news_article(article, nlp)
            preprocessed_docs.append(doc)
        except Exception as e:
            print(f"\nError preprocessing article: {e}")
            continue
    
    print(f"Successfully preprocessed {len(preprocessed_docs)} articles")
    return preprocessed_docs


def index_to_pinecone():
    """Main function to index all data into Pinecone."""
    # Load environment
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    load_dotenv(os.path.join(root_dir, '.env'))
    
    # Check API keys
    if not os.getenv('PINECONE_API_KEY'):
        print("ERROR: PINECONE_API_KEY not set in .env")
        print("Get your API key from: https://www.pinecone.io/")
        return
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return
    
    print("=" * 60)
    print("INDEXING DATA TO PINECONE")
    print("=" * 60)
    
    # Initialize components
    lang_proc = LangProcAgent()
    pinecone_store = PineconeVectorStore()
    
    print(f"\nPinecone Stats: {pinecone_store.get_stats()}")
    
    # PART 1: Index Preprocessed Dataset
    print("PART 1: Indexing Preprocessed Dataset (with labels)")
    
    dataset_path = os.path.join(root_dir, 'data/dataset/processed.jsonl')
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        dataset_docs = load_processed_data(dataset_path, limit=2000)
        print(f"Loaded {len(dataset_docs)} documents")
        
        # Generate embeddings and index
        embeddings = []
        valid_docs = []
        
        for doc in tqdm(dataset_docs, desc="Generating embeddings"):
            try:
                text = normalize_text(doc.get('text', '') or doc.get('claim', ''))
                if not text:
                    continue
                
                embedding = lang_proc.get_embeddings(text)
                embeddings.append(embedding)
                
                # Ensure label is included
                doc['type'] = 'dataset'
                valid_docs.append(doc)
                
            except Exception as e:
                print(f"\nError: {e}")
                continue
        
        if valid_docs:
            print(f"\nUpserting {len(valid_docs)} documents to Pinecone (namespace: 'dataset')...")
            pinecone_store.upsert_documents(valid_docs, embeddings, namespace="dataset")
            print("Dataset indexed successfully!")
    else:
        print(f"Dataset file not found: {dataset_path}")
    
    # PART 2: Index Preprocessed Live News
    print("PART 2: Indexing Live News (with NLP preprocessing)")
    
    # Fetch and preprocess news
    news_docs = asyncio.run(fetch_and_preprocess_news(limit=1000))
    print(f"Total preprocessed news: {len(news_docs)}")
    
    # Generate embeddings and index
    embeddings = []
    valid_docs = []
    
    for doc in tqdm(news_docs, desc="Generating news embeddings"):
        try:
            text = normalize_text(doc.get('text', '') or doc.get('title', ''))
            if not text:
                continue
            
            embedding = lang_proc.get_embeddings(text)
            embeddings.append(embedding)
            valid_docs.append(doc)
            
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    if valid_docs:
        print(f"\nUpserting {len(valid_docs)} preprocessed news articles to Pinecone (namespace: 'live_news')...")
        pinecone_store.upsert_documents(valid_docs, embeddings, namespace="live_news")
        print("Live news indexed successfully!")
    
    # Summary
    print("INDEXING COMPLETE")
    stats = pinecone_store.get_stats()
    print(f"Total vectors in Pinecone: {stats['total_vectors']}")
    print(f"Namespaces: {stats['namespaces']}")
    print("=" * 60)
    
    # Show sample of what was indexed
    print("\nSample Preprocessed News Document:")
    if valid_docs:
        sample = valid_docs[0]
        print(f"  Title: {sample.get('title', '')[:60]}...")
        print(f"  Source: {sample.get('source', '')}")
        print(f"  Entities: {sample.get('entities', {})}")
        print(f"  Nouns: {sample.get('nouns', [])}")
        print(f"  Has Claim Indicator: {sample.get('has_claim_indicator', False)}")


if __name__ == "__main__":
    index_to_pinecone()
