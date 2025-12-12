"""
Index data into Pinecone vector store.

This script:
1. Loads preprocessed dataset from processed.jsonl
2. Fetches latest news from scrapers
3. Generates embeddings via OpenRouter API
4. Stores everything in Pinecone
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


async def fetch_live_news(limit: int = 1000):
    """Fetch latest news from all sources."""
    aggregator = get_news_aggregator()
    articles = await aggregator.fetch_all_news(use_cache=False)
    
    # Convert to document format
    documents = []
    for article in articles[:limit]:
        documents.append({
            "id": article.id,
            "text": article.title,
            "title": article.title,
            "source": article.source,
            "url": article.url,
            "type": "live_news",
            "label": ""  # Unknown for live news
        })
    
    return documents


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
    
    # ========================================
    # PART 1: Index Preprocessed Dataset
    # ========================================
    print("\n" + "=" * 40)
    print("PART 1: Indexing Preprocessed Dataset")
    print("=" * 40)
    
    dataset_path = os.path.join(root_dir, 'data/dataset/processed.jsonl')
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        dataset_docs = load_processed_data(dataset_path, limit=2000)  # Limit for speed
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
    
    # ========================================
    # PART 2: Index Live News
    # ========================================
    print("\n" + "=" * 40)
    print("PART 2: Indexing Live News")
    print("=" * 40)
    
    print("Fetching latest news from scrapers...")
    news_docs = asyncio.run(fetch_live_news(limit=1000))
    print(f"Fetched {len(news_docs)} news articles")
    
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
        print(f"\nUpserting {len(valid_docs)} news articles to Pinecone (namespace: 'live_news')...")
        pinecone_store.upsert_documents(valid_docs, embeddings, namespace="live_news")
        print("Live news indexed successfully!")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    stats = pinecone_store.get_stats()
    print(f"Total vectors in Pinecone: {stats['total_vectors']}")
    print(f"Namespaces: {stats['namespaces']}")
    print("=" * 60)


if __name__ == "__main__":
    index_to_pinecone()
