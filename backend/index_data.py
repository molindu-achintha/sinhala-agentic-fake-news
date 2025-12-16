"""
index_data.py - Index Dataset into Pinecone

This script reads the processed.jsonl file and indexes documents
into the Pinecone vector database for semantic search.

Usage:
    cd backend
    python index_data.py
"""
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.agents.langproc_agent import LangProcAgent
from app.store.pinecone_store import PineconeVectorStore


def index_dataset(limit: int = None):
    """
    Index documents from processed.jsonl into Pinecone.
    
    Args:
        limit: Maximum number of documents to index (None for all)
    """
    print("=" * 50)
    print("Starting Pinecone Data Indexing")
    print("=" * 50)
    
    # Initialize components
    print("[index] Initializing LangProcAgent...")
    lang_proc = LangProcAgent()
    
    print("[index] Initializing PineconeVectorStore...")
    pinecone_store = PineconeVectorStore()
    
    # Path to processed data
    data_path = Path(__file__).parent.parent / "data" / "dataset" / "processed.jsonl"
    print("[index] Data path:", data_path)
    
    if not data_path.exists():
        print("[index] ERROR: Data file not found!")
        print("[index] Please ensure processed.jsonl exists in data/dataset/")
        return
    
    documents = []
    embeddings = []
    batch_size = 50
    total_indexed = 0
    
    print("[index] Reading data file...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Check limit
            if limit and i >= limit:
                print(f"[index] Reached limit of {limit} documents")
                break
            
            try:
                doc = json.loads(line)
                text = doc.get('text', '')[:500]
                
                if not text or len(text) < 10:
                    continue
                
                # Progress indicator
                if i % 100 == 0:
                    print(f"[index] Processing document {i}...")
                
                # Generate embedding
                embedding = lang_proc.get_embeddings(text)
                
                # Prepare document
                documents.append({
                    "id": f"doc_{i}",
                    "text": text,
                    "title": doc.get('title', '')[:200],
                    "source": doc.get('source', 'unknown'),
                    "label": doc.get('label', ''),
                    "url": doc.get('url', ''),
                    "type": "dataset"
                })
                embeddings.append(embedding.tolist())
                
                # Upsert in batches
                if len(documents) >= batch_size:
                    print(f"[index] Upserting batch at document {i}")
                    pinecone_store.upsert_documents(
                        documents, 
                        embeddings, 
                        namespace="dataset"
                    )
                    total_indexed += len(documents)
                    documents = []
                    embeddings = []
                    
            except json.JSONDecodeError:
                print(f"[index] Skipping invalid JSON at line {i}")
                continue
            except Exception as e:
                print(f"[index] Error at document {i}: {e}")
                continue
    
    # Upsert remaining documents
    if documents:
        print("[index] Upserting final batch...")
        pinecone_store.upsert_documents(documents, embeddings, namespace="dataset")
        total_indexed += len(documents)
    
    print("=" * 50)
    print("[index] INDEXING COMPLETE!")
    print(f"[index] Total documents indexed: {total_indexed}")
    print("[index] Pinecone stats:", pinecone_store.get_stats())
    print("=" * 50)


if __name__ == "__main__":
    # Parse command line argument for limit
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"[index] Limit set to {limit} documents")
        except ValueError:
            print("[index] Invalid limit, indexing all documents")
    
    index_dataset(limit=limit)
