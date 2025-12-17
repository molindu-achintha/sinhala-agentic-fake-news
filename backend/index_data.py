"""
index_data.py - Index data to Pinecone

Reads processed.jsonl and uploads to Pinecone.
Clears existing data before indexing.
Upserts each document immediately after embedding.

Usage:
    python index_data.py         # All documents
    python index_data.py 100     # First 100 documents
"""
import json
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.agents.langproc_agent import LangProcAgent
from app.store.pinecone_store import PineconeVectorStore
from app.utils.text_normalize import preprocess_for_indexing, is_valid_for_indexing


def index_dataset(limit: int = None, clear_first: bool = True):
    """
    Index documents to Pinecone.
    
    Args:
        limit: Max documents to index
        clear_first: If True, clear namespace before indexing
    """
    
    print("PINECONE INDEXING")
    print("")
    
    # Step 1: Init agents
    print("[index] Setting up...")
    lang_proc = LangProcAgent()
    
    try:
        store = PineconeVectorStore()
        print("[index] Current stats:", store.get_stats())
    except Exception as e:
        print("[index] Error:", e)
        return
    
    # Step 2: Clear existing data
    if clear_first:
        print("[index] Clearing dataset namespace...")
        try:
            store.delete_namespace("dataset")
            print("[index] Namespace cleared")
        except Exception as e:
            print("[index] Clear failed (may be empty):", e)
    
    # Step 3: Load data path - use unified labeled dataset
    data_path = Path(__file__).parent.parent / "data" / "dataset" / "unified_labeled.jsonl"
    print("[index] Data file:", data_path)
    
    if not data_path.exists():
        print("[index] File not found!")
        return
    
    # Step 4: Process and upsert immediately
    indexed = 0
    skipped = 0
    
    print("[index] Processing and uploading...")
    print("")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Check limit
            if limit and i >= limit:
                print(f"[index] Limit {limit} reached")
                break
            
            try:
                # Load document
                doc = json.loads(line)
                raw_text = doc.get('text', '')
                
                # Clean text
                text = preprocess_for_indexing(raw_text)
                
                # Skip if invalid
                if not is_valid_for_indexing(text, min_length=30):
                    skipped += 1
                    continue
                
                # Truncate
                text = text[:500]
                
                # Get embedding
                emb = lang_proc.get_embeddings(text)
                
                # Prepare document
                doc_data = {
                    "id": f"doc_{i}",
                    "text": text,
                    "title": doc.get('title', '')[:200],
                    "source": doc.get('source', 'unknown'),
                    "label": doc.get('label', ''),
                    "url": doc.get('url', ''),
                    "type": "dataset"
                }
                
                # Upsert immediately
                store.upsert_documents([doc_data], [emb.tolist()], namespace="dataset")
                indexed += 1
                
                # Progress
                print(f"[index] Doc {i}: Indexed (total: {indexed})")
                    
            except Exception as e:
                print(f"[index] Error at {i}: {e}")
                skipped += 1
    
    print("")
    print("DONE!")
    print(f"Indexed: {indexed}")
    print(f"Skipped: {skipped}")
    print(f"Final stats: {store.get_stats()}")


if __name__ == "__main__":
    # Get limit from command line
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except:
            pass
    
    index_dataset(limit=limit, clear_first=True)
