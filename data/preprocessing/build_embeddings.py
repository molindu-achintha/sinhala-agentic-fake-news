"""
Build embeddings and FAISS index from processed data using HF API with Rate Limiting.
"""
import json
import os
import sys
import numpy as np
import time
import pickle
from tqdm import tqdm
from tenacity import RetryError
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.agents.langproc_agent import LangProcAgent
from app.store.vector_store import VectorStore
from app.config import get_settings

def build_embeddings():
    # Explicitly load .env from root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    load_dotenv(os.path.join(root_dir, '.env'))
    
    input_file = 'data/dataset/processed.jsonl'
    settings = get_settings()
    
    if not os.path.exists(input_file):
        print(f"Processed file not found: {input_file}")
        return

    # Check key presence immediately
    if not settings.OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY is missing. Check your .env file.")
        return
    else:
        print(f"✅ Loaded API Key: {settings.OPENROUTER_API_KEY[:4]}...****")

    lang_proc = LangProcAgent()
    
    documents = []
    embeddings_list = []
    
    print("Reading documents...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
    print(f"Processing {len(all_lines)} documents via HF API...")
    
    # Process in chunks to handle saving partial progress if needed? 
    # Or just loop with sleep.
    
    for line in tqdm(all_lines):
        doc = json.loads(line)
        text = doc['text']
        
        try:
            emb = lang_proc.get_embeddings(text)
            embeddings_list.append(emb)
            documents.append(doc)
            # Gentle rate limiting
            time.sleep(0.2) 
        except RetryError as re:
            # Unwrap the actual exception from the retry wrapper
            actual_exception = re.last_attempt.exception()
            print(f"\n❌ Error processing doc {doc.get('id')}: {actual_exception}")
            # If authorized failed, stop.
            if "401" in str(actual_exception):
                print("Stopping due to Authorization Error.")
                break
        except Exception as e:
            print(f"\nError processing doc {doc.get('id')}: {e}")
            continue
    
    if not embeddings_list:
        print("No documents successfully embedded.")
        return

    embeddings = np.array(embeddings_list)
    
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
    
    store = VectorStore(index_path=settings.FAISS_INDEX_PATH)
    store.index_build(documents, embeddings)
    store.save_index()
    print(f"Index built with {len(documents)} documents.")

if __name__ == "__main__":
    build_embeddings()
