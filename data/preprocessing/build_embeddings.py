"""
Build embeddings and FAISS index from processed data.
"""
import json
import os
import sys
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.agents.langproc_agent import LangProcAgent
from app.store.vector_store import VectorStore
from app.config import get_settings

def build_embeddings():
    input_file = 'data/dataset/processed.jsonl'
    settings = get_settings()
    
    if not os.path.exists(input_file):
        print(f"Processed file not found: {input_file}")
        return

    lang_proc = LangProcAgent()
    
    documents = []
    embeddings_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            # Embedding context: Usually we embed the document text or claim
            # Here, let's embed the full text for retrieval relevance
            text = doc['text']
            emb = lang_proc.get_embeddings(text)
            
            embeddings_list.append(emb)
            documents.append(doc)
    
    if not embeddings_list:
        print("No documents found.")
        return

    embeddings = np.array(embeddings_list)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
    
    store = VectorStore(index_path=settings.FAISS_INDEX_PATH)
    store.index_build(documents, embeddings)
    store.save_index()
    print(f"Index built with {len(documents)} documents.")

if __name__ == "__main__":
    build_embeddings()
