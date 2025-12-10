"""
FAISS backed store with methods.
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict

class VectorStore:
    def __init__(self, index_path: str = None, dimension: int = 1536):
        self.index_path = index_path
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product matching cosine similarity if normalized
        self.documents = []  # In-memory metadata store
    
    def load_index(self):
        if self.index_path and os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            # Load metadata - assumption: saved alongside
            meta_path = self.index_path + ".meta"
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.documents = pickle.load(f)
        else:
            print("No existing index found. Starting fresh.")

    def save_index(self):
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            with open(self.index_path + ".meta", 'wb') as f:
                pickle.dump(self.documents, f)

    def add_documents(self, embeddings: np.ndarray, docs: List[Dict]):
        if len(docs) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        faiss.normalize_L2(embedding)
        # embedding shape needs to be (1, dim) if single query
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding, axis=0)
            
        distances, indices = self.index.search(embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                doc['score'] = float(distances[0][i])
                results.append(doc)
        return results

    def index_build(self, documents: List[Dict], embeddings: np.ndarray):
        """Rebuild index from scratch"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.add_documents(embeddings, documents)
