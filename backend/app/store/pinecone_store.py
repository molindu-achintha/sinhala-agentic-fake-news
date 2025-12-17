"""
pinecone_store.py - Pinecone Vector Database

Stores and searches news articles using vector embeddings.
Uses 1024 dimensions (multilingual-e5-large model).
"""
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import os
from datetime import datetime
import hashlib


class PineconeVectorStore:
    """
    Vector store using Pinecone cloud database.
    Handles adding and searching documents.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        index_name: str = "news-store",
        dimension: int = 1024,
        metric: str = "cosine"
    ):
        """
        Set up connection to Pinecone.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the index
            dimension: Vector dimension (1024)
            metric: Similarity metric
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.pc = None
        self.index = None
        
        print("[PineconeStore] Index:", index_name)
        print("[PineconeStore] Dimension:", dimension)
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not set")
        
        self._connect()
    
    def _connect(self):
        """Connect to Pinecone and create index if needed."""
        print("[PineconeStore] Connecting...")
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check existing indexes
        existing = [idx.name for idx in self.pc.list_indexes()]
        print("[PineconeStore] Existing indexes:", existing)
        
        # Create index if not exists
        if self.index_name not in existing:
            print("[PineconeStore] Creating index...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("[PineconeStore] Index created")
        
        self.index = self.pc.Index(self.index_name)
        print("[PineconeStore] Connected")
    
    def generate_id(self, text: str, source: str = "") -> str:
        """Create unique ID from text."""
        content = f"{source}:{text}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def upsert_documents(
        self, 
        documents: List[Dict], 
        embeddings: List[List[float]],
        namespace: str = "default"
    ) -> int:
        """
        Add documents to Pinecone.
        
        Args:
            documents: List of document dicts
            embeddings: List of embedding vectors
            namespace: Namespace (dataset or live_news)
        
        Returns:
            Number of documents added
        """
        print("[PineconeStore] Upserting", len(documents), "docs to", namespace)
        
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must match")
        
        # Build vectors
        vectors = []
        for doc, emb in zip(documents, embeddings):
            doc_id = doc.get('id') or self.generate_id(doc.get('text', ''), doc.get('source', ''))
            
            # Metadata
            metadata = {
                "text": doc.get('text', '')[:1000],
                "title": doc.get('title', '')[:500],
                "source": doc.get('source', 'unknown'),
                "label": doc.get('label', ''),
                "url": doc.get('url', ''),
                "type": doc.get('type', 'dataset'),
                "indexed_at": datetime.now().isoformat()
            }
            
            vectors.append({
                "id": doc_id,
                "values": emb,
                "metadata": metadata
            })
        
        # Upsert in batches
        batch_size = 100
        total = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
            print("[PineconeStore] Upserted", total, "/", len(vectors))
        
        return total
    
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        namespace: str = None,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            namespace: Namespace to search
        
        Returns:
            List of matching documents
        """
        print("[PineconeStore] Searching, top_k:", top_k, "namespace:", namespace)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or "",
            filter=filter_dict,
            include_metadata=True
        )
        
        # Format results
        docs = []
        for match in results.matches:
            doc = {
                "id": match.id,
                "score": float(match.score),
                "similarity": f"{match.score * 100:.1f}%",
                **match.metadata
            }
            docs.append(doc)
        
        print("[PineconeStore] Found", len(docs), "results")
        return docs
    
    def get_stats(self) -> Dict:
        """Get index stats."""
        print("[PineconeStore] Getting stats")
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "dimension": stats.dimension
        }
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in namespace."""
        print("[PineconeStore] Deleting namespace:", namespace)
        self.index.delete(delete_all=True, namespace=namespace)


# Singleton
_store = None

from ..config import get_settings

def get_pinecone_store() -> PineconeVectorStore:
    """Get or create Pinecone store."""
    global _store
    if _store is None:
        print("[PineconeStore] Creating store")
        settings = get_settings()
        _store = PineconeVectorStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION
        )
    return _store
