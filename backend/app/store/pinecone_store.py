"""
pinecone_store.py - Pinecone Vector Database Interface

This module provides the interface to Pinecone cloud vector database.
It handles:
1. Index initialization and connection
2. Document upserting (adding/updating)
3. Semantic similarity search
4. Namespace management (dataset and live_news)

The index uses cosine similarity with 1536-dimensional vectors
from text-embedding-3-small model.
"""

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import os
from datetime import datetime
import hashlib


class PineconeVectorStore:
    """
    Pinecone vector store for semantic search.
    
    This class manages all interactions with Pinecone:
    - Creating and connecting to indexes
    - Adding documents with embeddings
    - Searching for similar documents
    """
    
    def __init__(
        self, 
        api_key: str = None,
        index_name: str = "news-store",
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (from .env if not provided)
            index_name: Name of the Pinecone index
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            metric: Similarity metric (cosine, euclidean, or dotproduct)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.pc = None
        self.index = None
        
        print("[PineconeStore] Initializing with index:", index_name)
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not set. Please add it to your .env file.")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone client and create index if needed."""
        print("[PineconeStore] Connecting to Pinecone")
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        print("[PineconeStore] Existing indexes:", existing_indexes)
        
        if self.index_name not in existing_indexes:
            print("[PineconeStore] Creating new index:", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("[PineconeStore] Index created successfully")
        
        self.index = self.pc.Index(self.index_name)
        print("[PineconeStore] Connected to index:", self.index_name)
    
    def generate_id(self, text: str, source: str = "") -> str:
        """
        Generate unique ID from text content.
        
        Args:
            text: Document text
            source: Document source
            
        Returns:
            16-character hash ID
        """
        content = f"{source}:{text}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def upsert_documents(
        self, 
        documents: List[Dict], 
        embeddings: List[List[float]],
        namespace: str = "default"
    ) -> int:
        """
        Add or update documents in Pinecone.
        
        Args:
            documents: List of dicts with text, source, label, etc.
            embeddings: List of embedding vectors (1536-dim each)
            namespace: Namespace to store in (dataset or live_news)
        
        Returns:
            Number of documents upserted
        """
        print("[PineconeStore] Upserting", len(documents), "documents to namespace:", namespace)
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Prepare vectors for upsert
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.get('id') or self.generate_id(doc.get('text', ''), doc.get('source', ''))
            
            # Metadata (Pinecone supports up to 40KB per vector)
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
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            print("[PineconeStore] Upserted", upserted, "/", len(vectors), "vectors")
        
        return upserted
    
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
            query_embedding: Query vector (1536-dim)
            top_k: Number of results to return
            namespace: Namespace to search (None for all)
            filter_dict: Metadata filters
        
        Returns:
            List of matching documents with similarity scores
        """
        print("[PineconeStore] Searching with top_k:", top_k, "namespace:", namespace)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or "",
            filter=filter_dict,
            include_metadata=True
        )
        
        # Format results
        documents = []
        for match in results.matches:
            doc = {
                "id": match.id,
                "score": float(match.score),
                "similarity": f"{match.score * 100:.1f}%",
                **match.metadata
            }
            documents.append(doc)
        
        print("[PineconeStore] Found", len(documents), "results")
        return documents
    
    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with total vectors and namespace info
        """
        print("[PineconeStore] Getting index stats")
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "dimension": stats.dimension
        }
    
    def delete_namespace(self, namespace: str):
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to clear
        """
        print("[PineconeStore] Deleting namespace:", namespace)
        self.index.delete(delete_all=True, namespace=namespace)
        print("[PineconeStore] Namespace deleted")
    
    def delete_index(self):
        """Delete the entire index."""
        print("[PineconeStore] Deleting index:", self.index_name)
        self.pc.delete_index(self.index_name)
        print("[PineconeStore] Index deleted")


# Singleton instance for reuse
_pinecone_store = None


def get_pinecone_store() -> PineconeVectorStore:
    """
    Get or create the Pinecone vector store instance.
    
    This function returns a singleton instance to avoid
    creating multiple connections to Pinecone.
    
    Returns:
        PineconeVectorStore instance
    """
    global _pinecone_store
    if _pinecone_store is None:
        print("[PineconeStore] Creating new store instance")
        _pinecone_store = PineconeVectorStore()
    return _pinecone_store
