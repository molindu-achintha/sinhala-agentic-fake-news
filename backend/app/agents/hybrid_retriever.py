"""
hybrid_retriever.py

Hybrid Evidence Retrieval Agent.
Retrieves evidence from Vector DB and separates labeled from unlabeled.
"""
from typing import Dict, List, Optional
import numpy as np

from ..store.pinecone_store import PineconeVectorStore
from .langproc_agent import LangProcAgent


class HybridRetriever:
    """
    Retrieves evidence from Vector DB.
    Separates labeled (verified) from unlabeled evidence.
    """
    
    # Similarity thresholds
    HIGH_SIMILARITY = 0.90
    MEDIUM_SIMILARITY = 0.70
    LOW_SIMILARITY = 0.50
    
    def __init__(self):
        """Initialize retriever with stores."""
        print("[HybridRetriever] Initializing")
        self.lang_proc = LangProcAgent()
        self.vector_store = PineconeVectorStore()
        print("[HybridRetriever] Ready")
    
    def retrieve(
        self, 
        claim: str, 
        decomposed: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        Retrieve evidence from all sources.
        
        Args:
            claim: The claim text
            decomposed: Output from ClaimDecomposer
            top_k: Number of results per namespace
        
        Returns:
            Dict with labeled_history unlabeled_context web_results
        """
        print("[HybridRetriever] Starting retrieval")
        
        vector_query = decomposed.get("vector_query", "")
        web_query = decomposed.get("web_query", "")
        
        # 1. Get embedding
        emb_vector = self.encoder.get_embeddings(vector_query)
        
        # 2. Search Vector DB (Labeled & Unlabeled)
        labeled_results = self._search_namespace(emb_vector, "labeled_news", top_k=5)
        unlabeled_results = self._search_namespace(emb_vector, "unlabeled_news", top_k=5)
        
        # Combine for analysis
        all_results = labeled_results + unlabeled_results
        labeled_history = [d for d in all_results if d.get("namespace") == "labeled_news"]
        unlabeled_context = [d for d in all_results if d.get("namespace") == "unlabeled_news"]
        
        # Calculate text similarity
        top_similarity = 0
        if all_results:
            top_similarity = max(doc.get("score", 0) for doc in all_results)
        
        # 3. Web Search (if needed)
        needs_web = self._should_search_web(decomposed, top_similarity)
        web_results = []
        
        if needs_web:
            print(f"[HybridRetriever] Triggering Web Search query: {web_query}")
            try:
                web_results = self._perform_web_search(web_query)
                print(f"[HybridRetriever] Found {len(web_results)} web results")
            except Exception as e:
                print(f"[HybridRetriever] Web search failed: {e}")
        
        # Add web results to context
        # Convert web results to standardized format
        formatted_web = []
        for res in web_results:
            formatted_web.append({
                "text": res.get("body", "") or res.get("title", ""),
                "source": res.get("title", "Web Source"),
                "url": res.get("href", ""),
                "score": 0.8, # Estimated relevance
                "label": "unverified"
            })
            
        # Add to unlabeled context
        unlabeled_context.extend(formatted_web)
        
        result = {
            "labeled_history": labeled_history[:5],
            "unlabeled_context": unlabeled_context[:5],
            "web_results": web_results,
            "top_similarity": top_similarity,
            "similarity_level": self._get_similarity_level(top_similarity),
            "total_evidence": len(all_results),
            "labeled_count": len(labeled_history),
            "unlabeled_count": len(unlabeled_context)
        }
        
        print("[HybridRetriever] Found", len(labeled_history), "labeled", len(unlabeled_context), "unlabeled")
        print("[HybridRetriever] Top similarity:", round(top_similarity * 100), "percent")
        
        return result
    
    def _search_namespace(
        self, 
        embedding: np.ndarray, 
        namespace: str, 
        top_k: int
    ) -> List[Dict]:
        """Search a specific namespace."""
        print("[HybridRetriever] Searching namespace:", namespace)
        
        results = self.vector_store.search(
            query_embedding=embedding.tolist(),
            top_k=top_k,
            namespace=namespace
        )
        
        # Add namespace info to results
        for doc in results:
            doc["namespace"] = namespace
        
        return results
    
    def _should_search_web(self, decomposed: Dict, top_similarity: float) -> bool:
        """Determine if web search is needed."""
        if decomposed.get("temporal_type") == "recent":
            return True
        
        if top_similarity < self.LOW_SIMILARITY:
            return True
        
        return False
    
    def _get_similarity_level(self, score: float) -> str:
        """Categorize similarity score."""
        if score >= self.HIGH_SIMILARITY:
            return "high"
        elif score >= self.LOW_SIMILARITY:
            return "medium"
        else:
            return "low"

    def _perform_web_search(self, query: str) -> List[Dict]:
        """Execute web search using DuckDuckGo."""
        results = []
        try:
            with DDGS() as ddgs:
                # Search for news references
                search_results = ddgs.text(
                    query, 
                    region="lk-en", # Prioritize Sri Lanka
                    safesearch="off", 
                    max_results=5
                )
                
                if search_results:
                    results = list(search_results)
                    
        except Exception as e:
            print(f"[HybridRetriever] DDGS Error: {e}")
            
        return results
