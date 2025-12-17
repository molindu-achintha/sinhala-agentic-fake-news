"""
hybrid_retriever.py - Hybrid Evidence Retrieval

Retrieves evidence from multiple sources:
1. Vector DB (labeled history + unlabeled context)
2. Web search (for recent events) - optional

Returns categorized evidence for cross-examination.
"""
from typing import Dict, List, Optional
import numpy as np

from ..store.pinecone_store import PineconeVectorStore
from .langproc_agent import LangProcAgent


class HybridRetriever:
    """
    Retrieves evidence from Vector DB and Web.
    Separates labeled (verified) from unlabeled evidence.
    """
    
    # Similarity thresholds
    HIGH_SIMILARITY = 0.90   # Direct verdict possible
    MEDIUM_SIMILARITY = 0.70  # Need cross-reference
    LOW_SIMILARITY = 0.50     # Weak match
    
    def __init__(self):
        """Initialize retriever with stores."""
        print("[HybridRetriever] Initializing...")
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
            Dict with labeled_history, unlabeled_context, web_results
        """
        print("[HybridRetriever] Starting retrieval")
        
        # Generate embedding for claim
        query = decomposed.get("vector_query", claim)
        embedding = self.lang_proc.get_embeddings(query)
        
        # Search both namespaces
        dataset_results = self._search_namespace(embedding, "dataset", top_k)
        live_news_results = self._search_namespace(embedding, "live_news", top_k)
        
        # Combine all results
        all_results = dataset_results + live_news_results
        
        # Separate labeled vs unlabeled
        labeled_history = []
        unlabeled_context = []
        
        for doc in all_results:
            label = doc.get("label", "").lower()
            if label in ["true", "false", "misleading", "fake"]:
                labeled_history.append(doc)
            else:
                unlabeled_context.append(doc)
        
        # Sort by similarity
        labeled_history.sort(key=lambda x: x.get("score", 0), reverse=True)
        unlabeled_context.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Get top similarity
        top_similarity = 0
        if all_results:
            top_similarity = max(doc.get("score", 0) for doc in all_results)
        
        # Determine if web search needed
        needs_web = self._should_search_web(decomposed, top_similarity)
        
        # Web search placeholder (can be implemented later)
        web_results = []
        if needs_web:
            print("[HybridRetriever] Web search would be triggered")
            # web_results = self._search_web(decomposed["web_query"])
        
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
        
        print(f"[HybridRetriever] Found {len(labeled_history)} labeled, {len(unlabeled_context)} unlabeled")
        print(f"[HybridRetriever] Top similarity: {top_similarity:.2%}")
        
        return result
    
    def _search_namespace(
        self, 
        embedding: np.ndarray, 
        namespace: str, 
        top_k: int
    ) -> List[Dict]:
        """Search a specific namespace."""
        print(f"[HybridRetriever] Searching namespace: {namespace}")
        
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
        # Web search if:
        # 1. Temporal type is "recent"
        # 2. Low similarity (no good DB matches)
        
        if decomposed.get("temporal_type") == "recent":
            return True
        
        if top_similarity < self.LOW_SIMILARITY:
            return True
        
        return False
    
    def _get_similarity_level(self, score: float) -> str:
        """Categorize similarity score."""
        if score >= self.HIGH_SIMILARITY:
            return "high"
        elif score >= self.MEDIUM_SIMILARITY:
            return "medium"
        elif score >= self.LOW_SIMILARITY:
            return "low"
        else:
            return "none"
