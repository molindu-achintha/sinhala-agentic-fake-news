"""
hybrid_retriever.py

Hybrid Evidence Retrieval Agent.
Retrieves evidence from Vector DB and separates labeled from unlabeled.
"""
from typing import Dict, List, Optional
import numpy as np
from duckduckgo_search import DDGS

from ..store.pinecone_store import get_pinecone_store
from .langproc_agent import LangProcAgent


class HybridRetriever:
    """
    Retrieves evidence from Vector DB.
    Separates labeled (verified) from unlabeled evidence.
    """
    
    # Similarity thresholds
    # Similarity thresholds
    HIGH_SIMILARITY = 0.92
    MEDIUM_SIMILARITY = 0.80
    LOW_SIMILARITY = 0.75
    
    def __init__(self):
        """Initialize retriever with stores."""
        print("[HybridRetriever] Initializing")
        self.lang_proc = LangProcAgent()
        self.vector_store = get_pinecone_store()
        print("[HybridRetriever] Ready")
    
    def retrieve(
        self, 
        claim: str, 
        decomposed: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        Retrieve evidence from all sources.
        """
        print("[HybridRetriever] Starting retrieval")
        
        vector_query = decomposed.get("vector_query", "")
        web_query = decomposed.get("web_query", "")
        english_web_query = decomposed.get("english_web_query", "")
        
        # 1. Get embedding
        emb_vector = self.lang_proc.get_embeddings(vector_query)
        
        # 2. Search Vector DB (dataset namespace where data is indexed)
        labeled_results = self._search_namespace(emb_vector, "dataset", top_k=10)
        unlabeled_results = []  # No separate unlabeled namespace currently
        
        # Combine and Filter Results
        all_db_results = labeled_results + unlabeled_results
        filtered_db_results = []
        
        # Filter Logic: Remove low quality social media unless high match
        for doc in all_db_results:
            source = doc.get("source", "").lower()
            score = doc.get("score", 0)
            
            # Strict filter for social media
            if any(s in source for s in ["twitter", "facebook", "whatsapp", "social"]):
                if score < 0.88: # Very strict threshold for social media
                    continue
            
            filtered_db_results.append(doc)
            
        labeled_history = filtered_db_results  # All dataset results are labeled
        unlabeled_context = []
        
        # Calculate text similarity
        top_similarity = 0
        if filtered_db_results:
            top_similarity = max(doc.get("score", 0) for doc in filtered_db_results)
        
        # 3. Web Search (if needed)
        needs_web = self._should_search_web(decomposed, top_similarity)
        
        # Always search English web if we have a query (cross-lingual verification)
        if english_web_query:
            needs_web = True
            
        web_results = []
        
        if needs_web:
            # Search Sinhala Query
            print(f"[HybridRetriever] Web Search (LK): {web_query}")
            try:
                lk_results = self._perform_web_search(web_query, region="lk-en")
                web_results.extend(lk_results)
            except Exception as e:
                print(f"[HybridRetriever] Web search (LK) failed: {e}")
                
            # Search English Query (Global contexts)
            if english_web_query:
                print(f"[HybridRetriever] Web Search (Global): {english_web_query}")
                try:
                    en_results = self._perform_web_search(english_web_query, region="wt-wt")
                    # Add flag to indicate translated source
                    for res in en_results:
                        res['is_translated'] = True
                    web_results.extend(en_results)
                except Exception as e:
                    print(f"[HybridRetriever] Web search (Global) failed: {e}")
        
        # Add web results to context
        formatted_web = []
        for res in web_results:
            formatted_web.append({
                "text": res.get("body", "") or res.get("title", ""),
                "source": res.get("title", "Web Source"),
                "url": res.get("href", ""),
                "score": 0.82 if res.get('is_translated') else 0.80, # Boost english sources slightly
                "label": "unverified"
            })
            
        # Add to unlabeled context
        unlabeled_context.extend(formatted_web)
        
        result = {
            "labeled_history": labeled_history,  # All labeled results
            "unlabeled_context": unlabeled_context,  # All web results
            "web_results": web_results,
            "web_count": len(web_results),
            "top_similarity": top_similarity,
            "similarity_level": self._get_similarity_level(top_similarity),
            "total_evidence": len(filtered_db_results) + len(web_results),
            "labeled_count": len(labeled_history),
            "unlabeled_count": len(unlabeled_context)
        }
        
        print(f"[HybridRetriever] Final: {len(labeled_history)} labeled, {len(unlabeled_context)} context")
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
        # Always search for recent events
        if decomposed.get("temporal_type") == "recent":
            return True
        
        # Search if low similarity from DB
        if top_similarity < self.LOW_SIMILARITY:
            return True
        
        # ALWAYS search for general knowledge claims (like capital cities, facts)
        # This helps verify claims not in our database
        if decomposed.get("temporal_type") == "general":
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

    def _perform_web_search(self, query: str, region: str = "lk-en") -> List[Dict]:
        """Execute web search using DuckDuckGo."""
        results = []
        try:
            with DDGS() as ddgs:
                # Search for news references
                search_results = ddgs.text(
                    query, 
                    region=region,
                    safesearch="off", 
                    max_results=5
                )
                
                if search_results:
                    results = list(search_results)
                    
        except Exception as e:
            print(f"[HybridRetriever] DDGS Error: {e}")
            
        return results
