"""
retrieval_agent.py - Retrieval Agent

This agent searches for evidence documents related to a claim.
It uses the Pinecone vector database to find semantically similar content.
"""
from typing import List

from ..store.pinecone_store import PineconeVectorStore, get_pinecone_store
from .langproc_agent import LangProcAgent
from ..config import get_settings


class RetrievalAgent:
    """
    Agent for retrieving evidence from the vector database.
    
    This agent takes a claim text, converts it to an embedding,
    and searches Pinecone for similar documents.
    """
    
    def __init__(self, lang_proc: LangProcAgent = None):
        """
        Initialize the retrieval agent.
        
        Args:
            lang_proc: Language processing agent for embeddings
        """
        settings = get_settings()
        self.lang_proc = lang_proc or LangProcAgent()
        
        # Initialize Pinecone store
        try:
            self.pinecone_store = get_pinecone_store()
            self.use_pinecone = True
            print("[RetrievalAgent] Initialized with Pinecone")
        except Exception as e:
            print("[RetrievalAgent] Warning: Could not connect to Pinecone:", str(e))
            self.pinecone_store = None
            self.use_pinecone = False
    
    def retrieve_evidence(self, claim_text: str, top_k: int = 5) -> List[dict]:
        """
        Retrieve evidence documents for a claim.
        
        Args:
            claim_text: The claim text to search for
            top_k: Number of results to return
            
        Returns:
            List of evidence documents with similarity scores
        """
        print("[RetrievalAgent] Retrieving evidence for claim")
        print("[RetrievalAgent] Claim text:", claim_text[:50], "...")
        
        # Generate embedding for the claim
        embedding = self.lang_proc.get_embeddings(claim_text)
        print("[RetrievalAgent] Embedding generated")
        
        # Search Pinecone if available
        if self.use_pinecone and self.pinecone_store:
            try:
                # Search in dataset namespace
                print("[RetrievalAgent] Searching dataset namespace")
                dataset_results = self.pinecone_store.search(
                    query_embedding=embedding.tolist(),
                    top_k=top_k,
                    namespace="dataset"
                )
                
                # Search in live_news namespace
                print("[RetrievalAgent] Searching live_news namespace")
                news_results = self.pinecone_store.search(
                    query_embedding=embedding.tolist(),
                    top_k=top_k,
                    namespace="live_news"
                )
                
                # Combine and sort by score
                all_results = dataset_results + news_results
                all_results.sort(key=lambda x: x['score'], reverse=True)
                
                # Return top_k results
                results = all_results[:top_k]
                print("[RetrievalAgent] Found", len(results), "results")
                return results
                
            except Exception as e:
                print("[RetrievalAgent] Pinecone search failed:", str(e))
                return []
        
        print("[RetrievalAgent] Pinecone not available, returning empty results")
        return []
    
    def search_namespace(self, claim_text: str, namespace: str, top_k: int = 5) -> List[dict]:
        """
        Search a specific namespace.
        
        Args:
            claim_text: The claim text to search for
            namespace: Namespace to search (dataset or live_news)
            top_k: Number of results to return
            
        Returns:
            List of evidence documents
        """
        print("[RetrievalAgent] Searching namespace:", namespace)
        
        if not self.use_pinecone or not self.pinecone_store:
            print("[RetrievalAgent] Pinecone not available")
            return []
        
        # Generate embedding
        embedding = self.lang_proc.get_embeddings(claim_text)
        
        # Search
        results = self.pinecone_store.search(
            query_embedding=embedding.tolist(),
            top_k=top_k,
            namespace=namespace
        )
        
        print("[RetrievalAgent] Found", len(results), "results in", namespace)
        return results
