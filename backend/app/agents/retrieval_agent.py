"""
Retrieval agent using FAISS.
"""
from typing import List
from ..store.vector_store import VectorStore
from .langproc_agent import LangProcAgent
from ..config import get_settings

class RetrievalAgent:
    def __init__(self, lang_proc: LangProcAgent):
        settings = get_settings()
        self.vector_store = VectorStore(index_path=settings.FAISS_INDEX_PATH)
        self.vector_store.load_index()
        self.lang_proc = lang_proc

    def retrieve_evidence(self, claim_text: str, top_k: int = 5) -> List[dict]:
        # Generate embedding
        embedding = self.lang_proc.get_embeddings(claim_text)
        
        # Search
        results = self.vector_store.search(embedding, top_k=top_k)
        
        # In a real system, we might do query expansion here (Sinhala + English)
        
        return results
