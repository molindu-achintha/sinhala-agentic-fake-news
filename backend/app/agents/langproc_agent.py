"""
langproc_agent.py

Language Processing Agent.
Generates embeddings for text using OpenRouter API.
Uses multilingual e5 large model which supports Sinhala (1024 dimensions).
Caches embeddings in Redis for faster retrieval.
"""
import requests
import numpy as np
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..store.memory_store import get_memory_manager


class LangProcAgent:
    """
    Agent for generating text embeddings.
    Uses OpenRouter API with multilingual e5 large model.
    Caches embeddings in Redis for reuse.
    """
    
    def __init__(self):
        """Set up the agent with API settings."""
        settings = get_settings()
        
        # API settings
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.model_name = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENROUTER_API_KEY
        self.dimension = settings.EMBEDDING_DIMENSION
        
        # Request headers
        self.headers = {
            "Authorization": "Bearer " + (self.api_key or ""),
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sinhala Fake News Detector"
        }
        
        # Memory manager for caching
        self.memory = None
        
        print("[LangProcAgent] Model:", self.model_name)
        print("[LangProcAgent] Dimension:", self.dimension)
    
    def _get_memory(self):
        """Lazy load memory manager."""
        if self.memory is None:
            try:
                self.memory = get_memory_manager()
            except:
                pass
        return self.memory

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        Returns numpy array with 1024 dimensions.
        Checks cache first for faster retrieval.
        """
        print("[LangProcAgent] Embedding text:", text[:50])
        
        # Check cache first
        memory = self._get_memory()
        if memory:
            cached = memory.get_embedding(text)
            if cached:
                print("[LangProcAgent] Using cached embedding")
                return np.array(cached, dtype='float32')
        
        # Return dummy if no API key
        if not self.api_key:
            print("[LangProcAgent] No API key, using dummy embedding")
            return np.random.rand(self.dimension).astype('float32')
        
        # Call API
        payload = {"model": self.model_name, "input": text}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        # Check response
        if response.status_code != 200:
            print("[LangProcAgent] API error:", response.status_code)
            raise Exception("API Error: " + str(response.status_code))
        
        # Get embedding from response
        result = response.json()
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            print("[LangProcAgent] Got embedding, dim:", len(embedding))
            
            # Cache the embedding
            if memory:
                memory.cache_embedding(text, embedding)
                print("[LangProcAgent] Cached embedding")
            
            return np.array(embedding, dtype='float32')
        
        raise Exception("Bad API response")

    def preprocess_text(self, text: str) -> str:
        """Clean text before embedding."""
        if not text:
            return ""
        text = ' '.join(text.split())
        return text.strip()
