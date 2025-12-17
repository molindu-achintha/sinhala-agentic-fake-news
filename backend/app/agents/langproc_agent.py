"""
langproc_agent.py

Language Processing Agent.
Generates embeddings for text using:
1. OpenRouter API (primary)
2. Pinecone Inference API (fallback)

Uses multilingual e5 large model which supports Sinhala (1024 dimensions).
Caches embeddings in Redis for faster retrieval.
"""
import requests
import numpy as np
from typing import Optional

from ..config import get_settings
from ..store.memory_store import get_memory_manager


class LangProcAgent:
    """
    Agent for generating text embeddings.
    Uses OpenRouter and Pinecone embedding providers with fallback support.
    Caches embeddings in Redis for reuse.
    """
    
    def __init__(self):
        """Set up the agent with API settings."""
        settings = get_settings()
        
        # API settings
        self.openrouter_url = "https://openrouter.ai/api/v1/embeddings"
        self.model_name = settings.EMBEDDING_MODEL
        self.openrouter_key = settings.OPENROUTER_API_KEY
        self.pinecone_key = settings.PINECONE_API_KEY
        self.dimension = settings.EMBEDDING_DIMENSION
        
        # OpenRouter headers
        self.openrouter_headers = {
            "Authorization": "Bearer " + (self.openrouter_key or ""),
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sinhala Fake News Detector"
        }
        
        # Memory manager for caching
        self.memory = None
        
        # Get configured provider (auto, openrouter, pinecone)
        configured_provider = settings.EMBEDDING_PROVIDER.lower()
        
        # Set initial provider based on config
        if configured_provider == "auto":
            self._provider = "openrouter"  # Auto mode starts with openrouter
            self._auto_fallback = True
        else:
            self._provider = configured_provider
            self._auto_fallback = False  # Don't fallback if user specified provider
        
        print("[LangProcAgent] Model:", self.model_name)
        print("[LangProcAgent] Dimension:", self.dimension)
        print("[LangProcAgent] Provider:", self._provider, "(auto-fallback:", self._auto_fallback, ")")
    
    def _get_memory(self):
        """Lazy load memory manager."""
        if self.memory is None:
            try:
                self.memory = get_memory_manager()
            except:
                pass
        return self.memory

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        Returns numpy array with specified dimensions.
        Tries providers with fallback if configured.
        """
        print("[LangProcAgent] Embedding text:", text[:50])
        
        # Check cache first
        memory = self._get_memory()
        if memory:
            cached = memory.get_embedding(text)
            if cached:
                print("[LangProcAgent] Using cached embedding")
                return np.array(cached, dtype='float32')
        
        # Try providers based on current provider setting
        embedding = None
        
        # Try the configured/current provider
        if self._provider == "openrouter":
            if self.openrouter_key:
                embedding = self._try_openrouter(text)
            if embedding is None and self._auto_fallback:
                print("[LangProcAgent] OpenRouter failed, switching to Pinecone")
                self._provider = "pinecone"
        
        if self._provider == "pinecone" and embedding is None:
            if self.pinecone_key:
                embedding = self._try_pinecone(text)
        
        # Last resort: random embedding
        if embedding is None:
            print("[LangProcAgent] All providers failed, using random embedding")
            return np.random.rand(self.dimension).astype('float32')
        
        # Cache the embedding
        if memory and embedding is not None:
            try:
                memory.cache_embedding(text, embedding.tolist())
                print("[LangProcAgent] Cached embedding")
            except:
                pass
        
        return embedding
    
    def _try_openrouter(self, text: str) -> Optional[np.ndarray]:
        """Try to get embedding from OpenRouter."""
        try:
            payload = {"model": self.model_name, "input": text}
            response = requests.post(
                self.openrouter_url, 
                headers=self.openrouter_headers, 
                json=payload,
                timeout=15
            )
            
            if response.status_code == 402:
                print("[LangProcAgent] OpenRouter: Payment required (402)")
                return None
            
            if response.status_code != 200:
                print(f"[LangProcAgent] OpenRouter error: {response.status_code}")
                return None
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"[LangProcAgent] OpenRouter: Got embedding, dim: {len(embedding)}")
                return np.array(embedding, dtype='float32')
                
        except Exception as e:
            print(f"[LangProcAgent] OpenRouter exception: {e}")
        
        return None
    
    def _try_pinecone(self, text: str) -> Optional[np.ndarray]:
        """Try to get embedding from Pinecone Inference API."""
        try:
            # Pinecone inference endpoint
            url = "https://api.pinecone.io/embed"
            
            headers = {
                "Api-Key": self.pinecone_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "multilingual-e5-large",
                "inputs": [{"text": text}],
                "parameters": {"input_type": "query"}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code != 200:
                print(f"[LangProcAgent] Pinecone error: {response.status_code}")
                return None
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["values"]
                print(f"[LangProcAgent] Pinecone: Got embedding, dim: {len(embedding)}")
                return np.array(embedding, dtype='float32')
                
        except Exception as e:
            print(f"[LangProcAgent] Pinecone exception: {e}")
        
        return None

    def preprocess_text(self, text: str) -> str:
        """Clean text before embedding."""
        if not text:
            return ""
        text = ' '.join(text.split())
        return text.strip()
