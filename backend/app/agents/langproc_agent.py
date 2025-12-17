"""
langproc_agent.py

Language Processing Agent.
Generates embeddings for text using multiple providers:
1. OpenRouter API (primary)
2. Pinecone Inference API (fallback)
3. Sentence Transformers (local fallback)

Uses multilingual e5 large model which supports Sinhala (1024 dimensions).
Caches embeddings in Redis for faster retrieval.
"""
import requests
import numpy as np
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..store.memory_store import get_memory_manager


class LangProcAgent:
    """
    Agent for generating text embeddings.
    Uses multiple embedding providers with fallback support.
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
        
        # Sentence Transformer model (lazy loaded)
        self._local_model = None
        
        # Track which provider works
        self._provider = "openrouter"  # Start with openrouter
        
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
    
    def _get_local_model(self):
        """Lazy load sentence transformers model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[LangProcAgent] Loading local model: paraphrase-multilingual-mpnet-base-v2")
                self._local_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            except ImportError:
                print("[LangProcAgent] sentence-transformers not installed")
                return None
            except Exception as e:
                print(f"[LangProcAgent] Failed to load local model: {e}")
                return None
        return self._local_model

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        Returns numpy array with specified dimensions.
        Tries multiple providers with fallback.
        """
        print("[LangProcAgent] Embedding text:", text[:50])
        
        # Check cache first
        memory = self._get_memory()
        if memory:
            cached = memory.get_embedding(text)
            if cached:
                print("[LangProcAgent] Using cached embedding")
                return np.array(cached, dtype='float32')
        
        # Try providers in order
        embedding = None
        
        # 1. Try OpenRouter (if we haven't failed with it yet)
        if self._provider == "openrouter" and self.openrouter_key:
            embedding = self._try_openrouter(text)
            if embedding is None:
                print("[LangProcAgent] OpenRouter failed, switching to Pinecone")
                self._provider = "pinecone"
        
        # 2. Try Pinecone Inference API
        if embedding is None and self.pinecone_key:
            embedding = self._try_pinecone(text)
            if embedding is None:
                print("[LangProcAgent] Pinecone failed, switching to local model")
                self._provider = "local"
        
        # 3. Try local sentence transformer
        if embedding is None:
            embedding = self._try_local(text)
        
        # 4. Last resort: random embedding
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
    
    def _try_local(self, text: str) -> Optional[np.ndarray]:
        """Try to get embedding from local sentence transformer."""
        try:
            model = self._get_local_model()
            if model is None:
                return None
            
            embedding = model.encode(text, convert_to_numpy=True)
            print(f"[LangProcAgent] Local: Got embedding, dim: {len(embedding)}")
            
            # Pad or truncate to match expected dimension
            if len(embedding) < self.dimension:
                padding = np.zeros(self.dimension - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            
            return embedding.astype('float32')
            
        except Exception as e:
            print(f"[LangProcAgent] Local exception: {e}")
        
        return None

    def preprocess_text(self, text: str) -> str:
        """Clean text before embedding."""
        if not text:
            return ""
        text = ' '.join(text.split())
        return text.strip()
