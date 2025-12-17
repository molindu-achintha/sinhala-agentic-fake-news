"""
langproc_agent.py - Language Processing Agent

This agent generates embeddings for text using OpenRouter API.
Uses multilingual-e5-large model which supports Sinhala (1024 dimensions).
"""
import requests
import numpy as np
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings


class LangProcAgent:
    """
    Agent for generating text embeddings.
    Uses OpenRouter API with multilingual-e5-large model.
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sinhala Fake News Detector"
        }
        
        print("[LangProcAgent] Model:", self.model_name)
        print("[LangProcAgent] Dimension:", self.dimension)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        Returns numpy array with 1024 dimensions.
        """
        print("[LangProcAgent] Embedding text:", text[:50], "...")
        
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
            raise Exception(f"API Error: {response.status_code}")
        
        # Get embedding from response
        result = response.json()
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            print("[LangProcAgent] Got embedding, dim:", len(embedding))
            return np.array(embedding, dtype='float32')
        
        raise Exception("Bad API response")

    def preprocess_text(self, text: str) -> str:
        """Clean text before embedding."""
        if not text:
            return ""
        text = ' '.join(text.split())
        return text.strip()
