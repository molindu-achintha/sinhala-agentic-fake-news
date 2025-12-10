"""
Language processing agent: Tokenization, Embeddings via OpenRouter API.
"""
import requests
import numpy as np
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.sin_tokenizer import tokenize
from ..utils.transliteration import sinhala_to_latin
from ..config import get_settings

class LangProcAgent:
    def __init__(self):
        settings = get_settings()
        # OpenRouter API (OpenAI-compatible)
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.model_name = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENROUTER_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080", # Required by OpenRouter
            "X-Title": "Sinhala Fake News Detector"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings from OpenRouter API. Returns 1D numpy array.
        """
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not set. Returning dummy embedding.")
            return np.random.rand(1536).astype('float32') # text-embedding-3-small is 1536 dim

        payload = {
            "model": self.model_name,
            "input": text
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
            
        result = response.json()
        
        # OpenRouter returns: {"data": [{"embedding": [...]}]}
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            return np.array(embedding, dtype='float32')
            
        raise Exception(f"Unexpected API response format: {result}")

    def process_text(self, text: str) -> dict:
        tokens = tokenize(text)
        translit = sinhala_to_latin(text)
        return {
            "tokens": tokens,
            "transliterated": translit
        }
