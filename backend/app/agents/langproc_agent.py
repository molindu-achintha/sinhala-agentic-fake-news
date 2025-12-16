"""
langproc_agent.py - Language Processing Agent

This agent handles all language-related tasks:
1. Text preprocessing (cleaning, normalization)
2. Embedding generation (converting text to vectors)

It uses OpenRouter API for embedding generation with the
text-embedding-3-small model (1536 dimensions).
"""
import requests
import numpy as np
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.sin_tokenizer import tokenize
from ..utils.transliteration import sinhala_to_latin
from ..config import get_settings


class LangProcAgent:
    """
    Agent for language processing tasks.
    
    This agent converts text into embeddings using OpenAI's
    text-embedding-3-small model via OpenRouter API.
    """
    
    def __init__(self):
        """Initialize the language processing agent."""
        settings = get_settings()
        
        # OpenRouter API configuration
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.model_name = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENROUTER_API_KEY
        
        # HTTP headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sinhala Fake News Detector"
        }
        
        print("[LangProcAgent] Initialized with model:", self.model_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings from OpenRouter API.
        
        This method converts text into a 1536-dimensional vector
        that can be used for similarity search.
        
        Args:
            text: The input text to embed
            
        Returns:
            numpy array with 1536 dimensions
        """
        print("[LangProcAgent] Generating embedding for text:", text[:50], "...")
        
        # Check if API key is set
        if not self.api_key:
            print("[LangProcAgent] Warning: OPENROUTER_API_KEY not set")
            print("[LangProcAgent] Returning dummy embedding")
            return np.random.rand(1536).astype('float32')
        
        # Prepare API request
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        # Call OpenRouter API
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            print("[LangProcAgent] API Error:", response.status_code)
            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
        
        # Parse response
        result = response.json()
        
        # Extract embedding from response
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            print("[LangProcAgent] Embedding generated, dimension:", len(embedding))
            return np.array(embedding, dtype='float32')
        
        raise Exception(f"Unexpected API response format: {result}")

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        print("[LangProcAgent] Text preprocessed, length:", len(text))
        return text

    def process_text(self, text: str) -> dict:
        """
        Process text for analysis.
        
        This method tokenizes the text and creates a
        transliteration for debugging purposes.
        
        Args:
            text: Input text in Sinhala
            
        Returns:
            Dictionary with tokens and transliteration
        """
        tokens = tokenize(text)
        translit = sinhala_to_latin(text)
        
        print("[LangProcAgent] Text processed, tokens:", len(tokens))
        
        return {
            "tokens": tokens,
            "transliterated": translit
        }
