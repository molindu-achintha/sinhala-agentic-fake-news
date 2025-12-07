"""
Language processing agent: Tokenization, Embeddings.
"""
from typing import List
import numpy as np
from ..utils.sin_tokenizer import tokenize
from ..utils.transliteration import sinhala_to_latin
from ..config import get_settings

# Attempt to import SentenceTransformer, fallback if not installed (for docker build stage)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class LangProcAgent:
    def __init__(self):
        settings = get_settings()
        self.model_path = settings.MODEL_PATH
        self.encoder = None
        self._load_encoder()

    def _load_encoder(self):
        if SentenceTransformer:
            try:
                print(f"Loading embedding model: {self.model_path}")
                self.encoder = SentenceTransformer(self.model_path)
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.encoder = None
        else:
            print("SentenceTransformers library not found.")

    def get_embeddings(self, text: str) -> np.ndarray:
        if self.encoder:
            # Returns a 1D numpy array for a single string
            return self.encoder.encode(text, convert_to_numpy=True)
        else:
            # Dummy embedding 768 dim
            return np.random.rand(768).astype('float32')

    def process_text(self, text: str) -> dict:
        tokens = tokenize(text)
        translit = sinhala_to_latin(text)
        return {
            "tokens": tokens,
            "transliterated": translit
        }
