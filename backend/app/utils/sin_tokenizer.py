"""
Sinhala tokenizer utility.
"""
import re

def tokenize(text: str) -> list[str]:
    """
    Simple whitespace and punctuation based tokenizer for Sinhala.
    """
    # Remove common punctuation but keep sentence endings for splitting if needed
    text = re.sub(r'[^\w\s\u0D80-\u0DFF]', ' ', text)
    tokens = text.split()
    return tokens

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences based on punctuation.
    """
    # Sinhala uses '.', '?', '!' similar to English
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
