"""
claim_extractor.py - Claim Extractor Agent

This agent extracts the main verifiable claim from input text.
It identifies the factual statement that can be checked against evidence.
"""
import re
from ..utils.sin_tokenizer import split_sentences


class ClaimExtractorAgent:
    """
    Agent for extracting claims from text.
    
    This agent takes raw input text and identifies the main
    factual claim that can be verified.
    """
    
    def __init__(self):
        """Initialize the claim extractor agent."""
        print("[ClaimExtractor] Initialized")
    
    def extract_claim(self, raw_text: str) -> dict:
        """
        Extract the main claim from text.
        
        Args:
            raw_text: The raw input text from user
            
        Returns:
            Dictionary with claim_text, claim_span, and confidence
        """
        print("[ClaimExtractor] Extracting claim from text")
        print("[ClaimExtractor] Input length:", len(raw_text))
        
        # Split text into sentences
        sentences = split_sentences(raw_text)
        print("[ClaimExtractor] Found", len(sentences), "sentences")
        
        # Heuristic: First sentence is often the headline/claim
        # In news articles, the first sentence usually contains the main claim
        # For future improvement: Use an LLM to identify the main claim
        
        if sentences:
            extracted_claim = sentences[0]
        else:
            extracted_claim = raw_text
        
        # Clean up the claim
        extracted_claim = extracted_claim.strip()
        
        # Calculate confidence
        # Higher confidence if text is short (likely already a claim)
        if len(raw_text) < 200:
            confidence = 0.90
        elif len(sentences) == 1:
            confidence = 0.85
        else:
            confidence = 0.75
        
        # Find span in original text
        try:
            start_idx = raw_text.index(extracted_claim)
            end_idx = start_idx + len(extracted_claim)
        except ValueError:
            start_idx = 0
            end_idx = len(raw_text)
        
        print("[ClaimExtractor] Extracted claim:", extracted_claim[:50], "...")
        print("[ClaimExtractor] Confidence:", confidence)
        
        return {
            "claim_text": extracted_claim,
            "claim_span": [start_idx, end_idx],
            "confidence": confidence,
            "original_length": len(raw_text),
            "sentence_count": len(sentences)
        }
    
    def is_factual_claim(self, text: str) -> bool:
        """
        Check if text contains a factual claim.
        
        Factual claims can be verified. Opinions cannot.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a factual claim
        """
        # Check for opinion keywords
        opinion_keywords = [
            "මම හිතනවා",       # I think
            "මගේ මතය",        # My opinion
            "I think",
            "I believe",
            "in my opinion"
        ]
        
        text_lower = text.lower()
        for keyword in opinion_keywords:
            if keyword.lower() in text_lower:
                print("[ClaimExtractor] Text appears to be an opinion")
                return False
        
        print("[ClaimExtractor] Text appears to be a factual claim")
        return True
