"""
Identifies the primary checkable claim (single sentence).
"""
import re
from ..utils.sin_tokenizer import split_sentences

class ClaimExtractorAgent:
    def __init__(self):
        # Placeholder for a seq2seq model or LLM client
        pass

    def extract_claim(self, raw_text: str) -> dict:
        """
        Extracts the main claim from the text.
        Returns { "claim_text": "...", "claim_span": [start, end], "confidence": ... }
        """
        sentences = split_sentences(raw_text)
        
        # Heuristic: First sentence is often the headline/claim in news
        # Improved Heuristic: Look for sentences with specific keywords if available
        # In a real scenario, we call an LLM here.
        
        prompt = "මෙම පෙළෙන් එකම පරීක්ෂා කළ යුතු ප්‍රකාශයක් සරලව (සිංහල) 1 වාක්‍යක් ලෙස දක්වන්න. ඔබට එය සනාථ/විරෝධියා/අර්ධ-විරෝධියා බව පැවසීමට අවශ්‍ය නම් එම වචන පමණක් එවන්න."
        
        # Simulating extraction
        extracted_claim = sentences[0] if sentences else raw_text
        confidence = 0.85 
        
        # Mock logic for "span"
        try:
            start_idx = raw_text.index(extracted_claim)
            end_idx = start_idx + len(extracted_claim)
        except ValueError:
            start_idx = 0
            end_idx = len(raw_text)
            
        return {
            "claim_text": extracted_claim,
            "claim_span": [start_idx, end_idx],
            "confidence": confidence
        }
