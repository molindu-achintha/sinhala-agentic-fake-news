"""
claim_decomposer.py

Claim Decomposition Agent.
Breaks down claims into keywords dates and temporal type.
"""
import re
from datetime import datetime
from typing import Dict, List


from deep_translator import GoogleTranslator

class ClaimDecomposer:
    """
    Decomposes claims into searchable components.
    Capable of translating Sinhala to English for broader search coverage.
    """
    
    # Keywords indicating recent events
    RECENT_KEYWORDS = [
        "today", "yesterday", "breaking", "just now", "now",
        "අද", "ඊයේ", "දැන්", "මේ මොහොතේ", "නවතම"
    ]
    
    # Keywords indicating past events
    PAST_KEYWORDS = [
        "in 2019", "in 2020", "in 2021", "in 2022", "in 2023",
        "2019", "2020", "2021", "2022", "2023"
    ]
    
    def __init__(self):
        """Initialize decomposer."""
        print("[ClaimDecomposer] Initialized")
        self.translator = GoogleTranslator(source='auto', target='en')
    
    def decompose(self, claim: str) -> Dict:
        """
        Decompose a claim into searchable components.
        
        Args:
            claim: The raw claim text
        
        Returns:
            Dict with keywords dates temporal_type
        """
        print("[ClaimDecomposer] Decomposing claim")
        
        # Extract year references
        years = self._extract_years(claim)
        
        # Determine temporal type
        temporal_type = self._get_temporal_type(claim, years)
        
        # Extract keywords for search
        keywords = self._extract_keywords(claim)
        
        # Translate to English if Sinhala detected
        english_claim = claim
        english_keywords = keywords
        english_web_query = ""
        
        try:
            # Check if claim contains Sinhala (Unicode range 0D80-0DFF)
            if re.search(r'[\u0D80-\u0DFF]', claim):
                print("[ClaimDecomposer] Translating to English...")
                english_claim = self.translator.translate(claim)
                print("[ClaimDecomposer] Translated:", english_claim)
                
                # Extract English keywords
                english_keywords = self._extract_keywords(english_claim)
                english_web_query = " ".join(english_keywords[:7])
        except Exception as e:
            print(f"[ClaimDecomposer] Translation failed: {e}")
            english_web_query = ""
        
        # Generate search queries
        vector_query = self._create_vector_query(claim, keywords)
        
        # For Sinhala, use the original claim (first 150 chars) for search
        # This ensures the search query is readable and not broken keywords
        sinhala_web_query = claim[:150].strip() if claim else ""
        
        result = {
            "original_claim": claim,
            "translated_claim": english_claim,
            "keywords": keywords,
            "english_keywords": english_keywords,
            "years": years,
            "temporal_type": temporal_type,
            "vector_query": vector_query,
            "web_query": sinhala_web_query,  # Use original claim for Sinhala search
            "english_web_query": english_web_query,
            "needs_web_search": temporal_type == "recent"
        }
        
        print("[ClaimDecomposer] Temporal type:", temporal_type)
        print("[ClaimDecomposer] Keywords:", keywords[:5])
        
        return result
    
    def _extract_years(self, text: str) -> List[int]:
        """Extract year references from text."""
        years = re.findall(r'\b(20[1-2][0-9])\b', text)
        return [int(y) for y in years]
    
    def _get_temporal_type(self, claim: str, years: List[int]) -> str:
        """
        Determine if claim is about recent or historical events.
        
        Returns:
            recent historical or general
        """
        claim_lower = claim.lower()
        
        # Check for recent keywords
        for keyword in self.RECENT_KEYWORDS:
            if keyword.lower() in claim_lower:
                return "recent"
        
        # Check year references
        current_year = datetime.now().year
        if years:
            max_year = max(years)
            if max_year >= current_year:
                return "recent"
            elif max_year <= 2023:
                return "historical"
        
        return "general"
    
    def _extract_keywords(self, claim: str) -> List[str]:
        """Extract important keywords from claim."""
        stop_words = {
            # English
            "the", "a", "an", "is", "are", "was", "were", "has", "have",
            "will", "be", "been", "being", "that", "this", "it", "and",
            "or", "but", "if", "then", "so", "because", "as", "of", "in",
            "on", "at", "to", "for", "with", "by", "from", "about",
            
            # Sinhala
            "සහ", "හා", "හෝ", "නිසා", "බැවින්", "විට", "වඩා", "ගැන", 
            "තවත්", "මෙම", "ඔබ", "මම", "අපි", "ඔව්", "නැත", "ඇත", "nati",
            "වෙත", "සඳහා", "මගින්", "විසින්", "ලෙස", "පිළිබඳ", "පිළිබඳව",
            "තුළ", "මත", "සිට", "දක්වා", "හේතුවෙන්", "කර", "කරන", "කරයි"
        }
        
        # Use whitespace splitting for better Unicode support (Sinhala)
        words = claim.split()
        
        # Clean words (remove punctuation)
        cleaned_words = []
        for w in words:
            # Remove punctuation from start/end
            clean_w = w.strip(".,!?\"':;()[]{}")
            if clean_w and clean_w.lower() not in stop_words and len(clean_w) > 2:
                cleaned_words.append(clean_w)
        
        return cleaned_words
    
    def _create_vector_query(self, claim: str, keywords: List[str]) -> str:
        """Create optimized query for vector search."""
        return claim
    
    def _create_web_query(self, claim: str, keywords: List[str]) -> str:
        """Create query for web search."""
        # Use the first 5-7 meaningful keywords for search
        # Avoid adding "verify" or english words to Sinhala queries
        query = " ".join(keywords[:7])
        return query
