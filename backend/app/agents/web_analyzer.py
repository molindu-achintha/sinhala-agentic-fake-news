"""
web_analyzer.py

Advanced Web Evidence Analyzer.
Analyzes web search results to determine claim support/opposition.
Uses Wikipedia API for factual claims verification.
"""
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Stance(Enum):
    """Evidence stance relative to claim."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    IRRELEVANT = "irrelevant"


@dataclass
class WebEvidence:
    """Structured web evidence."""
    source: str
    title: str
    content: str
    url: str
    stance: Stance
    relevance_score: float
    credibility_score: float
    is_wikipedia: bool = False


class WebAnalyzer:
    """
    Advanced web evidence analyzer.
    
    Features:
    - Wikipedia API integration for factual claims
    - Multi-source credibility weighting
    - Stance detection (supports/refutes/neutral)
    - Relevance scoring
    """
    
    # Source credibility scores
    SOURCE_CREDIBILITY = {
        "wikipedia": 0.95,
        "bbc": 0.90,
        "reuters": 0.90,
        "britannica": 0.90,
        "gov": 0.85,
        "edu": 0.85,
        "news": 0.75,
        "default": 0.60
    }
    
    # Keywords that indicate support
    SUPPORT_KEYWORDS = {
        "en": ["is", "are", "was", "confirmed", "true", "correct", "yes", "indeed", "officially"],
        "si": ["වේ", "ය", "බව", "තහවුරු", "සත්‍ය", "නිවැරදි"]
    }
    
    # Keywords that indicate refutation
    REFUTE_KEYWORDS = {
        "en": ["not", "false", "incorrect", "myth", "wrong", "no", "never", "fake", "untrue", "however"],
        "si": ["නැත", "නොවේ", "බොරු", "වැරදි", "අසත්‍ය"]
    }
    
    def __init__(self):
        """Initialize the web analyzer."""
        self.wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wikipedia_search = "https://en.wikipedia.org/w/api.php"
        print("[WebAnalyzer] Initialized")
    
    def analyze(
        self, 
        claim: str,
        translated_claim: str,
        keywords: List[str],
        web_results: List[Dict]
    ) -> Dict:
        """
        Analyze web evidence for a claim.
        
        Args:
            claim: Original Sinhala claim
            translated_claim: English translation
            keywords: Extracted keywords
            web_results: Raw web search results
            
        Returns:
            Analysis with stance, confidence boost, and evidence
        """
        print(f"[WebAnalyzer] Analyzing claim: {translated_claim[:50]}...")
        
        all_evidence: List[WebEvidence] = []
        
        # 1. Search Wikipedia for factual claims
        wiki_evidence = self._search_wikipedia(translated_claim, keywords)
        if wiki_evidence:
            all_evidence.extend(wiki_evidence)
            print(f"[WebAnalyzer] Found {len(wiki_evidence)} Wikipedia results")
        
        # 2. Analyze DuckDuckGo results
        ddg_evidence = self._analyze_web_results(claim, translated_claim, web_results)
        all_evidence.extend(ddg_evidence)
        print(f"[WebAnalyzer] Analyzed {len(ddg_evidence)} web results")
        
        # 3. Aggregate evidence
        analysis = self._aggregate_evidence(all_evidence, translated_claim)
        
        print(f"[WebAnalyzer] Overall stance: {analysis['overall_stance']}")
        print(f"[WebAnalyzer] Confidence boost: {analysis['confidence_boost']:+.2f}")
        
        return analysis
    
    def _search_wikipedia(
        self, 
        translated_claim: str, 
        keywords: List[str]
    ) -> List[WebEvidence]:
        """Search Wikipedia for relevant articles."""
        evidence = []
        
        # Extract main topic from claim for Wikipedia search
        search_terms = self._extract_wikipedia_terms(translated_claim)
        
        for term in search_terms[:2]:  # Limit to 2 searches
            try:
                # Search Wikipedia
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "format": "json",
                    "srlimit": 3
                }
                
                response = requests.get(
                    self.wikipedia_search, 
                    params=params,
                    timeout=10
                )
                
                if response.status_code != 200:
                    continue
                    
                data = response.json()
                search_results = data.get("query", {}).get("search", [])
                
                for result in search_results[:2]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    # Get full summary
                    summary = self._get_wiki_summary(title)
                    if not summary:
                        continue
                    
                    # Determine stance
                    stance, relevance = self._detect_stance(
                        translated_claim, 
                        summary,
                        is_english=True
                    )
                    
                    if stance != Stance.IRRELEVANT:
                        evidence.append(WebEvidence(
                            source="Wikipedia",
                            title=title,
                            content=summary[:500],
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            stance=stance,
                            relevance_score=relevance,
                            credibility_score=0.95,
                            is_wikipedia=True
                        ))
                        
            except Exception as e:
                print(f"[WebAnalyzer] Wikipedia search error: {e}")
                
        return evidence
    
    def _get_wiki_summary(self, title: str) -> Optional[str]:
        """Get Wikipedia article summary."""
        try:
            url = self.wikipedia_api + title.replace(" ", "_")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("extract", "")
        except:
            pass
        return None
    
    def _extract_wikipedia_terms(self, claim: str) -> List[str]:
        """Extract search terms for Wikipedia."""
        import re
        
        # Remove common words
        stop_words = {"the", "is", "are", "was", "of", "in", "a", "an", "to", "that", "this"}
        words = re.findall(r'\b\w{3,}\b', claim.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Create search terms
        terms = []
        
        # Full claim (shortened)
        if len(claim) < 100:
            terms.append(claim)
        
        # Key noun phrases (simplified)
        if len(keywords) >= 2:
            terms.append(" ".join(keywords[:4]))
        
        # Individual important keywords
        for kw in keywords[:3]:
            if len(kw) > 4:
                terms.append(kw)
        
        return terms[:3]
    
    def _analyze_web_results(
        self, 
        claim: str,
        translated_claim: str,
        web_results: List[Dict]
    ) -> List[WebEvidence]:
        """Analyze DuckDuckGo web search results."""
        evidence = []
        
        for result in web_results:
            title = result.get("title", "")
            body = result.get("body", "")
            url = result.get("href", "")
            
            if not body:
                continue
            
            # Determine source credibility
            credibility = self._get_source_credibility(url)
            
            # Detect stance
            stance, relevance = self._detect_stance(
                translated_claim,
                body,
                is_english=True
            )
            
            if stance != Stance.IRRELEVANT and relevance > 0.2:
                evidence.append(WebEvidence(
                    source=self._extract_source_name(url),
                    title=title,
                    content=body[:500],
                    url=url,
                    stance=stance,
                    relevance_score=relevance,
                    credibility_score=credibility,
                    is_wikipedia="wikipedia" in url.lower()
                ))
        
        return evidence
    
    def _detect_stance(
        self, 
        claim: str, 
        evidence: str,
        is_english: bool = True
    ) -> Tuple[Stance, float]:
        """
        Detect if evidence supports or refutes the claim.
        
        Uses keyword overlap + negation detection.
        """
        import re
        
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract key terms from claim
        claim_words = set(re.findall(r'\b\w{3,}\b', claim_lower))
        evidence_words = set(re.findall(r'\b\w{3,}\b', evidence_lower))
        
        # Calculate word overlap (relevance)
        if not claim_words:
            return Stance.IRRELEVANT, 0.0
            
        overlap = len(claim_words & evidence_words)
        relevance = overlap / len(claim_words)
        
        if relevance < 0.15:
            return Stance.IRRELEVANT, relevance
        
        # Check for support keywords
        lang = "en" if is_english else "si"
        support_count = sum(1 for kw in self.SUPPORT_KEYWORDS[lang] if kw in evidence_lower)
        refute_count = sum(1 for kw in self.REFUTE_KEYWORDS[lang] if kw in evidence_lower)
        
        # Check for negation near claim keywords
        has_negation = self._check_negation(claim_lower, evidence_lower)
        
        # Determine stance
        if has_negation or refute_count > support_count + 1:
            return Stance.REFUTES, relevance
        elif support_count > refute_count:
            return Stance.SUPPORTS, relevance
        elif relevance > 0.3:
            return Stance.NEUTRAL, relevance
        else:
            return Stance.IRRELEVANT, relevance
    
    def _check_negation(self, claim: str, evidence: str) -> bool:
        """Check if evidence negates claim keywords."""
        import re
        
        # Extract main subject/object from claim
        claim_words = re.findall(r'\b\w{4,}\b', claim)
        
        negation_patterns = [
            r"not\s+\w*\s*" + word for word in claim_words[:3]
        ]
        negation_patterns.extend([
            r"is\s+not", r"are\s+not", r"was\s+not"
        ])
        negation_patterns.extend([
            r"never\s+\w*\s*" + word for word in claim_words[:2]
        ])
        
        for pattern in negation_patterns:
            if re.search(pattern, evidence, re.IGNORECASE):
                return True
        
        return False
    
    def _get_source_credibility(self, url: str) -> float:
        """Get credibility score for a source URL."""
        url_lower = url.lower()
        
        if "wikipedia" in url_lower:
            return self.SOURCE_CREDIBILITY["wikipedia"]
        elif "bbc" in url_lower:
            return self.SOURCE_CREDIBILITY["bbc"]
        elif "reuters" in url_lower:
            return self.SOURCE_CREDIBILITY["reuters"]
        elif "britannica" in url_lower:
            return self.SOURCE_CREDIBILITY["britannica"]
        elif ".gov" in url_lower:
            return self.SOURCE_CREDIBILITY["gov"]
        elif ".edu" in url_lower:
            return self.SOURCE_CREDIBILITY["edu"]
        elif "news" in url_lower or "times" in url_lower:
            return self.SOURCE_CREDIBILITY["news"]
        else:
            return self.SOURCE_CREDIBILITY["default"]
    
    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL."""
        import re
        
        # Extract domain
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            domain = match.group(1)
            # Clean up
            parts = domain.split('.')
            if len(parts) >= 2:
                return parts[-2].capitalize()
        return "Web Source"
    
    def _aggregate_evidence(
        self, 
        evidence: List[WebEvidence],
        claim: str
    ) -> Dict:
        """Aggregate all evidence into final analysis."""
        
        if not evidence:
            return {
                "overall_stance": "neutral",
                "confidence_boost": 0.0,
                "support_count": 0,
                "refute_count": 0,
                "evidence": [],
                "has_wikipedia": False,
                "verdict_override": None
            }
        
        # Count stances
        support_count = sum(1 for e in evidence if e.stance == Stance.SUPPORTS)
        refute_count = sum(1 for e in evidence if e.stance == Stance.REFUTES)
        neutral_count = sum(1 for e in evidence if e.stance == Stance.NEUTRAL)
        
        # Check for Wikipedia evidence
        wiki_supports = any(e.is_wikipedia and e.stance == Stance.SUPPORTS for e in evidence)
        wiki_refutes = any(e.is_wikipedia and e.stance == Stance.REFUTES for e in evidence)
        
        # Calculate weighted support score
        weighted_support = sum(
            e.credibility_score * e.relevance_score * (1 if e.stance == Stance.SUPPORTS else -1 if e.stance == Stance.REFUTES else 0)
            for e in evidence
        )
        
        # Determine overall stance
        if support_count > refute_count + 1:
            overall_stance = "supports"
        elif refute_count > support_count + 1:
            overall_stance = "refutes"
        else:
            overall_stance = "neutral"
        
        # Calculate confidence boost
        confidence_boost = 0.0
        verdict_override = None
        
        # Strong support
        if support_count >= 3 or wiki_supports:
            confidence_boost = min(0.35, 0.15 + support_count * 0.05)
            if wiki_supports:
                confidence_boost += 0.10
                if support_count >= 2:
                    verdict_override = "likely_true"
        
        # Strong refutation
        elif refute_count >= 3 or wiki_refutes:
            confidence_boost = max(-0.35, -0.15 - refute_count * 0.05)
            if wiki_refutes:
                confidence_boost -= 0.10
                if refute_count >= 2:
                    verdict_override = "likely_false"
        
        # Moderate evidence
        elif support_count > refute_count:
            confidence_boost = 0.10 + support_count * 0.03
        elif refute_count > support_count:
            confidence_boost = -0.10 - refute_count * 0.03
        
        # Format evidence for output
        formatted_evidence = []
        for e in evidence:
            formatted_evidence.append({
                "source": e.source,
                "title": e.title,
                "content": e.content[:200],
                "url": e.url,
                "stance": e.stance.value,
                "relevance": round(e.relevance_score, 2),
                "credibility": round(e.credibility_score, 2),
                "is_wikipedia": e.is_wikipedia
            })
        
        return {
            "overall_stance": overall_stance,
            "confidence_boost": round(confidence_boost, 2),
            "support_count": support_count,
            "refute_count": refute_count,
            "neutral_count": neutral_count,
            "has_wikipedia": wiki_supports or wiki_refutes,
            "verdict_override": verdict_override,
            "evidence": formatted_evidence,
            "weighted_score": round(weighted_support, 2)
        }


# Singleton instance
_analyzer = None

def get_web_analyzer() -> WebAnalyzer:
    """Get or create web analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = WebAnalyzer()
    return _analyzer
