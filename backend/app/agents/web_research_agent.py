"""
web_research_agent.py

The multi-agent orchestrator for deep web research.
Replaces the simple WebAnalyzer with an agent that plans, searches, browses, and analyzes.
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
import numpy as np
from datetime import datetime

from .browsing_tool import get_browsing_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchEvidence:
    """Evidence found during research."""
    source_url: str
    title: str
    content_snippet: str
    stance: str  # supports, refutes, neutral
    relevance_score: float
    published_date: Optional[str] = None

class WebResearchAgent:
    """
    Orchestrates the research process:
    1. Plan: Generate search queries
    2. Search: Find candidate URLs
    3. Browse: Read full content of promising pages
    4. Analyze: Determine stance and relevance using NLI
    5. Synthesize: Create a consolidated report
    """
    
    def __init__(self):
        print("[WebResearchAgent] Initializing...")
        self.browser = get_browsing_tool()
        
        # Load NLI model for analysis (lightweight)
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[WebResearchAgent] NLI Model loaded")
        except Exception as e:
            print(f"[WebResearchAgent] Error loading model: {e}")
            self.model = None
            
        self.max_browsing_steps = 3
        self.max_tokens_per_source = 1000
    
    def run(self, claim: str, keywords: List[str] = None) -> Dict:
        """
        Run the full research loop for a claim.
        """
        print(f"[WebResearchAgent] Starting research for: {claim[:50]}...")
        
        # 1. Plan
        queries = self._plan_search(claim, keywords)
        
        # 2. Search & Filter
        candidate_urls = self._search_web(queries)
        print(f"[WebResearchAgent] Found {len(candidate_urls)} candidate URLs")
        
        # 3. Browse & Analyze (Deep Reading)
        collected_evidence = []
        for url_info in candidate_urls[:self.max_browsing_steps]:
            evidence = self._process_url(url_info, claim)
            if evidence and evidence.relevance_score > 0.4:
                collected_evidence.append(evidence)
        
        # 4. Synthesize
        report = self._synthesize_report(claim, collected_evidence)
        
        return report

    def _plan_search(self, claim: str, keywords: List[str]) -> List[str]:
        """Generate search queries."""
        queries = [claim]
        if keywords:
            # Add English keywords query
            queries.append(" ".join(keywords))
            # Add specific news query
            queries.append(f"{' '.join(keywords[:5])} news sri lanka")
        return queries

    def _search_web(self, queries: List[str]) -> List[Dict]:
        """Execute searches and return unique URLs."""
        urls = []
        seen_links = set()
        
        with DDGS() as ddgs:
            for query in queries:
                print(f"[WebResearchAgent] Searching: {query}")
                try:
                    results = list(ddgs.text(query, region="wt-wt", safesearch="off", timelimit="y", max_results=5))
                    for r in results:
                        link = r.get("href")
                        if link and link not in seen_links:
                            seen_links.add(link)
                            urls.append({
                                "url": link,
                                "title": r.get("title", ""),
                                "snippet": r.get("body", "")
                            })
                except Exception as e:
                    print(f"[WebResearchAgent] Search error for '{query}': {e}")
        
        return urls

    def _process_url(self, url_info: Dict, claim: str) -> Optional[ResearchEvidence]:
        """Browse a URL and analyze its content against the claim."""
        url = url_info["url"]
        
        # Scrape full text
        page_data = self.browser.scrape_url(url)
        if page_data["status"] == "error":
            return None
            
        content = page_data["content"]
        if not content:
            return None
            
        # NLI Analysis
        stance, score, relevant_snippet = self._analyze_content(claim, content)
        
        return ResearchEvidence(
            source_url=url,
            title=page_data.get("title", url_info.get("title", "")),
            content_snippet=relevant_snippet,
            stance=stance,
            relevance_score=score
        )

    def _analyze_content(self, claim: str, content: str) -> (str, float, str):
        """
        Analyze content stance using semantic similarity.
        Returns: (stance, score, relevant_snippet)
        """
        if not self.model:
            return "neutral", 0.0, content[:200]
            
        # Split content into chunks/sentences
        sentences = [s.strip() for s in content.split('\n') if len(s.strip()) > 20]
        if not sentences:
            return "neutral", 0.0, content[:200]
            
        # Encode claim and sentences
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        sent_embs = self.model.encode(sentences, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(claim_emb, sent_embs)[0]
        
        # Find best matching sentence
        best_idx = np.argmax(cosine_scores.cpu().numpy())
        best_score = float(cosine_scores[best_idx])
        best_sentence = sentences[best_idx]
        
        # Determine Stance (Heuristic)
        # High similarity usually means support or direct addressal
        # To detect REFUTES, we need negation check.
        
        stance = "neutral"
        if best_score > 0.6:
            # Check for negation in the best sentence relative to claim
            if self._check_negation(claim, best_sentence):
                stance = "refutes"
            else:
                stance = "supports"
        elif best_score > 0.4:
            stance = "neutral"  # Related but not strong
        else:
            stance = "irrelevant"
            
        return stance, best_score, best_sentence

    def _check_negation(self, claim: str, evidence: str) -> bool:
        """Simple negation detection."""
        # This is a basic heuristic. A full NLI model would be better.
        negations = ["not", "no", "never", "false", "fake", "incorrect", "denied", "rejects"]
        
        # Count negations in evidence vs claim
        claim_negs = sum(1 for w in negations if w in claim.lower().split())
        ev_negs = sum(1 for w in negations if w in evidence.lower().split())
        
        # If mismatch in negation count, possible contradiction
        return claim_negs != ev_negs

    def _synthesize_report(self, claim: str, evidence_list: List[ResearchEvidence]) -> Dict:
        """Aggregates evidence into a final report."""
        
        support_count = sum(1 for e in evidence_list if e.stance == "supports")
        refute_count = sum(1 for e in evidence_list if e.stance == "refutes")
        
        # Calculate confidence boost
        confidence_boost = 0.0
        verdict_override = None
        
        if support_count >= 2 and refute_count == 0:
            confidence_boost = 0.2
            verdict_override = "likely_true"
        elif refute_count >= 2:
            confidence_boost = -0.2
            verdict_override = "likely_false"
        elif support_count > 0:
            confidence_boost = 0.1
            
        return {
            "evidence": [asdict(e) for e in evidence_list],
            "support_count": support_count,
            "refute_count": refute_count,
            "confidence_boost": confidence_boost,
            "verdict_override": verdict_override,
            "summary": f"Found {len(evidence_list)} relevant sources. {support_count} support, {refute_count} refute."
        }

# Singleton
_research_agent = None

def get_web_research_agent() -> WebResearchAgent:
    global _research_agent
    if _research_agent is None:
        _research_agent = WebResearchAgent()
    return _research_agent
