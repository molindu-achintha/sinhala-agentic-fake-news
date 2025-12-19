"""
verdict_agent.py - Verdict Orchestrator

This agent orchestrates the two-stage fact-checking pipeline:
1. Research Agent: Gathers evidence from the web
2. Judge Agent: Produces verdict with Sinhala explanation

It produces:
- Final verdict label (true, false, misleading, needs_verification)
- Confidence score (0-1)
- Explanation in Sinhala with inline citations
"""
from typing import Dict, List
import os

from .research_agent import get_research_agent
from .judge_agent import get_judge_agent


class VerdictAgent:
    """
    Main orchestrator for the two-stage agentic fact-checking pipeline.
    
    Stage 1: Research Agent (with tools) → Evidence JSON
    Stage 2: Judge Agent (no tools) → Sinhala Verdict + Explanation
    """
    
    # Sinhala explanations for fallback
    EXPLANATIONS_SI = {
        "true": "මෙම පුවත සත්‍ය බව තහවුරු විය.",
        "likely_true": "මෙම පුවත බොහෝ දුරට සත්‍ය විය හැක.",
        "needs_verification": "මෙම පුවත තවදුරටත් සත්‍යාපනය අවශ්‍ය වේ.",
        "likely_false": "මෙම පුවත බොහෝ දුරට අසත්‍ය විය හැක.",
        "false": "මෙම පුවත ව්‍යාජ බව තහවුරු විය.",
        "misleading": "මෙම පුවත නොමඟ යවන සුළුය."
    }
    
    def __init__(self):
        """Initialize the verdict orchestrator."""
        self.research_agent = get_research_agent()
        self.judge_agent = get_judge_agent()
        print("[VerdictAgent] Initialized with two-stage pipeline")
    
    def generate_verdict(
        self, 
        claim: dict, 
        reasoning: dict = None, 
        evidence: list = None,
        web_analysis: dict = None,
        llm_provider: str = "deepresearch",
        api_key: str = None
    ) -> dict:
        """
        Generate final verdict using the two-stage agentic pipeline.
        
        Args:
            claim: Claim information (original_claim, translated_claim, etc.)
            reasoning: Output from CoT reasoner (optional, for compatibility)
            evidence: List of evidence from vector DB (optional)
            web_analysis: Previous web analysis (deprecated, ignored)
            llm_provider: LLM to use (deepresearch recommended)
        
        Returns:
            Dictionary with verdict label, confidence, and Sinhala explanation
        """
        original_claim = claim.get("original_claim", "")
        print(f"[VerdictAgent] Starting two-stage pipeline for: {original_claim[:60]}...")
        
        # =========================================
        # STAGE 1: Research Agent (gather evidence)
        # =========================================
        print("[VerdictAgent] Stage 1: Calling Research Agent...")
        evidence_json = self.research_agent.research(original_claim, api_key=api_key)
        
        if not evidence_json:
            print("[VerdictAgent] Research failed, using fallback")
            return self._create_fallback_verdict(claim)
        
        print(f"[VerdictAgent] Research complete: {len(evidence_json.get('evidence', []))} evidence items")
        
        # =========================================
        # STAGE 2: Judge Agent (produce verdict)
        # =========================================
        print("[VerdictAgent] Stage 2: Calling Judge Agent...")
        verdict_result = self.judge_agent.judge(evidence_json, api_key=api_key)
        
        if not verdict_result:
            print("[VerdictAgent] Judgment failed, using fallback")
            return self._create_fallback_verdict(claim)
        
        print(f"[VerdictAgent] Verdict: {verdict_result.get('label')} ({verdict_result.get('confidence')})")
        
        # Add the evidence JSON for reference
        verdict_result["research_evidence"] = evidence_json
        
        return verdict_result
    
    def generate_verdict_simple(self, claim_text: str) -> dict:
        """
        Simplified interface: just pass the claim text.
        
        Args:
            claim_text: Raw claim text (any language)
            
        Returns:
            Full verdict result
        """
        claim = {"original_claim": claim_text}
        return self.generate_verdict(claim)
    
    def _create_fallback_verdict(self, claim: dict) -> dict:
        """Create a fallback verdict when the pipeline fails."""
        return {
            "label": "needs_verification",
            "confidence": 0.3,
            "explanation_si": self.EXPLANATIONS_SI["needs_verification"],
            "explanation_en": "This claim requires further verification.",
            "detailed_explanation": "Unable to complete fact-check. Please try again.",
            "citations": [],
            "evidence_count": 0,
            "supports_count": 0,
            "refutes_count": 0,
            "llm_powered": False,
            "claim_normalized_si": claim.get("original_claim", ""),
            "claim_normalized_en": claim.get("translated_claim", "")
        }


# Singleton instance
_verdict_agent = None

def get_verdict_agent() -> VerdictAgent:
    """Get or create the Verdict Agent singleton."""
    global _verdict_agent
    if _verdict_agent is None:
        _verdict_agent = VerdictAgent()
    return _verdict_agent
