"""
hybrid_verifier.py

Temporal Hybrid Verifier main orchestrator.
Coordinates the full verification pipeline with memory caching.

Simplified Pipeline (v4 - Two-Stage Agentic):
1. Check cache
2. Decompose claim
3. (Optional) Vector DB retrieval
4. Two-stage agentic verification (Research Agent → Judge Agent)
5. Store in memory
"""
from typing import Dict, Optional

from .claim_decomposer import ClaimDecomposer
from .hybrid_retriever import HybridRetriever
from .cross_examiner import CrossExaminer
from .verdict_agent import VerdictAgent
from .wikidata_client import get_wikidata_client
from ..store.memory_store import get_memory_manager


class HybridVerifier:
    """
    Main orchestrator for Temporal Hybrid Verification.
    Uses Redis for short term cache and PostgreSQL for long term memory.
    
    Simplified Flow (v4):
    1. Check memory cache for existing result
    2. Decompose claim (normalize, translate)
    3. Optional: Vector DB retrieval for context
    4. Two-stage agentic verification:
       - Stage 1: Research Agent (web search + evidence collection)
       - Stage 2: Judge Agent (verdict + Sinhala explanation)
    5. Store result in both short and long term memory
    """
    
    def __init__(self):
        """Initialize all agents and memory."""
        print("[HybridVerifier] Initializing")
        
        # Core agents
        self.decomposer = ClaimDecomposer()
        self.retriever = HybridRetriever()
        self.examiner = CrossExaminer()
        self.verdict_agent = VerdictAgent()
        self.wikidata = get_wikidata_client()
        
        # Initialize memory manager
        self.memory = get_memory_manager()
        
        print("[HybridVerifier] All agents and memory initialized")
    
    def verify(
        self, 
        claim: str, 
        use_cache: bool = True, 
        llm_provider: str = "deepresearch",
        use_vector_db: bool = True,
        openrouter_api_key: Optional[str] = None
    ) -> Dict:
        """
        Verify a claim using the two-stage agentic pipeline.
        
        Args:
            claim: The claim text to verify
            use_cache: Whether to check memory cache
            llm_provider: LLM to use ('deepresearch' recommended)
            use_vector_db: Whether to search vector database for context
            
        Returns:
            Dict containing verdict, confidence, and Sinhala explanation
        """
        print("[HybridVerifier] Starting verification")
        print(f"[HybridVerifier] Claim: {claim[:100]}")
        
        # Step 0: Check memory cache
        if use_cache:
            cached_result = self.memory.get_cached_result(claim)
            if cached_result:
                print("[HybridVerifier] Returning cached result")
                cached_result["from_cache"] = True
                return cached_result
        
        # Step 1: Decompose claim (normalize, translate, extract keywords)
        print("[HybridVerifier] Step 1: Decomposing claim")
        decomposed = self.decomposer.decompose(claim)
        
        # Step 2: (Optional) Wikidata verification for factual claims
        print("[HybridVerifier] Step 2: Checking Wikidata")
        wikidata_result = self.wikidata.verify_claim(
            claim=claim,
            translated_claim=decomposed.get("translated_claim", claim)
        )
        
        # Step 3: (Optional) Vector DB retrieval for context
        evidence = {"labeled_history": [], "unlabeled_context": []}
        cross_exam = {}
        
        if use_vector_db:
            print("[HybridVerifier] Step 3: Retrieving from Vector DB")
            evidence = self.retriever.retrieve(claim, decomposed)
            cross_exam = self.examiner.examine(evidence, decomposed)
        else:
            print("[HybridVerifier] Step 3: Vector DB disabled, skipping")
        
        # Step 4: Two-stage agentic verification (main pipeline)
        print("[HybridVerifier] Step 4: Running two-stage agentic verification")
        verdict_result = self.verdict_agent.generate_verdict(
            claim=decomposed,
            reasoning=None,  
            evidence=evidence.get("labeled_history", []),
            llm_provider=llm_provider,
            api_key=openrouter_api_key
        )
        
        # Build full result
        result = {
            "claim": {
                "original": claim,
                "translated": decomposed.get("translated_claim", ""),
                "normalized_si": verdict_result.get("claim_normalized_si", ""),
                "normalized_en": verdict_result.get("claim_normalized_en", ""),
                "temporal_type": decomposed.get("temporal_type", "general"),
                "keywords": decomposed.get("keywords", [])
            },
            "evidence": {
                "labeled_history": evidence.get("labeled_history", []),
                "unlabeled_context": evidence.get("unlabeled_context", []),
                "web_count": len(verdict_result.get("research_evidence", {}).get("evidence", []))
            },
            "cross_examination": cross_exam,
            "reasoning": {
                "wikidata": wikidata_result,
                "evidence_count": verdict_result.get("evidence_count", 0),
                "supports_count": verdict_result.get("supports_count", 0),
                "refutes_count": verdict_result.get("refutes_count", 0)
            },
            "verdict": {
                "label": verdict_result.get("label", "needs_verification"),
                "confidence": verdict_result.get("confidence", 0.5),
                "explanation_si": verdict_result.get("explanation_si", ""),
                "explanation_en": verdict_result.get("explanation_en", ""),
                "detailed_explanation": verdict_result.get("detailed_explanation", ""),
                "citations": verdict_result.get("citations", []),
                "llm_powered": verdict_result.get("llm_powered", False)
            },
            "research_evidence": verdict_result.get("research_evidence", {})
        }
        
        # Step 5: Store in memory
        print("[HybridVerifier] Step 5: Storing in memory")
        self.memory.store_result(claim, result)
        
        result["from_cache"] = False
        
        print("[HybridVerifier] Verification complete")
        print(f"[HybridVerifier] Verdict: {result['verdict']['label']}")
        
        return result


# Singleton instance
_verifier = None

def get_hybrid_verifier() -> HybridVerifier:
    """Get or create the hybrid verifier singleton."""
    global _verifier
    if _verifier is None:
        _verifier = HybridVerifier()
    return _verifier
