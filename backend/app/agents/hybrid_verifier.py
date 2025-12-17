"""
hybrid_verifier.py

Temporal Hybrid Verifier main orchestrator.
Coordinates the full verification pipeline with memory caching.
"""
from typing import Dict, Optional

from .claim_decomposer import ClaimDecomposer
from .hybrid_retriever import HybridRetriever
from .cross_examiner import CrossExaminer
from .cot_reasoner import CoTReasoner
from .verdict_agent import VerdictAgent
from .web_analyzer import get_web_analyzer
from ..store.memory_store import get_memory_manager


class HybridVerifier:
    """
    Main orchestrator for Temporal Hybrid Verification.
    Uses Redis for short term cache and PostgreSQL for long term memory.
    
    Flow:
    1. Check memory cache for existing result
    2. If not cached, run full verification pipeline
    3. Store result in both short and long term memory
    """
    
    def __init__(self):
        """Initialize all agents and memory."""
        print("[HybridVerifier] Initializing")
        
        # Initialize agents
        self.decomposer = ClaimDecomposer()
        self.retriever = HybridRetriever()
        self.examiner = CrossExaminer()
        self.reasoner = CoTReasoner()
        self.verdict_agent = VerdictAgent()
        self.web_analyzer = get_web_analyzer()
        
        # Initialize memory manager
        self.memory = get_memory_manager()
        
        print("[HybridVerifier] All agents and memory initialized")
    
    def verify(self, claim: str, use_cache: bool = True) -> Dict:
        """
        Verify a claim using the full hybrid pipeline.
        
        Args:
            claim: The news claim to verify
            use_cache: Whether to check cache first
        
        Returns:
            Complete verification result
        """
        print("[HybridVerifier] Starting verification")
        print("[HybridVerifier] Claim:", claim[:100])
        
        # Step 0: Check memory cache
        if use_cache:
            cached_result = self.memory.get_cached_result(claim)
            if cached_result:
                print("[HybridVerifier] Returning cached result")
                cached_result["from_cache"] = True
                return cached_result
        
        # Step 1: Decompose claim
        print("[HybridVerifier] Step 1 Decomposing claim")
        decomposed = self.decomposer.decompose(claim)
        
        # Step 2: Retrieve evidence
        print("[HybridVerifier] Step 2 Retrieving evidence")
        evidence = self.retriever.retrieve(claim, decomposed)
        
        # Step 3: Cross examine evidence
        print("[HybridVerifier] Step 3 Cross examining evidence")
        cross_exam = self.examiner.examine(evidence, decomposed)
        
        # Step 4: Analyze web evidence (NEW)
        print("[HybridVerifier] Step 4 Analyzing web evidence")
        web_analysis = self.web_analyzer.analyze(
            claim=claim,
            translated_claim=decomposed.get("translated_claim", claim),
            keywords=decomposed.get("english_keywords", []),
            web_results=evidence.get("web_results", [])
        )
        
        # Step 5: Get few shot examples
        few_shot_examples = self._get_few_shot_examples(evidence)
        
        # Step 6: Chain of Thought reasoning
        print("[HybridVerifier] Step 5 Chain of Thought reasoning")
        cot_result = self.reasoner.reason(
            claim, 
            evidence, 
            cross_exam,
            few_shot_examples
        )
        
        # Step 7: Generate final verdict (with web analysis)
        print("[HybridVerifier] Step 6 Generating verdict")
        result = self._generate_final_verdict(
            claim, 
            decomposed, 
            evidence, 
            cross_exam, 
            cot_result,
            web_analysis
        )
        
        # Step 7: Store in memory
        print("[HybridVerifier] Step 6 Storing in memory")
        self.memory.store_result(claim, result)
        
        result["from_cache"] = False
        
        print("[HybridVerifier] Verification complete")
        print("[HybridVerifier] Verdict:", result['verdict']['label'])
        
        return result
    
    def _get_few_shot_examples(self, evidence: Dict) -> list:
        """Extract few shot examples from labeled evidence."""
        examples = []
        labeled = evidence.get("labeled_history", [])
        
        for doc in labeled[:3]:
            if doc.get("score", 0) >= 0.70:
                examples.append({
                    "claim": doc.get("text", "")[:100],
                    "evidence": doc.get("text", ""),
                    "label": doc.get("label", "")
                })
        
        return examples
    
    def _generate_final_verdict(
        self,
        claim: str,
        decomposed: Dict,
        evidence: Dict,
        cross_exam: Dict,
        cot_result: Dict,
        web_analysis: Dict = None
    ) -> Dict:
        """Generate the final comprehensive verdict with web analysis."""
        
        # Use CoT result as primary verdict
        verdict_label = cot_result.get("verdict", "unverified")
        confidence = cot_result.get("confidence", 0.5)
        
        # APPLY WEB ANALYSIS (NEW)
        web_analysis = web_analysis or {}
        confidence_boost = web_analysis.get("confidence_boost", 0)
        verdict_override = web_analysis.get("verdict_override")
        
        # Apply confidence boost from web evidence
        if confidence_boost != 0:
            original_confidence = confidence
            confidence = max(0.1, min(0.95, confidence + confidence_boost))
            print(f"[HybridVerifier] Web boost: {original_confidence:.2f} + {confidence_boost:+.2f} = {confidence:.2f}")
        
        # Override verdict if web evidence is strong
        if verdict_override and web_analysis.get("support_count", 0) >= 2:
            print(f"[HybridVerifier] Verdict override: {verdict_label} -> {verdict_override}")
            verdict_label = verdict_override
            # Boost confidence for web-confirmed verdicts
            if web_analysis.get("has_wikipedia"):
                confidence = max(confidence, 0.85)
            else:
                confidence = max(confidence, 0.75)
        
        # Generate explanations
        verdict = self.verdict_agent.generate_verdict(
            claim={"claim_text": claim},
            reasoning={
                "verdict_recommendation": verdict_label,
                "match_analysis": {
                    "top_similarity": evidence.get("top_similarity", 0),
                    "match_level": evidence.get("similarity_level", "none"),
                    "web_count": evidence.get("web_count", 0)
                }
            },
            evidence=evidence.get("labeled_history", []) + evidence.get("unlabeled_context", [])
        )
        
        # Override with final results
        verdict["label"] = verdict_label
        verdict["confidence"] = confidence
        
        # Format web evidence for citations
        web_citations = []
        for we in web_analysis.get("evidence", []):
            source_prefix = "[Wiki] " if we.get("is_wikipedia") else "[Web] "
            stance_emoji = "✓" if we.get("stance") == "supports" else "✗" if we.get("stance") == "refutes" else "?"
            web_citations.append({
                "source": source_prefix + we.get("source", ""),
                "text": we.get("content", "")[:150],
                "url": we.get("url", ""),
                "stance": we.get("stance", "neutral"),
                "score": we.get("relevance", 0)
            })
        
        # Build complete result
        return {
            "claim": {
                "original": claim,
                "translated": decomposed.get("translated_claim", ""),
                "temporal_type": decomposed.get("temporal_type"),
                "keywords": decomposed.get("keywords", [])[:5]
            },
            "evidence": {
                "labeled_count": evidence.get("labeled_count", 0),
                "unlabeled_count": evidence.get("unlabeled_count", 0),
                "top_similarity": evidence.get("top_similarity", 0),
                "similarity_level": evidence.get("similarity_level", "none"),
                "citations": self._format_citations(evidence)
            },
            "web_analysis": {
                "stance": web_analysis.get("overall_stance", "neutral"),
                "support_count": web_analysis.get("support_count", 0),
                "refute_count": web_analysis.get("refute_count", 0),
                "has_wikipedia": web_analysis.get("has_wikipedia", False),
                "confidence_boost": confidence_boost,
                "citations": web_citations
            },
            "cross_examination": {
                "weighted_score": cross_exam.get("weighted_score", 0),
                "consensus": cross_exam.get("consensus", {}).get("type"),
                "zombie_detected": cross_exam.get("zombie_check", {}).get("is_zombie", False),
                "source_priority": cross_exam.get("source_priority")
            },
            "reasoning": {
                "cot_reasoning": cot_result.get("reasoning", ""),
                "steps": [
                    {
                        "step": "Claim Decomposition",
                        "result": "Type " + str(decomposed.get('temporal_type')) + " Keywords " + str(', '.join(decomposed.get('keywords', [])[:3]))
                    },
                    {
                        "step": "Evidence Retrieval",
                        "result": "Found " + str(evidence.get('labeled_count', 0)) + " labeled " + str(evidence.get('unlabeled_count', 0)) + " unlabeled"
                    },
                    {
                        "step": "Web Analysis",
                        "result": f"Stance: {web_analysis.get('overall_stance', 'neutral')}, Support: {web_analysis.get('support_count', 0)}, Refute: {web_analysis.get('refute_count', 0)}"
                    },
                    {
                        "step": "Cross Examination",
                        "result": cross_exam.get("consensus", {}).get("message", "Analyzed")
                    },
                    {
                        "step": "LLM Reasoning",
                        "result": cot_result.get("reasoning", "")[:100]
                    }
                ]
            },
            "verdict": verdict
        }
    
    def _format_citations(self, evidence: Dict) -> list:
        """Format evidence as citations."""
        citations = []
        
        for doc in evidence.get("labeled_history", [])[:3]:
            citations.append({
                "source": doc.get("source", "unknown"),
                "text": doc.get("text", "")[:100],
                "label": doc.get("label", ""),
                "similarity": str(round(doc.get('score', 0) * 100)) + " percent"
            })
        
        return citations
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return self.memory.get_stats()


# Create singleton instance
_verifier = None

def get_hybrid_verifier() -> HybridVerifier:
    """Get or create HybridVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = HybridVerifier()
    return _verifier
