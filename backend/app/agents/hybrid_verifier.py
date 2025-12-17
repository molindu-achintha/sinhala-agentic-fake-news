"""
hybrid_verifier.py

Temporal Hybrid Verifier main orchestrator.
Coordinates the full verification pipeline.
"""
from typing import Dict, Optional

from .claim_decomposer import ClaimDecomposer
from .hybrid_retriever import HybridRetriever
from .cross_examiner import CrossExaminer
from .cot_reasoner import CoTReasoner
from .verdict_agent import VerdictAgent


class HybridVerifier:
    """
    Main orchestrator for Temporal Hybrid Verification.
    
    Flow:
    1. Decompose claim
    2. Retrieve evidence from Vector DB
    3. Cross examine evidence
    4. Chain of Thought reasoning with LLM
    5. Generate final verdict
    """
    
    def __init__(self):
        """Initialize all agents."""
        print("[HybridVerifier] Initializing")
        
        self.decomposer = ClaimDecomposer()
        self.retriever = HybridRetriever()
        self.examiner = CrossExaminer()
        self.reasoner = CoTReasoner()
        self.verdict_agent = VerdictAgent()
        
        print("[HybridVerifier] All agents initialized")
    
    def verify(self, claim: str) -> Dict:
        """
        Verify a claim using the full hybrid pipeline.
        
        Args:
            claim: The news claim to verify
        
        Returns:
            Complete verification result with verdict evidence reasoning
        """
        print("[HybridVerifier] Starting verification")
        print("[HybridVerifier] Claim:", claim[:100])
        
        # Step 1: Decompose claim
        print("[HybridVerifier] Step 1 Decomposing claim")
        decomposed = self.decomposer.decompose(claim)
        
        # Step 2: Retrieve evidence
        print("[HybridVerifier] Step 2 Retrieving evidence")
        evidence = self.retriever.retrieve(claim, decomposed)
        
        # Step 3: Cross examine evidence
        print("[HybridVerifier] Step 3 Cross examining evidence")
        cross_exam = self.examiner.examine(evidence, decomposed)
        
        # Step 4: Get few shot examples from high similarity labeled matches
        few_shot_examples = self._get_few_shot_examples(evidence)
        
        # Step 5: Chain of Thought reasoning with LLM
        print("[HybridVerifier] Step 4 Chain of Thought reasoning")
        cot_result = self.reasoner.reason(
            claim, 
            evidence, 
            cross_exam,
            few_shot_examples
        )
        
        # Step 6: Generate final verdict
        print("[HybridVerifier] Step 5 Generating verdict")
        verdict = self._generate_final_verdict(
            claim, 
            decomposed, 
            evidence, 
            cross_exam, 
            cot_result
        )
        
        print("[HybridVerifier] Verification complete")
        print("[HybridVerifier] Verdict:", verdict['verdict']['label'])
        print("[HybridVerifier] Confidence:", verdict['verdict']['confidence'])
        
        return verdict
    
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
        cot_result: Dict
    ) -> Dict:
        """Generate the final comprehensive verdict."""
        
        # Use CoT result as primary verdict
        verdict_label = cot_result.get("verdict", "unverified")
        confidence = cot_result.get("confidence", 0.5)
        
        # Generate explanations
        verdict = self.verdict_agent.generate_verdict(
            claim={"claim_text": claim},
            reasoning={
                "verdict_recommendation": verdict_label,
                "match_analysis": {
                    "top_similarity": evidence.get("top_similarity", 0),
                    "match_level": evidence.get("similarity_level", "none")
                }
            },
            evidence=evidence.get("labeled_history", []) + evidence.get("unlabeled_context", [])
        )
        
        # Override with CoT results
        verdict["label"] = verdict_label
        verdict["confidence"] = confidence
        
        # Build complete result
        return {
            "claim": {
                "original": claim,
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
                        "step": "Cross Examination",
                        "result": cross_exam.get("consensus", {}).get("message", "")
                    },
                    {
                        "step": "LLM Reasoning",
                        "result": cot_result.get("reasoning", "No reasoning provided")
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


# Create singleton instance
_verifier = None

def get_hybrid_verifier() -> HybridVerifier:
    """Get or create HybridVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = HybridVerifier()
    return _verifier
