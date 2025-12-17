"""
cross_examiner.py

Evidence Cross Examination Agent.
Analyzes and weights evidence from multiple sources.
"""
from typing import Dict, List
from datetime import datetime
from collections import Counter


class CrossExaminer:
    """
    Cross examines evidence to determine reliability and consensus.
    """
    
    # Source reliability weights
    SOURCE_WEIGHTS = {
        "BBC Sinhala": 1.0,
        "Hiru News": 0.9,
        "Ada Derana": 0.9,
        "Lankadeepa": 0.9,
        "ITN News": 0.8,
        "Twitter": 0.5,
        "Facebook": 0.4,
        "unknown": 0.3
    }
    
    # Label weights for scoring
    LABEL_SCORES = {
        "true": 1.0,
        "real": 1.0,
        "verified": 1.0,
        "false": -1.0,
        "fake": -1.0,
        "misleading": -0.5,
        "unknown": 0.0
    }
    
    def __init__(self):
        """Initialize cross examiner."""
        print("[CrossExaminer] Initialized")
    
    def examine(
        self, 
        evidence: Dict, 
        decomposed: Dict
    ) -> Dict:
        """
        Cross examine all evidence and determine verdict recommendation.
        
        Args:
            evidence: Output from HybridRetriever
            decomposed: Output from ClaimDecomposer
        
        Returns:
            Dict with weighted evidence consensus and recommendation
        """
        print("[CrossExaminer] Starting cross examination")
        
        labeled = evidence.get("labeled_history", [])
        unlabeled = evidence.get("unlabeled_context", [])
        web_results = evidence.get("web_results", [])
        
        # Step 1: Date consistency check
        date_analysis = self._check_date_consistency(decomposed, labeled)
        
        # Step 2: Analyze labeled evidence
        label_analysis = self._analyze_labels(labeled)
        
        # Step 3: Check for zombie rumors
        zombie_check = self._check_zombie_rumors(labeled)
        
        # Step 4: Determine consensus
        consensus = self._determine_consensus(label_analysis, web_results)
        
        # Step 5: Calculate weighted score
        weighted_score = self._calculate_weighted_score(labeled, unlabeled)
        
        # Step 6: Determine primary source priority
        source_priority = self._get_source_priority(decomposed, evidence)
        
        # Step 7: Generate recommendation
        recommendation = self._generate_recommendation(
            label_analysis, 
            consensus, 
            zombie_check,
            evidence.get("similarity_level", "none")
        )
        
        result = {
            "date_analysis": date_analysis,
            "label_analysis": label_analysis,
            "zombie_check": zombie_check,
            "consensus": consensus,
            "weighted_score": weighted_score,
            "source_priority": source_priority,
            "recommendation": recommendation,
            "confidence": self._calculate_confidence(label_analysis, evidence)
        }
        
        print("[CrossExaminer] Recommendation:", recommendation)
        print("[CrossExaminer] Weighted score:", round(weighted_score, 2))
        
        return result
    
    def _check_date_consistency(self, decomposed: Dict, labeled: List[Dict]) -> Dict:
        """Check if claim date matches evidence dates."""
        claim_years = decomposed.get("years", [])
        temporal_type = decomposed.get("temporal_type", "general")
        
        if temporal_type == "historical" and claim_years:
            return {
                "type": "historical",
                "claim_years": claim_years,
                "trust_db": True,
                "message": "Historical claim Vector DB is authoritative"
            }
        elif temporal_type == "recent":
            return {
                "type": "recent",
                "trust_db": False,
                "message": "Recent claim Web search recommended"
            }
        else:
            return {
                "type": "general",
                "trust_db": True,
                "message": "General claim Using all sources"
            }
    
    def _analyze_labels(self, labeled: List[Dict]) -> Dict:
        """Analyze labels from evidence."""
        if not labeled:
            return {
                "has_labels": False,
                "dominant_label": None,
                "label_counts": {},
                "support_score": 0
            }
        
        label_counts = Counter()
        scores = []
        
        for doc in labeled:
            label = doc.get("label", "").lower()
            similarity = doc.get("score", 0.5)
            source = doc.get("source", "unknown")
            
            if label in self.LABEL_SCORES:
                label_counts[label] += 1
                weight = self.SOURCE_WEIGHTS.get(source, 0.3)
                score = self.LABEL_SCORES[label] * similarity * weight
                scores.append(score)
        
        support_score = sum(scores) / len(scores) if scores else 0
        dominant = label_counts.most_common(1)[0][0] if label_counts else None
        
        return {
            "has_labels": True,
            "dominant_label": dominant,
            "label_counts": dict(label_counts),
            "support_score": support_score,
            "true_count": label_counts.get("true", 0) + label_counts.get("real", 0),
            "false_count": label_counts.get("false", 0) + label_counts.get("fake", 0)
        }
    
    def _check_zombie_rumors(self, labeled: List[Dict]) -> Dict:
        """Check if this is a known recurring fake news zombie rumor."""
        for doc in labeled:
            label = doc.get("label", "").lower()
            similarity = doc.get("score", 0)
            
            # High similarity match with known FALSE label
            if label in ["false", "fake"] and similarity >= 0.90:
                return {
                    "is_zombie": True,
                    "matched_claim": doc.get("text", "")[:100],
                    "original_source": doc.get("source", ""),
                    "message": "This appears to be a known recurring false claim"
                }
        
        return {"is_zombie": False}
    
    def _determine_consensus(self, label_analysis: Dict, web_results: List) -> Dict:
        """Determine if sources agree."""
        true_count = label_analysis.get("true_count", 0)
        false_count = label_analysis.get("false_count", 0)
        
        if true_count > 0 and false_count > 0:
            return {
                "has_consensus": False,
                "type": "conflict",
                "message": "Sources disagree needs verification"
            }
        elif true_count > 0:
            return {
                "has_consensus": True,
                "type": "agree_true",
                "message": "Sources agree claim is TRUE"
            }
        elif false_count > 0:
            return {
                "has_consensus": True,
                "type": "agree_false",
                "message": "Sources agree claim is FALSE"
            }
        else:
            return {
                "has_consensus": False,
                "type": "no_labels",
                "message": "No labeled evidence found"
            }
    
    def _calculate_weighted_score(
        self, 
        labeled: List[Dict], 
        unlabeled: List[Dict]
    ) -> float:
        """Calculate weighted evidence score."""
        total_score = 0
        total_weight = 0
        
        # Labeled evidence high weight
        for doc in labeled:
            label = doc.get("label", "").lower()
            similarity = doc.get("score", 0.5)
            source = doc.get("source", "unknown")
            
            label_score = self.LABEL_SCORES.get(label, 0)
            source_weight = self.SOURCE_WEIGHTS.get(source, 0.3)
            
            weight = similarity * source_weight
            total_score += label_score * weight
            total_weight += weight
        
        # Unlabeled contributes less
        for doc in unlabeled:
            similarity = doc.get("score", 0.5)
            weight = similarity * 0.2
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return total_score / total_weight
    
    def _get_source_priority(self, decomposed: Dict, evidence: Dict) -> str:
        """Determine which source to prioritize."""
        temporal_type = decomposed.get("temporal_type", "general")
        labeled_count = evidence.get("labeled_count", 0)
        
        if temporal_type == "historical" and labeled_count > 0:
            return "labeled_db"
        elif temporal_type == "recent":
            return "web"
        elif labeled_count > 0:
            return "labeled_db"
        else:
            return "unlabeled_db"
    
    def _generate_recommendation(
        self, 
        label_analysis: Dict, 
        consensus: Dict,
        zombie_check: Dict,
        similarity_level: str
    ) -> str:
        """Generate verdict recommendation based on evidence."""
        # Check for zombie rumor first (known false claim)
        if zombie_check.get("is_zombie"):
            return "false"
        
        # No labels found - cannot verify
        if not label_analysis.get("has_labels"):
            if similarity_level in ["high", "medium"]:
                return "needs_verification"
            return "unverified"
        
        support_score = label_analysis.get("support_score", 0)
        true_count = label_analysis.get("true_count", 0)
        false_count = label_analysis.get("false_count", 0)
        
        # Check for conflicting evidence
        if consensus.get("type") == "conflict":
            return "needs_verification"
        
        # HIGH similarity required for definitive verdicts
        if similarity_level == "high":
            # Strong evidence for TRUE (score >= 0.7)
            if support_score >= 0.7 and true_count >= 2:
                return "true"
            # Strong evidence for FALSE (score <= -0.7)
            elif support_score <= -0.7 and false_count >= 2:
                return "false"
            # Moderate evidence
            elif support_score >= 0.4:
                return "likely_true"
            elif support_score <= -0.4:
                return "likely_false"
            else:
                return "needs_verification"
        
        # MEDIUM similarity - be more cautious
        elif similarity_level == "medium":
            if support_score >= 0.6:
                return "likely_true"
            elif support_score <= -0.6:
                return "likely_false"
            else:
                return "needs_verification"
        
        # LOW or NO similarity - cannot verify
        return "unverified"
    
    def _calculate_confidence(self, label_analysis: Dict, evidence: Dict) -> float:
        """Calculate confidence in recommendation."""
        base_confidence = 0.5
        
        # Boost from labeled evidence
        if label_analysis.get("has_labels"):
            base_confidence += 0.2
        
        # Boost from high similarity
        top_sim = evidence.get("top_similarity", 0)
        base_confidence += top_sim * 0.2
        
        # Boost from consensus
        labeled_count = evidence.get("labeled_count", 0)
        if labeled_count >= 3:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
