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
        "Web Source": 0.8, # Trusted web search
        "Twitter": 0.5,
        "Facebook": 0.4,
        "unknown": 0.3
    }
    
    # ... (LABEL_SCORES remain same) ...

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
            
        # Unlabeled/Web evidence (contributes to confidence, not direction)
        for doc in unlabeled:
            similarity = doc.get("score", 0)
            if similarity > 0.75:
                # Good web match increases confidence that we have info
                total_weight += 0.2
        
        if total_weight == 0:
            return 0
            
        return total_score / total_weight
    
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
        zombie_check = self._check_zombie_rumors(labeled, decomposed)
        
        # Step 4: Determine consensus
        consensus = self._determine_consensus(label_analysis, web_results)
        
        # Step 5: Calculate weighted score
        weighted_score = self._calculate_weighted_score(labeled, unlabeled)
        
        # Step 6: Determine primary source priority
        source_priority = self._get_source_priority(decomposed, evidence)
        
        # Step 7: Check topic relevance (Keyword Overlap)
        relevance = self._check_topic_relevance(decomposed, labeled)
        
        # Adjust weighted score if topic mismatch
        if not relevance["is_relevant"]:
            print(f"[CrossExaminer] Topic mismatch detected (Overlap: {relevance['overlap_ratio']:.2f})")
            weighted_score = 0
            label_analysis["support_score"] = 0
            label_analysis["has_labels"] = False  # Ignore labels from irrelevant docs
            
        # Step 8: Generate recommendation
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
            "topic_relevance": relevance,
            "recommendation": recommendation,
            "confidence": self._calculate_confidence(label_analysis, evidence)
        }
        
        print("[CrossExaminer] Recommendation:", recommendation)
        print("[CrossExaminer] Weighted score:", round(weighted_score, 2))
        
        return result

    def _check_topic_relevance(self, decomposed: Dict, labeled: List[Dict]) -> Dict:
        """
        Check if evidence actually talks about the same topic.
        Uses keyword overlap AND translated claim matching.
        """
        # Get both Sinhala and English keywords
        sinhala_keywords = set(k.lower() for k in decomposed.get("keywords", []))
        english_keywords = set(k.lower() for k in decomposed.get("english_keywords", []))
        translated_claim = decomposed.get("translated_claim", "").lower()
        
        # Key entities from translated claim (e.g., "capital", "Colombo", "Sri Lanka")
        key_entities = set()
        if translated_claim:
            # Extract important words from translation
            import re
            words = re.findall(r'\b\w{3,}\b', translated_claim)
            common = {"the", "is", "are", "was", "of", "in", "and", "that", "this", "has", "have", "been"}
            key_entities = set(w.lower() for w in words if w.lower() not in common)
        
        all_keywords = sinhala_keywords | english_keywords | key_entities
        
        if not all_keywords or not labeled:
            return {"is_relevant": True, "overlap_ratio": 1.0, "method": "default"}
            
        max_overlap = 0.0
        best_match_text = ""
        
        for doc in labeled:
            text = doc.get("text", "").lower()
            # Count how many of our keywords appear in this evidence
            found = sum(1 for k in all_keywords if k in text)
            ratio = found / len(all_keywords) if all_keywords else 0
            
            if ratio > max_overlap:
                max_overlap = ratio
                best_match_text = text[:100]
            
        # STRICTER Threshold: Need at least 25% keyword overlap
        # Or at least 2 matching keywords for short claims
        min_matches = min(2, len(all_keywords))
        found_count = int(max_overlap * len(all_keywords))
        is_relevant = max_overlap >= 0.25 or found_count >= min_matches
        
        print(f"[CrossExaminer] Topic check: {found_count}/{len(all_keywords)} keywords matched")
        
        return {
            "is_relevant": is_relevant,
            "overlap_ratio": max_overlap,
            "keyword_count": len(all_keywords),
            "matched_count": found_count,
            "method": "keyword_overlap"
        }
    
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
    
    def _check_zombie_rumors(self, labeled: List[Dict], decomposed: Dict) -> Dict:
        """
        Check for zombie rumors (recycled old news).
        Condition 1: Known recurring false claim (high similarity match)
        Condition 2: Old TRUE news being shared as RECENT news
        """
        current_year = datetime.now().year
        temporal_type = decomposed.get("temporal_type", "general")
        claim_years = decomposed.get("years", [])
        
        for doc in labeled:
            label = doc.get("label", "").lower()
            similarity = doc.get("score", 0)
            text = doc.get("text", "")
            
            # Condition 1: Known recurring false claim
            if label in ["false", "fake"] and similarity >= 0.90:
                return {
                    "is_zombie": True,
                    "type": "known_false",
                    "matched_claim": text[:100],
                    "message": "This is a known recurring false claim"
                }
            
            # Condition 2: Recycled old news
            # If claim is "recent" but matches old "true" news
            if temporal_type == "recent" and label in ["true", "real"] and similarity >= 0.85:
                # Check for old years in evidence text
                doc_years = [int(y) for y in re.findall(r'\b(20[1-2][0-9])\b', text)]
                if doc_years:
                    old_year = max(doc_years)
                    # If evidence is > 1 year old
                    if old_year < current_year - 1:
                         return {
                            "is_zombie": True,
                            "type": "recycled_news",
                            "matched_claim": text[:100],
                            "message": f"Old news from {old_year} being shared as new"
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
        
        # MEDIUM similarity
        elif similarity_level == "medium":
            if support_score >= 0.6:
                return "likely_true"
            elif support_score <= -0.6:
                return "likely_false"
            elif zombie_check.get("has_zombie_risk"): # Check zombie risk explicitly
                 return "misleading"
            else:
                # If we have web results but no labeled evidence
                if not label_analysis.get("has_labels"):
                    return "check_web"
                return "needs_verification"
        
        # LOW or NO similarity
        else:
            # If we have web results found (even with low vector match)
            if not label_analysis.get("has_labels"):
                return "check_web"
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
