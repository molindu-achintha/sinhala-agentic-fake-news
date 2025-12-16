"""
reasoning_agent.py - Reasoning Agent

This agent analyzes retrieved evidence and determines:
1. Match level (high, medium, low, none)
2. Evidence labels (true, false, misleading)
3. Initial verdict recommendation
4. Confidence score

It implements the Map/No-Map logic:
- HIGH MATCH (>0.7 similarity): Check metadata labels
- MEDIUM MATCH (0.5-0.7): Needs verification  
- NO MATCH (<0.5): Likely false (no supporting evidence)
"""
from typing import List, Dict
from collections import Counter


class ReasoningAgent:
    """
    Agent for analyzing evidence and generating reasoning.
    
    This agent looks at the similarity scores and labels from
    retrieved documents to make a verdict recommendation.
    """
    
    # Similarity thresholds for matching
    HIGH_SIMILARITY_THRESHOLD = 0.7
    MEDIUM_SIMILARITY_THRESHOLD = 0.5
    
    # Trusted news sources in Sri Lanka
    TRUSTED_SOURCES = [
        "BBC Sinhala",
        "Ada Derana",
        "Lankadeepa",
        "Hiru News",
        "ITN News"
    ]
    
    # Weights for different labels
    LABEL_WEIGHTS = {
        "fake": -1.0,
        "false": -1.0,
        "misleading": -0.5,
        "partially_true": 0.3,
        "partly_true": 0.3,
        "true": 1.0,
        "real": 1.0,
        "verified": 1.0,
        "unknown": 0.0,
        "": 0.0
    }
    
    def __init__(self):
        """Initialize the reasoning agent."""
        print("[ReasoningAgent] Initialized")
    
    def reason(self, claim: str, evidence: List[Dict]) -> Dict:
        """
        Perform verification reasoning.
        
        This is the main method that coordinates all reasoning steps.
        
        Args:
            claim: The claim text to verify
            evidence: List of evidence documents from Pinecone
        
        Returns:
            Dictionary with reasoning results and verdict recommendation
        """
        print("[ReasoningAgent] Starting reasoning for claim")
        print("[ReasoningAgent] Evidence count:", len(evidence) if evidence else 0)
        
        # Handle case with no evidence
        if not evidence:
            print("[ReasoningAgent] No evidence found")
            return self._no_evidence_result()
        
        # Step 1: Analyze similarity scores
        print("[ReasoningAgent] Step 1: Analyzing matches")
        match_analysis = self._analyze_matches(evidence)
        print("[ReasoningAgent] Match level:", match_analysis['match_level'])
        
        # Step 2: Based on match level, determine verification path
        if match_analysis['match_level'] == 'high':
            # HIGH MATCH: Check metadata labels
            print("[ReasoningAgent] High match - checking labels")
            label_analysis = self._analyze_labels(evidence)
            source_analysis = self._analyze_sources(evidence)
            verdict_recommendation = self._determine_verdict_from_labels(label_analysis)
        
        elif match_analysis['match_level'] == 'medium':
            # MEDIUM MATCH: Needs verification
            print("[ReasoningAgent] Medium match - needs verification")
            label_analysis = self._analyze_labels(evidence)
            source_analysis = self._analyze_sources(evidence)
            verdict_recommendation = "needs_verification"
        
        else:
            # NO MATCH: Likely false
            print("[ReasoningAgent] No match - likely false")
            label_analysis = {"label_counts": {}, "support_score": 0, "has_conflicts": False}
            source_analysis = {"trusted_count": 0, "credibility_score": 0}
            verdict_recommendation = "likely_false"
        
        print("[ReasoningAgent] Verdict recommendation:", verdict_recommendation)
        
        # Build reasoning result
        return {
            "summary": self._generate_summary(match_analysis, verdict_recommendation),
            "match_analysis": match_analysis,
            "label_analysis": label_analysis,
            "source_analysis": source_analysis,
            "verdict_recommendation": verdict_recommendation,
            "evidence_count": len(evidence),
            "conflicting_evidence": label_analysis.get("has_conflicts", False),
            "statments": [
                {
                    "step": "Similarity Check",
                    "result": f"Match level: {match_analysis['match_level'].upper()}. "
                              f"Top similarity: {match_analysis['top_similarity']:.1%}"
                },
                {
                    "step": "Evidence Mapping",
                    "result": f"Found {match_analysis['high_matches']} high matches, "
                              f"{match_analysis['medium_matches']} medium matches"
                },
                {
                    "step": "Label Verification" if match_analysis['match_level'] != 'none' else "No Match Found",
                    "result": self._get_label_step_result(match_analysis, label_analysis, verdict_recommendation)
                },
                {
                    "step": "Verdict Recommendation",
                    "result": f"Recommended: {verdict_recommendation.upper().replace('_', ' ')}"
                }
            ]
        }
    
    def _analyze_matches(self, evidence: List[Dict]) -> Dict:
        """
        Analyze how well the claim maps to existing data.
        
        This method calculates statistics about similarity scores.
        """
        scores = [doc.get('score', 0) for doc in evidence]
        
        if not scores:
            return {
                "match_level": "none",
                "top_similarity": 0,
                "avg_similarity": 0,
                "high_matches": 0,
                "medium_matches": 0
            }
        
        top_similarity = max(scores)
        avg_similarity = sum(scores) / len(scores)
        high_matches = sum(1 for s in scores if s >= self.HIGH_SIMILARITY_THRESHOLD)
        medium_matches = sum(1 for s in scores if self.MEDIUM_SIMILARITY_THRESHOLD <= s < self.HIGH_SIMILARITY_THRESHOLD)
        
        # Determine match level based on best match
        if top_similarity >= self.HIGH_SIMILARITY_THRESHOLD:
            match_level = "high"
        elif top_similarity >= self.MEDIUM_SIMILARITY_THRESHOLD:
            match_level = "medium"
        else:
            match_level = "none"
        
        return {
            "match_level": match_level,
            "top_similarity": top_similarity,
            "avg_similarity": avg_similarity,
            "high_matches": high_matches,
            "medium_matches": medium_matches
        }
    
    def _analyze_labels(self, evidence: List[Dict]) -> Dict:
        """
        Analyze labels from matched documents.
        
        This method counts how many documents have each label type.
        """
        label_counts = Counter()
        weighted_scores = []
        labeled_evidence = []
        
        for doc in evidence:
            label = doc.get('label', '').lower().strip()
            similarity = doc.get('score', 0.5)
            
            if label and label != 'unknown':
                label_counts[label] += 1
                weight = self.LABEL_WEIGHTS.get(label, 0.0)
                weighted_scores.append(weight * similarity)
                labeled_evidence.append(doc)
        
        # Calculate support score
        support_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        
        # Check for conflicts (both fake and true labels)
        has_fake = any(l in ['fake', 'false'] for l in label_counts.keys())
        has_true = any(l in ['true', 'real', 'verified'] for l in label_counts.keys())
        
        return {
            "label_counts": dict(label_counts),
            "support_score": support_score,
            "has_conflicts": has_fake and has_true,
            "labeled_count": len(labeled_evidence),
            "fake_count": label_counts.get('fake', 0) + label_counts.get('false', 0),
            "true_count": label_counts.get('true', 0) + label_counts.get('real', 0)
        }
    
    def _analyze_sources(self, evidence: List[Dict]) -> Dict:
        """
        Analyze source credibility.
        
        This method counts how many sources are from trusted news outlets.
        """
        trusted_count = sum(1 for doc in evidence if doc.get('source') in self.TRUSTED_SOURCES)
        
        return {
            "trusted_count": trusted_count,
            "total_count": len(evidence),
            "credibility_score": trusted_count / len(evidence) if evidence else 0
        }
    
    def _determine_verdict_from_labels(self, label_analysis: Dict) -> str:
        """
        Determine verdict based on labels of matched documents.
        
        This uses the support score to make a decision.
        """
        support_score = label_analysis['support_score']
        has_conflicts = label_analysis['has_conflicts']
        labeled_count = label_analysis['labeled_count']
        
        # If no labeled evidence, cannot verify
        if labeled_count == 0:
            return "needs_verification"
        
        # If conflicting evidence
        if has_conflicts:
            return "needs_verification"
        
        # Based on support score
        if support_score <= -0.5:
            return "false"
        elif support_score < 0:
            return "likely_false"
        elif support_score < 0.3:
            return "needs_verification"
        elif support_score < 0.7:
            return "likely_true"
        else:
            return "true"
    
    def _generate_summary(self, match_analysis: Dict, verdict: str) -> str:
        """Generate human-readable summary of the reasoning."""
        match_level = match_analysis['match_level']
        
        if match_level == 'none':
            return "NO MATCHING EVIDENCE FOUND. This claim cannot be verified with existing data. Likely FALSE or unverified."
        
        elif match_level == 'medium':
            return "PARTIAL MATCH FOUND. Some related content exists but requires further verification."
        
        else:
            verdicts = {
                "true": "VERIFIED TRUE. Claim matches verified true content in database.",
                "likely_true": "LIKELY TRUE. Matched content suggests claim is accurate.",
                "needs_verification": "NEEDS VERIFICATION. Mixed or insufficient evidence.",
                "likely_false": "LIKELY FALSE. Matched content suggests claim is inaccurate.",
                "false": "VERIFIED FALSE. Claim matches known false/fake content in database."
            }
            return verdicts.get(verdict, "Unable to determine.")
    
    def _get_label_step_result(self, match_analysis: Dict, label_analysis: Dict, verdict: str) -> str:
        """Get result text for label verification step."""
        if match_analysis['match_level'] == 'none':
            return "No similar content in database. Cannot verify against known data."
        
        labeled = label_analysis['labeled_count']
        if labeled > 0:
            return f"Checked {labeled} labeled documents. Labels: {label_analysis['label_counts']}"
        else:
            return "Matched live news only (no labeled dataset matches)."
    
    def _no_evidence_result(self) -> Dict:
        """Return result when no evidence is found."""
        return {
            "summary": "NO EVIDENCE RETRIEVED. Cannot verify this claim. Likely FALSE or completely new information.",
            "match_analysis": {"match_level": "none", "top_similarity": 0},
            "label_analysis": {},
            "source_analysis": {},
            "verdict_recommendation": "likely_false",
            "evidence_count": 0,
            "conflicting_evidence": False,
            "statments": [
                {"step": "Evidence Retrieval", "result": "No matching documents found in database."},
                {"step": "Verdict", "result": "LIKELY FALSE - No supporting evidence exists."}
            ]
        }
