"""
reasoning_agent.py - Reasoning Agent

This agent analyzes retrieved evidence and determines verdict.

Thresholds:
- Dataset evidence: 0.7 (standard threshold)
- Live news (trusted sources): 0.8 (higher threshold)
"""
from typing import List, Dict
from collections import Counter


class ReasoningAgent:
    """
    Agent for analyzing evidence and generating reasoning.
    """
    
    # Similarity thresholds
    HIGH_THRESHOLD_DATASET = 0.7     # Standard for labeled dataset
    HIGH_THRESHOLD_LIVE_NEWS = 0.8   # Higher for scraped news
    MEDIUM_THRESHOLD = 0.5
    
    # Trusted news sources
    TRUSTED_SOURCES = [
        "BBC Sinhala",
        "Ada Derana", 
        "Lankadeepa",
        "Hiru News",
        "ITN News"
    ]
    
    # Label weights
    LABEL_WEIGHTS = {
        "fake": -1.0,
        "false": -1.0,
        "misleading": -0.5,
        "partially_true": 0.3,
        "true": 1.0,
        "real": 1.0,
        "verified": 1.0,
        "unknown": 0.0,
        "": 0.0
    }
    
    def __init__(self):
        """Initialize agent."""
        print("[ReasoningAgent] Initialized")
        print("[ReasoningAgent] Dataset threshold:", self.HIGH_THRESHOLD_DATASET)
        print("[ReasoningAgent] Live news threshold:", self.HIGH_THRESHOLD_LIVE_NEWS)
    
    def reason(self, claim: str, evidence: List[Dict]) -> Dict:
        """
        Main reasoning method.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence from Pinecone
        
        Returns:
            Reasoning result with verdict recommendation
        """
        print("[ReasoningAgent] Starting reasoning")
        print("[ReasoningAgent] Evidence count:", len(evidence) if evidence else 0)
        
        # No evidence case
        if not evidence:
            print("[ReasoningAgent] No evidence found")
            return self._no_evidence_result()
        
        # Step 1: Separate evidence by type
        print("[ReasoningAgent] Step 1: Separating evidence by type")
        dataset_evidence = [e for e in evidence if e.get('type') == 'dataset']
        live_news_evidence = [e for e in evidence if e.get('type') == 'live_news']
        
        print(f"[ReasoningAgent] Dataset evidence: {len(dataset_evidence)}")
        print(f"[ReasoningAgent] Live news evidence: {len(live_news_evidence)}")
        
        # Step 2: Analyze with appropriate thresholds
        print("[ReasoningAgent] Step 2: Analyzing matches")
        dataset_analysis = self._analyze_matches(dataset_evidence, self.HIGH_THRESHOLD_DATASET)
        live_news_analysis = self._analyze_matches(live_news_evidence, self.HIGH_THRESHOLD_LIVE_NEWS)
        
        # Combine analysis
        match_analysis = self._combine_analysis(dataset_analysis, live_news_analysis)
        print("[ReasoningAgent] Combined match level:", match_analysis['match_level'])
        
        # Step 3: Determine verdict
        print("[ReasoningAgent] Step 3: Determining verdict")
        if match_analysis['match_level'] == 'high':
            label_analysis = self._analyze_labels(evidence)
            source_analysis = self._analyze_sources(evidence)
            verdict_recommendation = self._determine_verdict(label_analysis)
        
        elif match_analysis['match_level'] == 'medium':
            label_analysis = self._analyze_labels(evidence)
            source_analysis = self._analyze_sources(evidence)
            verdict_recommendation = "needs_verification"
        
        else:
            label_analysis = {"label_counts": {}, "support_score": 0, "has_conflicts": False}
            source_analysis = {"trusted_count": 0, "credibility_score": 0}
            verdict_recommendation = "likely_false"
        
        print("[ReasoningAgent] Verdict:", verdict_recommendation)
        
        return {
            "summary": self._make_summary(match_analysis, verdict_recommendation),
            "match_analysis": match_analysis,
            "label_analysis": label_analysis,
            "source_analysis": source_analysis,
            "verdict_recommendation": verdict_recommendation,
            "evidence_count": len(evidence),
            "dataset_matches": len(dataset_evidence),
            "live_news_matches": len(live_news_evidence),
            "conflicting_evidence": label_analysis.get("has_conflicts", False),
            "statments": [
                {
                    "step": "Evidence Classification",
                    "result": f"Dataset: {len(dataset_evidence)}, Live News: {len(live_news_evidence)}"
                },
                {
                    "step": "Similarity Check",
                    "result": f"Match level: {match_analysis['match_level'].upper()}, "
                              f"Top score: {match_analysis['top_similarity']:.1%}"
                },
                {
                    "step": "Label Analysis",
                    "result": self._get_label_result(label_analysis)
                },
                {
                    "step": "Verdict",
                    "result": verdict_recommendation.upper().replace('_', ' ')
                }
            ]
        }
    
    def _analyze_matches(self, evidence: List[Dict], threshold: float) -> Dict:
        """Analyze matches using given threshold."""
        if not evidence:
            return {
                "match_level": "none",
                "top_similarity": 0,
                "high_matches": 0,
                "medium_matches": 0,
                "threshold_used": threshold
            }
        
        scores = [doc.get('score', 0) for doc in evidence]
        top_similarity = max(scores)
        high_matches = sum(1 for s in scores if s >= threshold)
        medium_matches = sum(1 for s in scores if self.MEDIUM_THRESHOLD <= s < threshold)
        
        # Determine match level
        if top_similarity >= threshold:
            match_level = "high"
        elif top_similarity >= self.MEDIUM_THRESHOLD:
            match_level = "medium"
        else:
            match_level = "none"
        
        return {
            "match_level": match_level,
            "top_similarity": top_similarity,
            "high_matches": high_matches,
            "medium_matches": medium_matches,
            "threshold_used": threshold
        }
    
    def _combine_analysis(self, dataset_analysis: Dict, live_news_analysis: Dict) -> Dict:
        """Combine analysis from both evidence types."""
        # Priority: dataset > live_news (dataset has verified labels)
        if dataset_analysis['match_level'] == 'high':
            print("[ReasoningAgent] High match from dataset")
            return {
                **dataset_analysis,
                "primary_source": "dataset"
            }
        
        if live_news_analysis['match_level'] == 'high':
            print("[ReasoningAgent] High match from live news (trusted sources)")
            return {
                **live_news_analysis,
                "primary_source": "live_news"
            }
        
        # Use medium if any
        if dataset_analysis['match_level'] == 'medium' or live_news_analysis['match_level'] == 'medium':
            return {
                "match_level": "medium",
                "top_similarity": max(
                    dataset_analysis.get('top_similarity', 0),
                    live_news_analysis.get('top_similarity', 0)
                ),
                "high_matches": 0,
                "medium_matches": dataset_analysis.get('medium_matches', 0) + 
                                  live_news_analysis.get('medium_matches', 0),
                "primary_source": "combined"
            }
        
        return {
            "match_level": "none",
            "top_similarity": 0,
            "high_matches": 0,
            "medium_matches": 0,
            "primary_source": "none"
        }
    
    def _analyze_labels(self, evidence: List[Dict]) -> Dict:
        """Analyze labels from evidence."""
        label_counts = Counter()
        weighted_scores = []
        
        for doc in evidence:
            label = doc.get('label', '').lower().strip()
            similarity = doc.get('score', 0.5)
            
            if label and label != 'unknown':
                label_counts[label] += 1
                weight = self.LABEL_WEIGHTS.get(label, 0.0)
                weighted_scores.append(weight * similarity)
        
        support_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        
        has_fake = any(l in ['fake', 'false'] for l in label_counts.keys())
        has_true = any(l in ['true', 'real'] for l in label_counts.keys())
        
        return {
            "label_counts": dict(label_counts),
            "support_score": support_score,
            "has_conflicts": has_fake and has_true,
            "labeled_count": sum(label_counts.values()),
            "true_count": label_counts.get('true', 0) + label_counts.get('real', 0),
            "false_count": label_counts.get('false', 0) + label_counts.get('fake', 0)
        }
    
    def _analyze_sources(self, evidence: List[Dict]) -> Dict:
        """Analyze source credibility."""
        trusted = sum(1 for doc in evidence if doc.get('source') in self.TRUSTED_SOURCES)
        total = len(evidence)
        
        return {
            "trusted_count": trusted,
            "total_count": total,
            "credibility_score": trusted / total if total > 0 else 0
        }
    
    def _determine_verdict(self, label_analysis: Dict) -> str:
        """Determine verdict from labels."""
        score = label_analysis['support_score']
        
        if label_analysis['labeled_count'] == 0:
            return "needs_verification"
        
        if label_analysis['has_conflicts']:
            return "needs_verification"
        
        if score <= -0.5:
            return "false"
        elif score < 0:
            return "likely_false"
        elif score < 0.3:
            return "needs_verification"
        elif score < 0.7:
            return "likely_true"
        else:
            return "true"
    
    def _make_summary(self, match_analysis: Dict, verdict: str) -> str:
        """Make human-readable summary."""
        match_level = match_analysis['match_level']
        
        if match_level == 'none':
            return "NO MATCHING EVIDENCE. Claim cannot be verified."
        elif match_level == 'medium':
            return "PARTIAL MATCH. Needs further verification."
        else:
            summaries = {
                "true": "VERIFIED TRUE. Matches verified content.",
                "likely_true": "LIKELY TRUE. Evidence supports claim.",
                "needs_verification": "NEEDS VERIFICATION. Mixed evidence.",
                "likely_false": "LIKELY FALSE. Evidence contradicts claim.",
                "false": "VERIFIED FALSE. Matches known fake content."
            }
            return summaries.get(verdict, "Unable to determine.")
    
    def _get_label_result(self, label_analysis: Dict) -> str:
        """Get label analysis result text."""
        if label_analysis.get('labeled_count', 0) > 0:
            return f"Labels: {label_analysis.get('label_counts', {})}"
        return "No labeled evidence found"
    
    def _no_evidence_result(self) -> Dict:
        """Return result when no evidence found."""
        return {
            "summary": "NO EVIDENCE. Cannot verify. Likely FALSE.",
            "match_analysis": {"match_level": "none", "top_similarity": 0},
            "label_analysis": {},
            "source_analysis": {},
            "verdict_recommendation": "likely_false",
            "evidence_count": 0,
            "dataset_matches": 0,
            "live_news_matches": 0,
            "conflicting_evidence": False,
            "statments": [
                {"step": "Evidence", "result": "No matching documents."},
                {"step": "Verdict", "result": "LIKELY FALSE"}
            ]
        }
