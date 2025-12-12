"""
Enhanced Reasoning Agent using Pinecone and preprocessed labeled data.

Uses:
- Evidence retrieved from Pinecone (dataset + live_news namespaces)
- Labels from preprocessed data (fake, true, misleading, etc.)
- Source credibility scoring
- Content similarity analysis
"""
from typing import List, Dict
from collections import Counter


class ReasoningAgent:
    """
    Multi-agent reasoning for claim verification.
    
    Analyzes:
    1. Labels from preprocessed evidence (fake/true/misleading)
    2. Source credibility (trusted sources)
    3. Evidence similarity scores
    4. Content analysis
    """
    
    # Trusted news sources (higher credibility)
    TRUSTED_SOURCES = [
        "BBC Sinhala",
        "Ada Derana", 
        "Lankadeepa",
        "Hiru News",
        "ITN News"
    ]
    
    # Label weights for verdict calculation
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
        pass
    
    def reason(self, claim: str, evidence: List[Dict]) -> Dict:
        """
        Perform multi-agent reasoning on claim and evidence.
        
        Args:
            claim: The claim text to verify
            evidence: List of evidence documents from Pinecone
                      Each doc has: id, text, title, source, label, score, similarity
        
        Returns:
            Reasoning results with support/contradiction analysis
        """
        if not evidence:
            return self._empty_result()
        
        # Agent A: Source Credibility Analysis
        source_analysis = self._analyze_sources(evidence)
        
        # Agent B: Label-based Analysis 
        label_analysis = self._analyze_labels(evidence)
        
        # Agent C: Similarity Score Analysis
        similarity_analysis = self._analyze_similarity(evidence)
        
        # Agent D: Content Analysis
        content_analysis = self._analyze_content(claim, evidence)
        
        # Combine all analyses
        combined_score = self._calculate_combined_score(
            source_analysis,
            label_analysis, 
            similarity_analysis
        )
        
        # Determine overall stance
        stance = self._determine_stance(combined_score, label_analysis)
        
        return {
            "summary": self._generate_summary(stance, evidence),
            "combined_score": round(combined_score, 3),
            "stance": stance,
            "evidence_count": len(evidence),
            "evidence_gaps": self._identify_gaps(evidence),
            "conflicting_evidence": label_analysis["has_conflicts"],
            "source_analysis": source_analysis,
            "label_analysis": label_analysis,
            "similarity_analysis": similarity_analysis,
            "statments": [
                {
                    "step": "Source Credibility",
                    "result": f"{source_analysis['trusted_count']}/{len(evidence)} from trusted sources. "
                              f"Credibility: {source_analysis['credibility_score']:.1%}"
                },
                {
                    "step": "Label Analysis (Preprocessed Data)",
                    "result": f"Labels found: {label_analysis['label_counts']}. "
                              f"Support score: {label_analysis['support_score']:.2f}"
                },
                {
                    "step": "Similarity Analysis",
                    "result": f"Avg similarity: {similarity_analysis['avg_similarity']:.1%}. "
                              f"Top match: {similarity_analysis['top_score']:.1%}"
                },
                {
                    "step": "Content Analysis",
                    "result": f"Analyzed {len(evidence)} documents. {content_analysis['summary']}"
                },
                {
                    "step": "Final Assessment",
                    "result": f"Combined score: {combined_score:.3f}. Stance: {stance}"
                }
            ]
        }
    
    def _analyze_sources(self, evidence: List[Dict]) -> Dict:
        """Analyze source credibility."""
        trusted_count = 0
        source_counts = Counter()
        
        for doc in evidence:
            source = doc.get('source', 'unknown')
            source_counts[source] += 1
            if source in self.TRUSTED_SOURCES:
                trusted_count += 1
        
        credibility_score = trusted_count / len(evidence) if evidence else 0
        
        return {
            "trusted_count": trusted_count,
            "total_count": len(evidence),
            "credibility_score": credibility_score,
            "source_distribution": dict(source_counts)
        }
    
    def _analyze_labels(self, evidence: List[Dict]) -> Dict:
        """Analyze labels from preprocessed data."""
        label_counts = Counter()
        weighted_scores = []
        
        for doc in evidence:
            label = doc.get('label', '').lower().strip()
            label_counts[label if label else 'unlabeled'] += 1
            
            # Get weight for this label
            weight = self.LABEL_WEIGHTS.get(label, 0.0)
            similarity = doc.get('score', 0.5)
            
            # Weighted by similarity score
            weighted_scores.append(weight * similarity)
        
        # Calculate support score (-1 to 1)
        support_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        
        # Check for conflicts (both fake and true labels present)
        has_fake = any(l in ['fake', 'false'] for l in label_counts.keys())
        has_true = any(l in ['true', 'real', 'verified'] for l in label_counts.keys())
        has_conflicts = has_fake and has_true
        
        return {
            "label_counts": dict(label_counts),
            "support_score": support_score,
            "has_conflicts": has_conflicts,
            "fake_count": label_counts.get('fake', 0) + label_counts.get('false', 0),
            "true_count": label_counts.get('true', 0) + label_counts.get('real', 0)
        }
    
    def _analyze_similarity(self, evidence: List[Dict]) -> Dict:
        """Analyze similarity scores from Pinecone."""
        scores = [doc.get('score', 0) for doc in evidence]
        
        if not scores:
            return {"avg_similarity": 0, "top_score": 0, "min_score": 0}
        
        return {
            "avg_similarity": sum(scores) / len(scores),
            "top_score": max(scores),
            "min_score": min(scores),
            "high_similarity_count": sum(1 for s in scores if s > 0.7)
        }
    
    def _analyze_content(self, claim: str, evidence: List[Dict]) -> Dict:
        """Analyze content of evidence documents."""
        # Simple content analysis
        evidence_texts = [doc.get('text', '') or doc.get('title', '') for doc in evidence]
        
        # Count evidence types
        live_news_count = sum(1 for doc in evidence if doc.get('type') == 'live_news')
        dataset_count = len(evidence) - live_news_count
        
        return {
            "summary": f"{dataset_count} from dataset, {live_news_count} from live news.",
            "live_news_count": live_news_count,
            "dataset_count": dataset_count
        }
    
    def _calculate_combined_score(
        self, 
        source_analysis: Dict, 
        label_analysis: Dict,
        similarity_analysis: Dict
    ) -> float:
        """Calculate combined reasoning score."""
        # Weights for different factors
        source_weight = 0.2
        label_weight = 0.5  # Labels are most important
        similarity_weight = 0.3
        
        # Normalize scores to 0-1 range
        source_score = source_analysis['credibility_score']
        label_score = (label_analysis['support_score'] + 1) / 2  # -1 to 1 -> 0 to 1
        similarity_score = similarity_analysis['avg_similarity']
        
        combined = (
            source_weight * source_score +
            label_weight * label_score +
            similarity_weight * similarity_score
        )
        
        return combined
    
    def _determine_stance(self, combined_score: float, label_analysis: Dict) -> str:
        """Determine overall stance on the claim."""
        support_score = label_analysis['support_score']
        
        if support_score < -0.3:
            return "likely_fake"
        elif support_score < 0:
            return "possibly_misleading"
        elif support_score < 0.3:
            return "needs_verification"
        elif support_score < 0.6:
            return "possibly_true"
        else:
            return "likely_true"
    
    def _identify_gaps(self, evidence: List[Dict]) -> List[str]:
        """Identify gaps in evidence."""
        gaps = []
        
        if len(evidence) < 3:
            gaps.append("Limited evidence found (less than 3 sources)")
        
        # Check for low similarity
        avg_sim = sum(doc.get('score', 0) for doc in evidence) / len(evidence) if evidence else 0
        if avg_sim < 0.5:
            gaps.append("Evidence has low relevance to claim")
        
        # Check for unlabeled evidence
        unlabeled = sum(1 for doc in evidence if not doc.get('label'))
        if unlabeled == len(evidence):
            gaps.append("No labeled evidence from dataset (only live news)")
        
        return gaps
    
    def _generate_summary(self, stance: str, evidence: List[Dict]) -> str:
        """Generate human-readable summary."""
        summaries = {
            "likely_fake": "Evidence suggests this claim is likely FALSE. Multiple sources indicate misinformation.",
            "possibly_misleading": "This claim appears to be MISLEADING. Some evidence contradicts the claim.",
            "needs_verification": "This claim NEEDS VERIFICATION. Evidence is inconclusive.",
            "possibly_true": "This claim is POSSIBLY TRUE. Some supporting evidence found.",
            "likely_true": "Evidence suggests this claim is likely TRUE. Multiple trusted sources confirm."
        }
        
        return summaries.get(stance, "Unable to determine claim veracity.")
    
    def _empty_result(self) -> Dict:
        """Return empty result when no evidence found."""
        return {
            "summary": "No evidence found to verify this claim.",
            "combined_score": 0,
            "stance": "no_evidence",
            "evidence_count": 0,
            "evidence_gaps": ["No evidence retrieved from Pinecone"],
            "conflicting_evidence": False,
            "statments": [
                {"step": "Evidence Retrieval", "result": "No matching documents found."}
            ]
        }
