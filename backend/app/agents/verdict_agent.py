"""
Verdict Agent with Map/No-Map Verification Logic.

Uses reasoning output to generate final verdict:
- HIGH MATCH + TRUE labels → TRUE
- HIGH MATCH + FALSE labels → FALSE
- NO MATCH → LIKELY FALSE (unverified)
"""
from typing import Dict, List


class VerdictAgent:
    """
    Generates final verdict based on reasoning analysis.
    """
    
    # Sinhala explanations for each verdict
    EXPLANATIONS_SI = {
        "true": "මෙම පුවත සත්‍ය බව තහවුරු විය. දත්ත ගබඩාවේ ඇති සත්‍ය ලේබල් කළ අන්තර්ගතය සමඟ ගැලපේ.",
        "likely_true": "මෙම පුවත බොහෝ දුරට සත්‍ය විය හැක. සහාය සාක්ෂි හමු විය.",
        "needs_verification": "මෙම පුවත තවදුරටත් සත්‍යාපනය අවශ්‍ය වේ. මිශ්‍ර හෝ ප්‍රමාණවත් නොවන සාක්ෂි.",
        "likely_false": "මෙම පුවත බොහෝ දුරට අසත්‍ය විය හැක. ගැලපෙන සාක්ෂි හමු නොවීය හෝ අසත්‍ය ලේබල් වලට ගැලපේ.",
        "false": "මෙම පුවත ව්‍යාජ බව තහවුරු විය. දත්ත ගබඩාවේ ඇති ව්‍යාජ ලේබල් කළ අන්තර්ගතය සමඟ ගැලපේ."
    }
    
    # English explanations
    EXPLANATIONS_EN = {
        "true": "This claim is VERIFIED TRUE. Matches true-labeled content in the database.",
        "likely_true": "This claim is LIKELY TRUE. Supporting evidence was found.",
        "needs_verification": "This claim NEEDS VERIFICATION. Mixed or insufficient evidence.",
        "likely_false": "This claim is LIKELY FALSE. No matching evidence or matches false-labeled content.",
        "false": "This claim is VERIFIED FALSE. Matches fake-labeled content in the database."
    }
    
    def __init__(self):
        pass
    
    def generate_verdict(self, claim: dict, reasoning: dict, evidence: list) -> dict:
        """
        Generate final verdict based on reasoning analysis.
        
        Args:
            claim: Extracted claim information
            reasoning: Output from ReasoningAgent
            evidence: List of evidence documents
        
        Returns:
            Verdict with label, confidence, and explanations
        """
        # Get verdict recommendation from reasoning
        verdict_recommendation = reasoning.get('verdict_recommendation', 'needs_verification')
        match_analysis = reasoning.get('match_analysis', {})
        label_analysis = reasoning.get('label_analysis', {})
        
        # Calculate confidence based on match quality
        confidence = self._calculate_confidence(match_analysis, label_analysis, evidence)
        
        # Get explanations
        explanation_si = self.EXPLANATIONS_SI.get(verdict_recommendation, self.EXPLANATIONS_SI['needs_verification'])
        explanation_en = self.EXPLANATIONS_EN.get(verdict_recommendation, self.EXPLANATIONS_EN['needs_verification'])
        
        # Build detailed explanation
        detailed_explanation = self._build_detailed_explanation(
            verdict_recommendation, 
            match_analysis, 
            label_analysis, 
            evidence
        )
        
        # Get citations from evidence
        citations = self._extract_citations(evidence)
        
        return {
            "label": verdict_recommendation,
            "confidence": confidence,
            "explanation_si": explanation_si,
            "explanation_en": explanation_en,
            "detailed_explanation": detailed_explanation,
            "citations": citations,
            "match_level": match_analysis.get('match_level', 'none'),
            "evidence_count": len(evidence)
        }
    
    def _calculate_confidence(self, match_analysis: Dict, label_analysis: Dict, evidence: List) -> float:
        """Calculate confidence score (0-1)."""
        if not evidence:
            return 0.1  # Very low confidence when no evidence
        
        match_level = match_analysis.get('match_level', 'none')
        top_similarity = match_analysis.get('top_similarity', 0)
        labeled_count = label_analysis.get('labeled_count', 0)
        has_conflicts = label_analysis.get('has_conflicts', False)
        
        # Base confidence from match level
        if match_level == 'high':
            base_confidence = 0.7
        elif match_level == 'medium':
            base_confidence = 0.5
        else:
            base_confidence = 0.3
        
        # Adjust for labeled evidence
        if labeled_count > 0:
            base_confidence += 0.1
        
        # Reduce for conflicts
        if has_conflicts:
            base_confidence -= 0.2
        
        # Factor in top similarity
        confidence = base_confidence * (0.5 + 0.5 * top_similarity)
        
        return round(max(0.1, min(0.95, confidence)), 2)
    
    def _build_detailed_explanation(
        self, 
        verdict: str, 
        match_analysis: Dict, 
        label_analysis: Dict,
        evidence: List
    ) -> str:
        """Build detailed explanation of the verdict."""
        match_level = match_analysis.get('match_level', 'none')
        top_sim = match_analysis.get('top_similarity', 0)
        
        if match_level == 'none':
            return (
                f"No matching content was found in our database (top similarity: {top_sim:.1%}). "
                f"This claim cannot be verified against known true or false news. "
                f"Without supporting evidence, this is classified as LIKELY FALSE."
            )
        
        elif match_level == 'medium':
            return (
                f"Partial matches found (top similarity: {top_sim:.1%}). "
                f"Related content exists but not strong enough for definitive verification. "
                f"Further fact-checking is recommended."
            )
        
        else:  # high match
            label_counts = label_analysis.get('label_counts', {})
            labeled_count = label_analysis.get('labeled_count', 0)
            
            if labeled_count > 0:
                return (
                    f"Strong matches found (top similarity: {top_sim:.1%}). "
                    f"Matched {labeled_count} labeled documents with labels: {label_counts}. "
                    f"Verdict based on label analysis: {verdict.upper().replace('_', ' ')}."
                )
            else:
                return (
                    f"Matches found in live news (top similarity: {top_sim:.1%}). "
                    f"No labeled dataset matches. Verdict based on presence in recent news."
                )
    
    def _extract_citations(self, evidence: List) -> List[str]:
        """Extract citations from evidence documents."""
        citations = []
        
        for doc in evidence[:5]:  # Limit to 5 citations
            source = doc.get('source', 'Unknown')
            title = doc.get('title', doc.get('text', ''))[:50]
            url = doc.get('url', '')
            label = doc.get('label', '')
            similarity = doc.get('score', 0)
            
            citation = f"{source}: {title}..."
            if label:
                citation += f" [Label: {label}]"
            citation += f" (Similarity: {similarity:.0%})"
            if url:
                citation += f" - {url}"
            
            citations.append(citation)
        
        return citations
