"""
verdict_agent.py - Verdict Agent

This agent takes the reasoning output and generates the final verdict.
It produces:
1. Final verdict label (true, false, misleading, needs_verification)
2. Confidence score (0-1)
3. Explanation in Sinhala and English
4. Citations from evidence
"""
from typing import Dict, List


class VerdictAgent:
    """
    Agent for generating final verdicts based on reasoning analysis.
    
    This agent takes the output from ReasoningAgent and produces
    user-friendly verdicts with explanations.
    """
    
    # Sinhala explanations for each verdict type
    EXPLANATIONS_SI = {
        "true": "මෙම පුවත සත්‍ය බව තහවුරු විය. දත්ත ගබඩාවේ ඇති සත්‍ය ලේබල් කළ අන්තර්ගතය සමඟ ගැලපේ.",
        "likely_true": "මෙම පුවත බොහෝ දුරට සත්‍ය විය හැක. සහාය සාක්ෂි හමු විය.",
        "needs_verification": "මෙම පුවත තවදුරටත් සත්‍යාපනය අවශ්‍ය වේ. මිශ්‍ර හෝ ප්‍රමාණවත් නොවන සාක්ෂි.",
        "likely_false": "මෙම පුවත බොහෝ දුරට අසත්‍ය විය හැක. ගැලපෙන සාක්ෂි හමු නොවීය හෝ අසත්‍ය ලේබල් වලට ගැලපේ.",
        "false": "මෙම පුවත ව්‍යාජ බව තහවුරු විය. දත්ත ගබඩාවේ ඇති ව්‍යාජ ලේබල් කළ අන්තර්ගතය සමඟ ගැලපේ."
    }
    
    # English explanations for each verdict type
    EXPLANATIONS_EN = {
        "true": "This claim is VERIFIED TRUE. Matches true-labeled content in the database.",
        "likely_true": "This claim is LIKELY TRUE. Supporting evidence was found.",
        "needs_verification": "This claim NEEDS VERIFICATION. Mixed or insufficient evidence.",
        "likely_false": "This claim is LIKELY FALSE. No matching evidence or matches false-labeled content.",
        "false": "This claim is VERIFIED FALSE. Matches fake-labeled content in the database."
    }
    
    def __init__(self):
        """Initialize the verdict agent."""
        print("[VerdictAgent] Initialized")
    
    def generate_verdict(self, claim: dict, reasoning: dict, evidence: list) -> dict:
        """
        Generate final verdict based on reasoning analysis.
        
        Args:
            claim: Extracted claim information
            reasoning: Output from ReasoningAgent
            evidence: List of evidence documents
        
        Returns:
            Dictionary with verdict label, confidence, and explanations
        """
        print("[VerdictAgent] Generating verdict")
        
        # Get verdict recommendation from reasoning
        verdict_recommendation = reasoning.get('verdict_recommendation', 'needs_verification')
        match_analysis = reasoning.get('match_analysis', {})
        label_analysis = reasoning.get('label_analysis', {})
        
        print("[VerdictAgent] Verdict recommendation:", verdict_recommendation)
        
        # Calculate confidence based on match quality
        confidence = self._calculate_confidence(match_analysis, label_analysis, evidence)
        print("[VerdictAgent] Confidence score:", confidence)
        
        # Get explanations in both languages
        explanation_si = self.EXPLANATIONS_SI.get(
            verdict_recommendation, 
            self.EXPLANATIONS_SI['needs_verification']
        )
        explanation_en = self.EXPLANATIONS_EN.get(
            verdict_recommendation, 
            self.EXPLANATIONS_EN['needs_verification']
        )
        
        # Build detailed explanation
        detailed_explanation = self._build_detailed_explanation(
            verdict_recommendation, 
            match_analysis, 
            label_analysis, 
            evidence
        )
        
        # Get citations from evidence
        citations = self._extract_citations(evidence)
        print("[VerdictAgent] Citations count:", len(citations))
        
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
        """
        Calculate confidence score between 0 and 1.
        
        Higher scores mean more confidence in the verdict.
        Boosts confidence when web search confirms the claim.
        """
        if not evidence:
            print("[VerdictAgent] No evidence - very low confidence")
            return 0.1
        
        match_level = match_analysis.get('match_level', 'none')
        top_similarity = match_analysis.get('top_similarity', 0)
        labeled_count = label_analysis.get('labeled_count', 0)
        has_conflicts = label_analysis.get('has_conflicts', False)
        web_count = match_analysis.get('web_count', 0)
        
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
        
        # BOOST for web search results
        if web_count > 0:
            web_boost = min(0.2, web_count * 0.05)  # Up to +0.2 for web results
            base_confidence += web_boost
            print(f"[VerdictAgent] Web search boost: +{web_boost:.2f} ({web_count} results)")
        
        # Reduce for conflicting evidence
        if has_conflicts:
            base_confidence -= 0.2
        
        # Factor in top similarity
        confidence = base_confidence * (0.5 + 0.5 * top_similarity)
        
        # Keep confidence in valid range
        return round(max(0.1, min(0.95, confidence)), 2)
    
    def _build_detailed_explanation(
        self, 
        verdict: str, 
        match_analysis: Dict, 
        label_analysis: Dict,
        evidence: List
    ) -> str:
        """
        Build detailed explanation of the verdict.
        
        This provides more context than the simple explanation.
        """
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
        
        else:
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
        """
        Extract citations from evidence documents.
        
        Creates readable citation strings from evidence metadata.
        Includes all sources without limit.
        """
        citations = []
        
        for doc in evidence:
            source = doc.get('source', 'Unknown')
            title = doc.get('title', doc.get('text', ''))[:80]
            url = doc.get('url', '')
            label = doc.get('label', '')
            similarity = doc.get('score', 0)
            is_web = 'web' in source.lower() or url.startswith('http')
            
            # Build citation string
            if is_web:
                citation = f"[Web] {source}: {title}..."
            else:
                citation = f"{source}: {title}..."
            
            if label and label != 'unverified':
                citation += f" [Label: {label}]"
            citation += f" (Similarity: {similarity:.0%})"
            if url:
                citation += f" - {url}"
            
            citations.append(citation)
        
        return citations
