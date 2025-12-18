"""
verdict_agent.py - Verdict Agent

This agent takes the reasoning output and generates the final verdict.
Uses LLM (Groq/OpenRouter) for intelligent verdict generation with citations.

It produces:
1. Final verdict label (true, false, misleading, needs_verification)
2. Confidence score (0-1)
3. Explanation with ChatGPT-style reasoning and citations
"""
from typing import Dict, List
import os
import requests
import re


class VerdictAgent:
    """
    Agent for generating final verdicts based on evidence analysis.
    Uses LLM for intelligent reasoning when API key is available.
    """
    
    # Sinhala explanations for each verdict type
    EXPLANATIONS_SI = {
        "true": "මෙම පුවත සත්‍ය බව තහවුරු විය.",
        "likely_true": "මෙම පුවත බොහෝ දුරට සත්‍ය විය හැක.",
        "needs_verification": "මෙම පුවත තවදුරටත් සත්‍යාපනය අවශ්‍ය වේ.",
        "likely_false": "මෙම පුවත බොහෝ දුරට අසත්‍ය විය හැක.",
        "false": "මෙම පුවත ව්‍යාජ බව තහවුරු විය.",
        "misleading": "මෙම පුවත නොමඟ යවන සුළුය."
    }
    
    # LLM endpoints
    LLM_ENDPOINTS = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions"
    }
    
    LLM_MODELS = {
        "groq": "llama-3.1-70b-versatile",
        "openrouter": "mistralai/mistral-7b-instruct"
    }
    
    def __init__(self):
        """Initialize the verdict agent."""
        print("[VerdictAgent] Initialized")
    
    def generate_verdict(
        self, 
        claim: dict, 
        reasoning: dict, 
        evidence: list,
        web_analysis: dict = None,
        llm_provider: str = "groq"
    ) -> dict:
        """
        Generate final verdict using LLM for intelligent reasoning.
        
        Args:
            claim: Extracted claim information
            reasoning: Output from ReasoningAgent
            evidence: List of evidence documents
            web_analysis: Results from WebResearchAgent
            llm_provider: 'groq' or 'openrouter'
        
        Returns:
            Dictionary with verdict label, confidence, and LLM explanation
        """
        print(f"[VerdictAgent] Generating verdict with LLM ({llm_provider})")
        
        # Get API key
        if llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
        else:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        # Build context from all evidence
        evidence_context = self._build_evidence_context(evidence, web_analysis)
        claim_text = claim.get("translated_claim", claim.get("original_claim", ""))
        original_claim = claim.get("original_claim", "")
        
        # Try LLM verdict
        if api_key:
            llm_result = self._call_llm_for_verdict(
                original_claim,
                claim_text,
                evidence_context,
                llm_provider,
                api_key
            )
            if llm_result:
                return llm_result
        else:
            print(f"[VerdictAgent] No API key for {llm_provider}, using rule-based")
        
        # Fallback to rule-based verdict
        return self._generate_rule_based_verdict(claim, reasoning, evidence)
    
    def _call_llm_for_verdict(
        self,
        original_claim: str,
        translated_claim: str,
        evidence_context: str,
        provider: str,
        api_key: str
    ) -> dict:
        """Call LLM to generate verdict with ChatGPT-style reasoning."""
        
        prompt = f"""You are a professional fact-checker. Analyze this claim and provide a verdict.

CLAIM (Sinhala): {original_claim}
CLAIM (English): {translated_claim}

EVIDENCE FROM MULTIPLE SOURCES:
{evidence_context}

Respond in this EXACT JSON format:
{{
    "verdict": "TRUE" or "FALSE" or "MISLEADING" or "UNVERIFIED",
    "confidence": 0.0 to 1.0,
    "reasoning": "Your detailed analysis with source citations like [Daily Mirror], [Hiru News], etc."
}}

Your reasoning should be detailed like ChatGPT:
1. Start with whether it's true/false and why
2. List what each source says with citations
3. Explain the context (political statement vs fact, etc.)
4. Use emojis: ✔ for confirmations, ❌ for contradictions

Return ONLY valid JSON, no extra text."""

        endpoint = self.LLM_ENDPOINTS.get(provider)
        model = self.LLM_MODELS.get(provider)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://sinhala-fake-news-detector.com"
            headers["X-Title"] = "Sinhala Fake News Detector"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.2
        }
        
        try:
            print(f"[VerdictAgent] Calling {provider} API...")
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print("[VerdictAgent] LLM response received")
                return self._parse_llm_response(content, original_claim)
            else:
                print(f"[VerdictAgent] API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[VerdictAgent] LLM error: {e}")
            return None
    
    def _parse_llm_response(self, content: str, original_claim: str) -> dict:
        """Parse LLM JSON response into verdict format."""
        try:
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```json?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            import json
            data = json.loads(content)
            
            verdict_map = {
                "TRUE": "true",
                "FALSE": "false", 
                "MISLEADING": "misleading",
                "UNVERIFIED": "needs_verification"
            }
            
            label = verdict_map.get(data.get("verdict", "").upper(), "needs_verification")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            
            return {
                "label": label,
                "confidence": round(min(1.0, max(0.1, confidence)), 2),
                "explanation_si": self.EXPLANATIONS_SI.get(label, ""),
                "explanation_en": reasoning,
                "detailed_explanation": reasoning,
                "citations": [],
                "match_level": "llm",
                "evidence_count": 0,
                "llm_powered": True
            }
            
        except Exception as e:
            print(f"[VerdictAgent] Parse error: {e}")
            return None
    
    def _build_evidence_context(self, evidence: list, web_analysis: dict) -> str:
        """Build context string from all evidence sources."""
        context_parts = []
        
        # From database evidence
        for i, e in enumerate(evidence[:5], 1):
            text = e.get("text", "")[:300]
            source = e.get("source", "Unknown")
            context_parts.append(f"[DB-{i}] {source}: {text}")
        
        # From web research
        if web_analysis:
            web_evidence = web_analysis.get("evidence", [])
            for i, e in enumerate(web_evidence[:5], 1):
                url = e.get("source_url", "")
                title = e.get("title", "")
                snippet = e.get("content_snippet", "")[:300]
                stance = e.get("stance", "")
                
                # Extract source name
                source_name = "Web"
                for key, name in [
                    ("dailymirror", "Daily Mirror"),
                    ("hirunews", "Hiru News"),
                    ("adaderana", "Ada Derana"),
                    ("newsfirst", "News First"),
                    ("bbc", "BBC")
                ]:
                    if key in url.lower():
                        source_name = name
                        break
                
                context_parts.append(f"[{source_name}] {title}\nStance: {stance}\nContent: {snippet}")
        
        return "\n\n".join(context_parts) if context_parts else "No evidence found."
    
    def _generate_rule_based_verdict(self, claim: dict, reasoning: dict, evidence: list) -> dict:
        """Fallback rule-based verdict generation."""
        verdict_recommendation = reasoning.get('verdict_recommendation', 'needs_verification')
        match_analysis = reasoning.get('match_analysis', {})
        label_analysis = reasoning.get('label_analysis', {})
        
        confidence = self._calculate_confidence(match_analysis, label_analysis, evidence)
        
        explanation_si = self.EXPLANATIONS_SI.get(verdict_recommendation, "")
        explanation_en = f"This claim is {verdict_recommendation.upper()}. Based on evidence analysis."
        
        citations = self._extract_citations(evidence)
        
        return {
            "label": verdict_recommendation,
            "confidence": confidence,
            "explanation_si": explanation_si,
            "explanation_en": explanation_en,
            "detailed_explanation": explanation_en,
            "citations": citations,
            "match_level": match_analysis.get('match_level', 'none'),
            "evidence_count": len(evidence),
            "llm_powered": False
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
