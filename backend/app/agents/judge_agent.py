"""
judge_agent.py - Stage 2: Judge Agent

This agent takes the evidence JSON from Research Agent and produces
a verdict with Sinhala explanation and inline citations.

Input: Evidence JSON from Research Agent
Output: Verdict (TRUE/FALSE/PARTLY_TRUE/UNVERIFIED) + Sinhala explanation
"""
import os
import json
import requests
import re
from typing import Dict, Optional


class JudgeAgent:
    """
    Stage 2 Agent: Produces verdict and Sinhala explanation based on evidence.
    Does NOT use any tools - relies only on provided evidence JSON.
    """
    
    SYSTEM_PROMPT = """You are a FACT-CHECKING JUDGE for news claims.

You will receive a single JSON object produced by a web research agent.
It already contains:
- the original claim,
- normalized Sinhala and English versions of the claim,
- a list of evidence snippets from real web pages,
- and, for each snippet, a relation label (SUPPORTS/REFUTES/IRRELEVANT), source type, outlet, date, and URL.

Your job is to:
1. Decide whether the claim is TRUE, FALSE, PARTLY_TRUE, or UNVERIFIED.
2. Produce a clear explanation in SINHALA, with inline citations, based ONLY on the evidence list.

You MUST NOT browse or invent new sources.
You MUST NOT make up new URLs or snippets.
You MUST BASE EVERYTHING ONLY on the provided JSON evidence.

=====================
DECISION RULES
=====================

Output exactly one of these verdict labels:

- TRUE: Multiple credible evidence items with relation = SUPPORTS. No strong REFUTES from credible sources.
- FALSE: Multiple credible evidence items with relation = REFUTES. SUPPORTS (if any) comes mostly from low-credibility evidence.
- PARTLY_TRUE: Credible SUPPORT for some parts, but also credible REFUTE for others, or claim is exaggerated/misleading.
- UNVERIFIED: Evidence is too weak, sparse, or conflicting to decide.

Source weighting:
- Give more weight to: source_type = "official", "intl_mainstream", "local_mainstream" with credibility_hint "high".
- Be cautious about: source_type = "other" or "local_other" with credibility_hint "low".

=====================
OUTPUT FORMAT
=====================

Your response must be a single plaintext output (no JSON).
Use this structure:

1) Verdict line: `තීන්දුව: <LABEL>`
   where <LABEL> is: TRUE, FALSE, PARTLY_TRUE, or UNVERIFIED

2) Restate the claim in Sinhala (use claim_normalized_si)

3) Explanation in Sinhala:
   - What supporting evidence says
   - What refuting evidence says
   - Which sources are most credible
   - Whether the claim is fully accurate, partially accurate, or not supported
   - Mention if there's a difference between local Sinhala coverage and international/official sources

4) Time awareness:
   - Mention the cutoff_time_utc, e.g.: "මෙම තීන්දුව 2025-12-18T12:34:56Z (UTC) වන තොරතුරු මත පදනම්වයි."

5) Inline citations:
   - Every key factual sentence MUST have at least one citation [id]
   - Use the `id` field from the evidence array
   - Example: "ශ්‍රී ලංකා මහ බැංකුවේ වාර්තාව අනුව රන් තොගය සම්පූර්ණයෙන්ම විකිණී නොමැති බව පැහැදිලි කරයි [2][3]."

6) Citation list at the end:
   - Each line: `[id] outlet (lang, source_type), "title", date, url`
   - Example: `[1] Reuters (en, intl_mainstream), "Sri Lanka retains gold reserves", 2024-09-20, https://...`

=====================
SAFETY RULES
=====================

- You MAY NOT introduce any new URLs or sources beyond those in the evidence list.
- You MAY NOT invent facts not implied by the snippets.
- If evidence is insufficient, choose UNVERIFIED instead of guessing.
- If local media suggests something but official/intl sources contradict, explain that tension.

Language:
- The explanation and verdict MUST be in SINHALA.
- You may use English terms (e.g., IMF, GDP) inside Sinhala sentences where natural."""

    USER_PROMPT_TEMPLATE = """Here is the evidence JSON from the research agent. Please judge the claim and explain in Sinhala with citations, following your instructions.

EVIDENCE_JSON:
{evidence_json}
"""

    def __init__(self):
        """Initialize the Judge Agent."""
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "alibaba/tongyi-deepresearch-30b-a3b"
        print("[JudgeAgent] Initialized with DeepResearch model")
    
    def judge(self, evidence_json: Dict, api_key: str = None) -> Dict:
        """
        Judge the claim based on the evidence and produce a Sinhala verdict.
        
        Args:
            evidence_json: The structured evidence from Research Agent
            
        Returns:
            Dict with verdict, explanation_si, and parsed data
        """
        print("[JudgeAgent] Starting judgment...")
        
        # Use passed API key or fallback to env var
        current_api_key = api_key if api_key else self.api_key
        
        if not current_api_key:
            print("[JudgeAgent] No API key, returning default verdict")
            return self._create_default_verdict(evidence_json)
        
        # Format evidence as JSON string
        evidence_str = json.dumps(evidence_json, ensure_ascii=False, indent=2)
        user_prompt = self.USER_PROMPT_TEMPLATE.format(evidence_json=evidence_str)
        
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sinhala-fake-news-detector.com",
            "X-Title": "Sinhala Fake News Detector"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.1
        }
        
        try:
            print("[JudgeAgent] Calling DeepResearch API...")
            response = requests.post(
                self.endpoint, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print("[JudgeAgent] Response received, parsing verdict...")
                return self._parse_verdict(content, evidence_json)
            else:
                print(f"[JudgeAgent] API error: {response.status_code}")
                return self._create_default_verdict(evidence_json)
                
        except Exception as e:
            print(f"[JudgeAgent] Error: {e}")
            return self._create_default_verdict(evidence_json)
    
    def _parse_verdict(self, content: str, evidence_json: Dict) -> Dict:
        """Parse the judge's verdict from the response."""
        content = content.strip()
        
        # Extract verdict label
        verdict_label = "needs_verification"
        verdict_match = re.search(r'තීන්දුව:\s*(TRUE|FALSE|PARTLY_TRUE|UNVERIFIED)', content, re.IGNORECASE)
        if verdict_match:
            label = verdict_match.group(1).upper()
            label_map = {
                "TRUE": "true",
                "FALSE": "false",
                "PARTLY_TRUE": "misleading",
                "UNVERIFIED": "needs_verification"
            }
            verdict_label = label_map.get(label, "needs_verification")
        
        # Calculate confidence based on evidence
        evidence_list = evidence_json.get("evidence", [])
        supports = sum(1 for e in evidence_list if e.get("relation") == "SUPPORTS")
        refutes = sum(1 for e in evidence_list if e.get("relation") == "REFUTES")
        total = supports + refutes
        
        if total > 0:
            if verdict_label in ["true", "false"]:
                confidence = min(0.95, 0.6 + (total * 0.05))
            else:
                confidence = 0.5
        else:
            confidence = 0.3
        
        # Build citations list
        citations = []
        for ev in evidence_list:
            citations.append({
                "id": ev.get("id"),
                "outlet": ev.get("outlet", "Unknown"),
                "url": ev.get("url", ""),
                "relation": ev.get("relation", "IRRELEVANT")
            })
        
        return {
            "label": verdict_label,
            "confidence": round(confidence, 2),
            "explanation_si": content,
            "explanation_en": "",  # Could add translation later
            "detailed_explanation": content,
            "citations": citations,
            "evidence_count": len(evidence_list),
            "supports_count": supports,
            "refutes_count": refutes,
            "llm_powered": True,
            "claim_normalized_si": evidence_json.get("claim_normalized_si", ""),
            "claim_normalized_en": evidence_json.get("claim_normalized_en", "")
        }
    
    def _create_default_verdict(self, evidence_json: Dict) -> Dict:
        """Create a default verdict when judgment fails."""
        evidence_list = evidence_json.get("evidence", [])
        
        return {
            "label": "needs_verification",
            "confidence": 0.3,
            "explanation_si": "මෙම පුවත තවදුරටත් සත්‍යාපනය අවශ්‍ය වේ. සාක්ෂි ප්‍රමාණවත් නොවේ.",
            "explanation_en": "This claim needs further verification. Evidence is insufficient.",
            "detailed_explanation": "Unable to generate detailed analysis.",
            "citations": [],
            "evidence_count": len(evidence_list),
            "supports_count": 0,
            "refutes_count": 0,
            "llm_powered": False,
            "claim_normalized_si": evidence_json.get("claim_normalized_si", ""),
            "claim_normalized_en": evidence_json.get("claim_normalized_en", "")
        }


# Singleton instance
_judge_agent = None

def get_judge_agent() -> JudgeAgent:
    """Get or create the Judge Agent singleton."""
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = JudgeAgent()
    return _judge_agent
