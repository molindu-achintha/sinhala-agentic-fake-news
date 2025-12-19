"""
research_agent.py - Stage 1: Research Agent

This agent uses DeepResearch (via OpenRouter) with web search capabilities
to gather evidence for fact-checking claims.

Input: Raw claim (Sinhala/English/Singlish/mixed)
Output: Structured JSON with normalized claims and evidence list
"""
import os
import json
import requests
from typing import Dict, Optional
from datetime import datetime, timezone


class ResearchAgent:
    """
    Stage 1 Agent: Gathers evidence from the web using DeepResearch.
    Uses Search and Visit tools to collect evidence snippets.
    """
    
    SYSTEM_PROMPT = """You are a WEB RESEARCH AGENT for FACT-CHECKING.

Your ONLY job is to gather and structure evidence for or against a NEWS CLAIM.
You will NOT give a verdict (TRUE/FALSE). You will NOT write a natural-language explanation.
You will ONLY return a single JSON object summarizing the evidence.

=====================
CRITICAL RULE - DO NOT ALTER THE CLAIM
=====================

The claim_original MUST be EXACTLY what the user provided - character for character.
The claim_normalized_si MUST preserve the EXACT SAME MEANING as the original.
The claim_normalized_en is a DIRECT TRANSLATION - do NOT change the meaning, context, or any facts.

YOU ARE FORBIDDEN FROM:
- Changing names, places, or events mentioned in the claim
- Simplifying or summarizing the claim
- Adding or removing any information
- Changing the meaning in any way

If the claim is about "Mihintale police arrested 3 suspects for elephant burning in Seeppukulama", 
you CANNOT change it to something like "elephant is the largest animal".

=====================
1. CLAIM HANDLING
=====================

1.1 Keep claim_original EXACTLY as provided by the user (copy-paste, no changes).

1.2 For claim_normalized_si:
- If already in Sinhala: keep it exactly the same
- If in English/Singlish: translate to Sinhala preserving ALL details

1.3 For claim_normalized_en:
- Translate to English preserving ALL details (names, places, dates, events)
- Do NOT simplify or summarize

1.4 detection_notes: Describe the SPECIFIC factual proposition (who, what, where, when).

=====================
2. EVIDENCE COLLECTION
=====================

Search BOTH:
a) Sinhala/Local news (use queries in Sinhala AND transliterated names, site:.lk domains)
b) International/Official sources (use English queries with proper names)

For each result, extract 1-3 short snippets (1-3 sentences) most relevant to the claim.

=====================
3. EVIDENCE LABELING
=====================

For each snippet:
- relation: "SUPPORTS" | "REFUTES" | "IRRELEVANT"
- lang: "si" | "en"
- source_type: "official" | "intl_mainstream" | "local_mainstream" | "local_other" | "other"
- credibility_hint: "high" | "medium" | "low"

=====================
4. OUTPUT FORMAT (STRICT)
=====================

Return ONLY this JSON structure, nothing else:

{
  "claim_original": "<EXACT user-provided claim, no changes>",
  "claim_language_guess": "si" | "en" | "singlish" | "mixed",
  "claim_normalized_si": "<Sinhala version, same meaning as original>",
  "claim_normalized_en": "<English translation, same meaning as original>",
  "detection_notes": "<SPECIFIC description: who did what, where, when>",
  "cutoff_time_utc": "<current UTC time ISO 8601>",
  "evidence": [
    {
      "id": 1,
      "relation": "SUPPORTS" | "REFUTES" | "IRRELEVANT",
      "credibility_hint": "high" | "medium" | "low",
      "lang": "si" | "en",
      "source_type": "official" | "intl_mainstream" | "local_mainstream" | "local_other" | "other",
      "outlet": "<site name>",
      "title": "<page title>",
      "date": "<publication date or empty>",
      "url": "<full URL>",
      "snippet": "<1-3 relevant sentences>"
    }
  ]
}

Include at least 4-5 evidence items if possible.
Do NOT give any verdict. Just fill JSON."""

    USER_PROMPT_TEMPLATE = """Here is a claim I want to fact-check. Do NOT give me a verdict or explanation.
Use Search and Visit tools to gather evidence from both Sinhala/local sources and international/official sources.
Then return ONLY the JSON object as specified in your instructions.

Claim:
"{claim}"
"""

    def __init__(self):
        """Initialize the Research Agent."""
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "alibaba/tongyi-deepresearch-30b-a3b"
        print("[ResearchAgent] Initialized with DeepResearch model")
    
    def research(self, claim: str, api_key: str = None) -> Optional[Dict]:
        """
        Execute research on the claim and return structured evidence JSON.
        
        Args:
            claim: The raw claim text (any language)
            
        Returns:
            Dict with evidence structure, or None if failed
        """
        print(f"[ResearchAgent] Starting research for: {claim[:80]}...")
        
        # Use passed API key or fallback to env var
        current_api_key = api_key if api_key else self.api_key
        
        if not current_api_key:
            print("[ResearchAgent] No API key, cannot perform research")
            return self._create_empty_result(claim)
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(claim=claim)
        
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
            "max_tokens": 4000,
            "temperature": 0.1
        }
        
        try:
            print("[ResearchAgent] Calling DeepResearch API...")
            response = requests.post(
                self.endpoint, 
                headers=headers, 
                json=payload, 
                timeout=120  # Research takes time
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                print("[ResearchAgent] Response received, parsing JSON...")
                return self._parse_response(content, claim)
            else:
                print(f"[ResearchAgent] API error: {response.status_code}")
                print(f"[ResearchAgent] Response: {response.text[:500]}")
                return self._create_empty_result(claim)
                
        except Exception as e:
            print(f"[ResearchAgent] Error: {e}")
            return self._create_empty_result(claim)
    
    def _parse_response(self, content: str, original_claim: str) -> Dict:
        """Parse the LLM response into structured JSON."""
        try:
            # Clean up response
            content = content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            
            # Find JSON object
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            result = json.loads(content)
            print(f"[ResearchAgent] Parsed {len(result.get('evidence', []))} evidence items")
            return result
            
        except Exception as e:
            print(f"[ResearchAgent] Parse error: {e}")
            # Return what we got as a fallback
            return {
                "claim_original": original_claim,
                "claim_language_guess": "unknown",
                "claim_normalized_si": "",
                "claim_normalized_en": original_claim,
                "detection_notes": "Failed to parse research response",
                "cutoff_time_utc": datetime.now(timezone.utc).isoformat(),
                "evidence": [],
                "raw_response": content[:1000] if content else ""
            }
    
    def _create_empty_result(self, claim: str) -> Dict:
        """Create an empty result structure when research fails."""
        return {
            "claim_original": claim,
            "claim_language_guess": "unknown",
            "claim_normalized_si": "",
            "claim_normalized_en": claim,
            "detection_notes": "Research could not be performed",
            "cutoff_time_utc": datetime.now(timezone.utc).isoformat(),
            "evidence": []
        }


# Singleton instance
_research_agent = None

def get_research_agent() -> ResearchAgent:
    """Get or create the Research Agent singleton."""
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent
