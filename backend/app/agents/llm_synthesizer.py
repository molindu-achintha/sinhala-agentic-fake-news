"""
llm_synthesizer.py

LLM integration for generating fact-check reports.
Supports Groq and OpenRouter APIs.
"""
import os
import requests
from typing import Dict, List, Optional


class LLMSynthesizer:
    """
    Generates fact-check reports using LLMs.
    Supports Groq and OpenRouter.
    """
    
    def __init__(self, provider: str = "groq"):
        """
        Initialize the synthesizer.
        provider: 'groq' or 'openrouter'
        """
        self.provider = provider.lower()
        print(f"[LLMSynthesizer] Using provider: {self.provider}")
        
        # API endpoints
        self.endpoints = {
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions"
        }
        
        # Default models
        self.models = {
            "groq": "llama-3.1-70b-versatile",
            "openrouter": "mistralai/mistral-7b-instruct"
        }
        
        # Get API key
        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY", "")
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            
        if not self.api_key:
            print(f"[LLMSynthesizer] Warning: No API key for {self.provider}")
    
    def generate_report(
        self,
        claim: str,
        evidence: List[Dict],
        translated_claim: str = ""
    ) -> Dict:
        """
        Generate a fact-check report from evidence.
        Returns dict with 'report' and 'verdict'.
        """
        print(f"[LLMSynthesizer] Generating report for: {claim[:50]}...")
        
        # Build the prompt
        prompt = self._build_prompt(claim, evidence, translated_claim)
        
        # Call the LLM
        response = self._call_llm(prompt)
        
        if response:
            print("[LLMSynthesizer] Report generated successfully")
            return {
                "report": response,
                "provider": self.provider,
                "model": self.models.get(self.provider, "unknown")
            }
        else:
            print("[LLMSynthesizer] Failed to generate report")
            return {
                "report": "Unable to generate report. Please check API keys.",
                "provider": self.provider,
                "model": "error"
            }
    
    def _build_prompt(
        self,
        claim: str,
        evidence: List[Dict],
        translated_claim: str
    ) -> str:
        """Build the fact-check prompt."""
        
        # Format evidence
        evidence_text = ""
        for i, e in enumerate(evidence, 1):
            source = e.get("source_url", "Unknown")
            title = e.get("title", "No title")
            snippet = e.get("content_snippet", "")[:300]
            stance = e.get("stance", "neutral")
            evidence_text += f"{i}. [{title}]({source})\n   Stance: {stance}\n   Content: {snippet}\n\n"
        
        prompt = f"""You are a fact-checker. Analyze the claim below using the provided evidence.

CLAIM (Sinhala): {claim}
CLAIM (English): {translated_claim}

EVIDENCE FROM WEB SEARCH:
{evidence_text if evidence_text else "No evidence found."}

Generate a fact-check report in this format:

## ✅ Verdict: [TRUE / FALSE / MISLEADING / UNVERIFIED]

### 🔎 What the claim says
[Brief summary of what is being claimed]

### 📌 What we found
[List each source with a brief summary]
- **Source 1**: Summary
- **Source 2**: Summary

### 🧠 Analysis
[Explain why the claim is true/false/misleading. Be specific.]

### 📊 Confidence: [HIGH / MEDIUM / LOW]

Keep the response concise but informative. Use emojis for formatting."""

        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM API."""
        
        if not self.api_key:
            print("[LLMSynthesizer] No API key available")
            return None
        
        endpoint = self.endpoints.get(self.provider)
        model = self.models.get(self.provider)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenRouter needs extra headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://sinhala-fake-news-detector.com"
            headers["X-Title"] = "Sinhala Fake News Detector"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            print(f"[LLMSynthesizer] Calling {self.provider} API...")
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                print(f"[LLMSynthesizer] API error: {response.status_code}")
                print(f"[LLMSynthesizer] Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"[LLMSynthesizer] Error: {e}")
            return None


# Singleton
_synthesizer = None

def get_llm_synthesizer(provider: str = "groq") -> LLMSynthesizer:
    """Get or create LLM synthesizer."""
    global _synthesizer
    # Always create new if provider changes
    if _synthesizer is None or _synthesizer.provider != provider.lower():
        _synthesizer = LLMSynthesizer(provider)
    return _synthesizer
