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
        """Build the fact-check prompt with ChatGPT-style formatting."""
        
        # Extract source names from URLs
        evidence_text = ""
        source_names = []
        for i, e in enumerate(evidence, 1):
            url = e.get("source_url", "")
            title = e.get("title", "No title")
            snippet = e.get("content_snippet", "")[:400]
            stance = e.get("stance", "neutral")
            
            # Extract source name from URL
            source_name = self._extract_source_name(url)
            source_names.append(source_name)
            
            evidence_text += f"""
Source {i}: {source_name}
URL: {url}
Title: {title}
Stance: {stance}
Content: {snippet}
"""
        
        prompt = f"""You are a professional fact-checker. Analyze this claim using the evidence provided.

CLAIM (Sinhala): {claim}
CLAIM (English): {translated_claim}

EVIDENCE COLLECTED FROM WEB SEARCH:
{evidence_text if evidence_text else "No evidence found from web search."}

Generate a detailed fact-check report EXACTLY in this format:

---

This claim is [TRUE/FALSE/MISLEADING/UNVERIFIED]. [Brief one-line summary of why]

✔ What Actually Happened

[Explain the verified facts with specific details from the sources]

✔ Confirmations from Multiple Outlets

[List each source with what they reported]:
- **{source_names[0] if source_names else 'Source'}**: [What this source says]
- **{source_names[1] if len(source_names) > 1 else 'Other Source'}**: [What this source says]
[Add more sources if available]

🧾 What the Statement Means

[Provide context - is this a political statement vs legal fact? Is there exaggeration? What's the nuance?]

📌 Key Points
- ❌ or ✅ [Key finding 1]
- ❌ or ✅ [Key finding 2]
- ❌ or ✅ [Key finding 3]

---

IMPORTANT RULES:
1. Use the ACTUAL source names from the evidence (like Daily Mirror, Hiru News, etc.)
2. Include URLs as citations where possible
3. Be specific about what each source says
4. If the claim is partially true, explain what part is true and what is false
5. Use emojis for visual structure
6. Keep it professional but readable
"""

        return prompt
    
    def _extract_source_name(self, url: str) -> str:
        """Extract readable source name from URL."""
        if not url:
            return "Unknown Source"
            
        # Common Sri Lankan news sources
        source_map = {
            "dailymirror": "Daily Mirror",
            "hirunews": "Hiru News",
            "adaderana": "Ada Derana",
            "newsfirst": "News First",
            "lankadeepa": "Lankadeepa",
            "divaina": "Divaina",
            "themorning": "The Morning",
            "island": "The Island",
            "sundaytimes": "Sunday Times",
            "bbc": "BBC",
            "reuters": "Reuters",
            "wikipedia": "Wikipedia",
            "economictimes": "Economic Times",
            "onlanka": "OnLanka",
            "newswire": "Newswire"
        }
        
        url_lower = url.lower()
        for key, name in source_map.items():
            if key in url_lower:
                return name
        
        # Try to extract domain
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Clean up domain
            domain = domain.replace("www.", "").split(".")[0]
            return domain.title()
        except:
            return "Web Source"
    
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
