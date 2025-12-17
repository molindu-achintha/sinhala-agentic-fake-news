"""
cot_reasoner.py

Chain of Thought Reasoning Agent.
Uses LLM with structured prompting for verification reasoning.
"""
import requests
from typing import Dict, List, Optional

from ..config import get_settings


class CoTReasoner:
    """
    Chain of Thought reasoning using LLM.
    Uses few shot examples from labeled data.
    """
    
    # Verification prompt template
    PROMPT_TEMPLATE = '''You are a SKEPTICAL Fact Checking Agent for Sinhala news.
Your job is to verify a news claim based ONLY on the provided evidence.
BE SKEPTICAL. Default to "Needs Verification" unless you have STRONG evidence.

CLAIM TO VERIFY:
{claim}

LABELED EVIDENCE (Verified sources with labels):
{labeled_history}

UNLABELED CONTEXT (Related but unverified):
{unlabeled_context}

CROSS EXAMINATION RESULTS:
Weighted Score: {weighted_score} (range: -1 to 1, negative means FALSE, positive means TRUE)
Source Priority: {source_priority}
Consensus: {consensus}
Zombie Rumor Check: {zombie_check}

CRITICAL RULES:
1. If weighted_score is between -0.5 and 0.5, return "Needs Verification"
2. If no labeled evidence matches the claim, return "Unverified"
3. Only return "True" if weighted_score >= 0.7 AND multiple sources agree
4. Only return "False" if weighted_score <= -0.7 AND multiple sources agree
5. A claim being similar to TRUE news does NOT make it true

OUTPUT FORMAT (use exactly this format):

VERDICT: [True / False / Misleading / Needs Verification / Unverified]
CONFIDENCE: [0 to 100] percent
REASONING: [2 to 3 sentences explaining your decision]
CITATIONS: [List the sources you used]

Now analyze:'''

    # Few shot example template
    FEW_SHOT_TEMPLATE = '''
Example {i}:
Claim: "{claim}"
Evidence: {evidence}
Verdict: {label}
'''

    def __init__(self):
        """Initialize reasoner with LLM settings."""
        settings = get_settings()
        self.api_key = settings.OPENROUTER_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.2-3b-instruct:free"
        
        self.headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sinhala Fake News Verifier"
        }
        
        print("[CoTReasoner] Initialized with model:", self.model)
    
    def reason(
        self, 
        claim: str, 
        evidence: Dict,
        cross_exam: Dict,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Perform Chain of Thought reasoning.
        
        Args:
            claim: The claim to verify
            evidence: Output from HybridRetriever
            cross_exam: Output from CrossExaminer
            few_shot_examples: Optional examples from labeled data
        
        Returns:
            Dict with verdict confidence reasoning citations
        """
        print("[CoTReasoner] Starting CoT reasoning")
        
        # Build prompt
        prompt = self._build_prompt(claim, evidence, cross_exam, few_shot_examples)
        
        # Call LLM
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
        except Exception as e:
            print("[CoTReasoner] LLM error:", str(e))
            result = self._fallback_reasoning(cross_exam)
        
        print("[CoTReasoner] Verdict:", result.get('verdict'))
        print("[CoTReasoner] Confidence:", result.get('confidence'))
        
        return result
    
    def _build_prompt(
        self, 
        claim: str, 
        evidence: Dict,
        cross_exam: Dict,
        few_shot_examples: Optional[List[Dict]]
    ) -> str:
        """Build the full prompt with few shot examples."""
        # Format labeled history
        labeled = evidence.get("labeled_history", [])
        labeled_text = self._format_evidence(labeled, include_label=True)
        
        # Format unlabeled context
        unlabeled = evidence.get("unlabeled_context", [])
        unlabeled_text = self._format_evidence(unlabeled, include_label=False)
        
        # Add few shot examples if provided
        few_shot = ""
        if few_shot_examples:
            for i, ex in enumerate(few_shot_examples[:3], 1):
                few_shot += self.FEW_SHOT_TEMPLATE.format(
                    i=i,
                    claim=ex.get("claim", "")[:100],
                    evidence=ex.get("evidence", "")[:100],
                    label=ex.get("label", "")
                )
        
        # Build main prompt
        prompt = self.PROMPT_TEMPLATE.format(
            claim=claim,
            labeled_history=labeled_text or "No labeled evidence found.",
            unlabeled_context=unlabeled_text or "No additional context.",
            weighted_score=str(round(cross_exam.get('weighted_score', 0), 2)),
            source_priority=cross_exam.get("source_priority", "unknown"),
            consensus=cross_exam.get("consensus", {}).get("message", "Unknown"),
            zombie_check=cross_exam.get("zombie_check", {}).get("message", "Not detected")
        )
        
        if few_shot:
            prompt = "EXAMPLES:\n" + few_shot + "\n\n" + prompt
        
        return prompt
    
    def _format_evidence(self, docs: List[Dict], include_label: bool) -> str:
        """Format evidence documents for prompt."""
        if not docs:
            return ""
        
        lines = []
        for i, doc in enumerate(docs[:5], 1):
            text = doc.get("text", "")[:200]
            source = doc.get("source", "unknown")
            score = doc.get("score", 0)
            
            line = str(i) + ". [" + source + "] (similarity: " + str(round(score * 100)) + " percent): " + text
            if include_label:
                label = doc.get("label", "unknown")
                line += " [LABEL: " + label.upper() + "]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        print("[CoTReasoner] Calling LLM")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception("LLM API error: " + str(response.status_code))
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured output."""
        lines = response.strip().split("\n")
        
        verdict = "unverified"
        confidence = 50
        reasoning = ""
        citations = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith("verdict:"):
                verdict_text = line.split(":", 1)[1].strip().lower()
                verdict = self._normalize_verdict(verdict_text)
            
            elif line_lower.startswith("confidence:"):
                conf_text = line.split(":", 1)[1].strip()
                try:
                    confidence = int(conf_text.replace("%", "").replace("percent", "").strip())
                except:
                    confidence = 50
            
            elif line_lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            
            elif line_lower.startswith("citations:"):
                citations_text = line.split(":", 1)[1].strip()
                citations = [c.strip() for c in citations_text.split(",")]
        
        return {
            "verdict": verdict,
            "confidence": confidence / 100,
            "reasoning": reasoning,
            "citations": citations,
            "raw_response": response
        }
    
    def _normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict to standard labels."""
        verdict = verdict.lower().strip()
        
        if "true" in verdict and "false" not in verdict:
            return "true"
        elif "false" in verdict:
            return "false"
        elif "misleading" in verdict or "partial" in verdict:
            return "misleading"
        elif "needs" in verdict or "verification" in verdict:
            return "needs_verification"
        else:
            return "unverified"
    
    def _fallback_reasoning(self, cross_exam: Dict) -> Dict:
        """Fallback when LLM fails use cross exam results."""
        recommendation = cross_exam.get("recommendation", "unverified")
        confidence = cross_exam.get("confidence", 0.5)
        
        return {
            "verdict": recommendation,
            "confidence": confidence,
            "reasoning": "LLM unavailable verdict based on cross examination.",
            "citations": [],
            "fallback": True
        }
