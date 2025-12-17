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

⚠️ CRITICAL FIRST STEP - TOPIC RELEVANCE CHECK:
Before trusting ANY evidence, you MUST check if it is about the SAME TOPIC as the claim.

For example:
- Claim about "Cricket match stopped" → Evidence about "Train service stopped" = NOT RELEVANT (different topic)
- Claim about "Fuel prices" → Evidence about "Electricity prices" = NOT RELEVANT (different topic)
- Claim about "President visited India" → Evidence about "President visited India" = RELEVANT (same topic)

If evidence is about a DIFFERENT TOPIC, you MUST:
1. IGNORE that evidence completely
2. Mark the claim as "Unverified" due to lack of relevant evidence

CLAIM TO VERIFY:
{claim}

LABELED EVIDENCE (Check if these are actually about the SAME TOPIC as the claim):
{labeled_history}

UNLABELED CONTEXT:
{unlabeled_context}

CROSS EXAMINATION RESULTS:
Weighted Score: {weighted_score}
Source Priority: {source_priority}
Consensus: {consensus}
Zombie Rumor Check: {zombie_check}

VERIFICATION CHECKLIST (You must perform these checks):

1. TOPIC CHECK: 
   - Is evidence about the SAME event? (e.g. Cricket != Trains)
   - If NO, stop and return "Unverified".

2. DATE CHECK (Zombie Rumor Detection):
   - Check the dates in the evidence vs the claim.
   - Is the claim recycling an old event as "new"?
   - If evidence is from 2019 but claim says "today", it is MISLEADING/FALSE.

3. SOURCE CHECK:
   - Are the sources reputable news outlets (e.g. Adaderana, BBC, Lankadeepa)?
   - Or are they random social media posts?
   - Give less weight to random sources.

4. LOGIC CHECK:
   - Does the evidence *actually* prove the claim, or just mention similar keywords?
   - "Government discussing fuel" does NOT prove "Fuel price increased".

VERIFICATION STEPS:
1. Run Topic Check. If fail -> Unverified.
2. Run Date Check. If fail -> Misleading/False.
3. Check if reliable sources confirm the specific details.
4. If sources conflict -> Needs Verification.
5. If score is high but topic mismatch -> Unverified.

OUTPUT FORMAT:

TOPIC_MATCH: [Yes / No] - Are the evidence pieces about the same topic as the claim?
VERDICT: [True / False / Misleading / Needs Verification / Unverified]
CONFIDENCE: [0 to 100] percent
REASONING: [Explain your topic relevance check and decision]
CITATIONS: [List sources, or "None relevant" if topic mismatch]

Analyze now:'''

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
        
        # Special instruction for Web Search Fallback
        recommendation = cross_exam.get("recommendation")
        if recommendation == "check_web":
            prompt += """\n
IMPORTANT INSTRUCTION:
No pre-verified labeled evidence was found database. 
However, WEB SEARCH results are available in 'UNLABELED CONTEXT'.
Please verify the claim using ONLY the information in 'UNLABELED CONTEXT'.
If the web search results clearly confirm or debunk the claim, issue a verdict based on that.
IGNORE the "Labels" section as it is empty. Focus on content analysis of the web results.
"""
        
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
        topic_match = True  # Default to true
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith("topic_match:"):
                topic_text = line.split(":", 1)[1].strip().lower()
                topic_match = "yes" in topic_text
            
            elif line_lower.startswith("verdict:"):
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
        
        # CRITICAL: If topic doesn't match, force unverified
        if not topic_match:
            print("[CoTReasoner] Topic mismatch detected - forcing unverified")
            verdict = "unverified"
            confidence = 30
            reasoning = "Evidence is about a different topic than the claim. " + reasoning
        
        return {
            "verdict": verdict,
            "confidence": confidence / 100,
            "reasoning": reasoning,
            "citations": citations,
            "topic_match": topic_match,
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
