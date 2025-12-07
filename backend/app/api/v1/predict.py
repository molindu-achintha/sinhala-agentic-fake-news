"""
Prediction endpoint.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from ...agents.claim_extractor import ClaimExtractorAgent
from ...agents.langproc_agent import LangProcAgent
from ...agents.retrieval_agent import RetrievalAgent
from ...agents.reasoning_agent import ReasoningAgent
from ...agents.verdict_agent import VerdictAgent

router = APIRouter()

# Instantiate agents (Singleton pattern or dependency injection preferred)
lang_proc = LangProcAgent()
claim_extractor = ClaimExtractorAgent()
retrieval_agent = RetrievalAgent(lang_proc)
reasoning_agent = ReasoningAgent()
verdict_agent = VerdictAgent()

class PredictRequest(BaseModel):
    text: str
    source: Optional[str] = None
    top_k: int = 5

class PredictResponse(BaseModel):
    claim: dict
    retrieved_evidence: List[dict]
    reasoning: dict
    verdict: dict

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # 1. Extract Claim
    claim = claim_extractor.extract_claim(request.text)
    
    # 2. Retrieve Evidence
    evidence = retrieval_agent.retrieve_evidence(claim['claim_text'], top_k=request.top_k)
    
    # 3. Reason
    reasoning = reasoning_agent.reason(claim['claim_text'], evidence)
    
    # 4. Verdict
    verdict = verdict_agent.generate_verdict(claim, reasoning, evidence)
    
    return PredictResponse(
        claim=claim,
        retrieved_evidence=evidence,
        reasoning=reasoning,
        verdict=verdict
    )
