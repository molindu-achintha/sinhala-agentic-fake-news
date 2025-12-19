"""
predict.py - Prediction API Endpoint

Uses the Temporal-Hybrid Verifier for fake news detection.
Pipeline:
1. Claim Decomposition - Extract keywords, dates, entities
2. Hybrid Retrieval - Search Vector DB (labeled + unlabeled)
3. Cross-Examination - Weight evidence, check consensus
4. CoT Reasoning - LLM with few-shot examples
5. Verdict Generation - Final result with citations
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ...agents.hybrid_verifier import get_hybrid_verifier
from ...config import get_settings

router = APIRouter()

# Initialize hybrid verifier
print("[predict] Prediction API ready")


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str
    source: Optional[str] = None
    top_k: int = 10
    llm_provider: str = "groq"  # 'groq' or 'openrouter'
    use_vector_db: bool = True  # Toggle for using Vector DB (Knowledge Base)
    openrouter_api_key: Optional[str] = None  # Dynamic API Key from Frontend


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    claim: dict
    evidence: dict
    cross_examination: dict
    reasoning: dict
    verdict: dict


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Verify a claim using the Temporal-Hybrid Verifier.
    
    This endpoint:
    1. Decomposes claim (keywords, dates, temporal type)
    2. Retrieves evidence from labeled DB and live news
    3. Cross-examines evidence (consensus, conflicts, zombie rumors)
    4. Performs Chain of Thought reasoning with LLM
    5. Returns verdict with citations
    
    Args:
        request: PredictRequest with text to verify
        
    Returns:
        PredictResponse with full verification result
    """
    print("[predict] New verification request")
    print("[predict] Text length:", len(request.text))
    
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400, 
            detail="Text too short for verification"
        )
    
    try:
        # Get hybrid verifier
        verifier = get_hybrid_verifier()
        
        # Run full verification pipeline
        print(f"[predict] Using LLM provider: {request.llm_provider}")
        print(f"[predict] Use Vector DB: {request.use_vector_db}")
        result = verifier.verify(
            request.text, 
            llm_provider=request.llm_provider,
            use_vector_db=request.use_vector_db,
            openrouter_api_key=request.openrouter_api_key
        )
        
        print("[predict] Verification complete")
        
        # Add LLM report if available
        llm_report = result.get("llm_report", "")
        
        # Handle cached results that may have different structure
        claim_data = result.get("claim", {
            "original": request.text,
            "translated": "",
            "normalized_si": "",
            "normalized_en": ""
        })
        evidence_data = result.get("evidence", {"labeled_count": 0, "top_similarity": 0})
        cross_exam_data = result.get("cross_examination", {})
        reasoning_data = result.get("reasoning", {})
        verdict_data = result.get("verdict", {"label": "unknown", "confidence": 0})
        
        # Handle case where fields might be strings (old cache format)
        if not isinstance(claim_data, dict):
            claim_data = {"original": request.text, "translated": "", "normalized_si": "", "normalized_en": ""}
        if not isinstance(evidence_data, dict):
            evidence_data = {"labeled_count": 0, "top_similarity": 0}
        if not isinstance(cross_exam_data, dict):
            cross_exam_data = {}
        if not isinstance(reasoning_data, dict):
            reasoning_data = {}
        if not isinstance(verdict_data, dict):
            verdict_data = {"label": str(verdict_data) if verdict_data else "unknown", "confidence": 0}
        
        return PredictResponse(
            claim=claim_data,
            evidence=evidence_data,
            cross_examination=cross_exam_data,
            reasoning=reasoning_data,
            verdict={**verdict_data, "llm_report": llm_report}
        )
        
    except Exception as e:
        print(f"[predict] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/simple")
async def predict_simple(request: PredictRequest):
    """
    Simplified prediction - returns just verdict and confidence.
    """
    result = await predict(request)
    
    return {
        "verdict": result.verdict.get("label", "unknown"),
        "confidence": result.verdict.get("confidence", 0),
        "explanation_si": result.verdict.get("explanation_si", ""),
        "explanation_en": result.verdict.get("explanation_en", "")
    }


@router.get("/predict/health")
async def predict_health():
    """Health check for prediction service."""
    try:
        verifier = get_hybrid_verifier()
        return {
            "status": "healthy",
            "verifier": "HybridVerifier",
            "agents": [
                "ClaimDecomposer",
                "HybridRetriever",
                "CrossExaminer",
                "CoTReasoner",
                "VerdictAgent"
            ]
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
