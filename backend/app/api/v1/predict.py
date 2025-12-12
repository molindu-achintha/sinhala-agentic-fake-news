"""
Prediction endpoint with Pinecone Vector Search.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from ...agents.claim_extractor import ClaimExtractorAgent
from ...agents.langproc_agent import LangProcAgent
from ...agents.reasoning_agent import ReasoningAgent
from ...agents.verdict_agent import VerdictAgent
from ...config import get_settings

router = APIRouter()

# Instantiate agents
lang_proc = LangProcAgent()
claim_extractor = ClaimExtractorAgent()
reasoning_agent = ReasoningAgent()
verdict_agent = VerdictAgent()


class PredictRequest(BaseModel):
    text: str
    source: Optional[str] = None
    top_k: int = 5
    use_pinecone: bool = True  # Use Pinecone by default


class PredictResponse(BaseModel):
    claim: dict
    retrieved_evidence: List[dict]
    reasoning: dict
    verdict: dict


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Verify a claim using Pinecone vector search.
    
    1. Extracts claim from input text
    2. Generates embedding for claim
    3. Searches Pinecone for similar documents
    4. Reasons about the claim with evidence
    5. Returns verdict
    """
    settings = get_settings()
    
    # 1. Extract Claim
    claim = claim_extractor.extract_claim(request.text)
    claim_text = claim['claim_text']
    
    # 2. Generate claim embedding
    from ...utils.text_normalize import normalize_text
    normalized_claim = normalize_text(claim_text)
    claim_embedding = lang_proc.get_embeddings(normalized_claim)
    
    evidence = []
    
    # 3. Search Pinecone
    if request.use_pinecone and settings.PINECONE_API_KEY:
        try:
            from ...store.pinecone_store import PineconeVectorStore
            
            pinecone_store = PineconeVectorStore(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME
            )
            
            # Search in both namespaces
            # First search dataset
            dataset_results = pinecone_store.search(
                query_embedding=claim_embedding,
                top_k=request.top_k,
                namespace="dataset"
            )
            
            # Then search live news
            news_results = pinecone_store.search(
                query_embedding=claim_embedding,
                top_k=request.top_k,
                namespace="live_news"
            )
            
            # Combine and sort by score
            all_results = dataset_results + news_results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top_k
            evidence = all_results[:request.top_k]
            
        except Exception as e:
            print(f"Pinecone search failed: {e}")
            # Fall back to empty evidence
            evidence = []
    
    # 4. Reason about the claim
    reasoning = reasoning_agent.reason(claim_text, evidence)
    
    # 5. Generate verdict
    verdict = verdict_agent.generate_verdict(claim, reasoning, evidence)
    
    return PredictResponse(
        claim=claim,
        retrieved_evidence=evidence,
        reasoning=reasoning,
        verdict=verdict
    )


@router.get("/predict/stats")
async def get_pinecone_stats():
    """Get Pinecone index statistics."""
    settings = get_settings()
    
    if not settings.PINECONE_API_KEY:
        return {"error": "PINECONE_API_KEY not set"}
    
    try:
        from ...store.pinecone_store import PineconeVectorStore
        
        pinecone_store = PineconeVectorStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME
        )
        
        return {
            "success": True,
            "stats": pinecone_store.get_stats()
        }
    except Exception as e:
        return {"error": str(e)}
