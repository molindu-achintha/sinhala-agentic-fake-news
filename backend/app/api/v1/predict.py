"""
predict.py - Prediction API Endpoint

This module provides the main prediction endpoint for fake news detection.
It orchestrates the multi-agent pipeline:
1. Claim Extraction - Extract verifiable claim from input
2. Embedding Generation - Convert claim to vector
3. Evidence Retrieval - Search Pinecone for similar content
4. Reasoning - Analyze evidence and determine verdict
5. Verdict Generation - Produce final result with explanations
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

# Initialize agents (singleton pattern)
print("[predict] Initializing agents")
lang_proc = LangProcAgent()
claim_extractor = ClaimExtractorAgent()
reasoning_agent = ReasoningAgent()
verdict_agent = VerdictAgent()
print("[predict] All agents initialized")


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str
    source: Optional[str] = None
    top_k: int = 5
    use_pinecone: bool = True


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    claim: dict
    retrieved_evidence: List[dict]
    reasoning: dict
    verdict: dict


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Verify a claim using the multi-agent pipeline.
    
    This endpoint:
    1. Extracts claim from input text
    2. Generates embedding for claim  
    3. Searches Pinecone for similar documents
    4. Reasons about the claim with evidence
    5. Returns verdict with explanations
    
    Args:
        request: PredictRequest with text to verify
        
    Returns:
        PredictResponse with claim, evidence, reasoning, verdict
    """
    print("=" * 50)
    print("[predict] New prediction request received")
    print("[predict] Text length:", len(request.text))
    
    settings = get_settings()
    
    # Step 1: Extract Claim
    print("[predict] Step 1: Extracting claim")
    claim = claim_extractor.extract_claim(request.text)
    claim_text = claim['claim_text']
    print("[predict] Claim extracted:", claim_text[:50], "...")
    
    # Step 2: Generate embedding for claim
    print("[predict] Step 2: Generating embedding")
    from ...utils.text_normalize import normalize_text
    normalized_claim = normalize_text(claim_text)
    claim_embedding = lang_proc.get_embeddings(normalized_claim)
    print("[predict] Embedding dimension:", len(claim_embedding))
    
    evidence = []
    
    # Step 3: Search Pinecone for evidence
    print("[predict] Step 3: Searching for evidence")
    if request.use_pinecone and settings.PINECONE_API_KEY:
        try:
            from ...store.pinecone_store import PineconeVectorStore
            
            pinecone_store = PineconeVectorStore(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME
            )
            
            # Search dataset namespace (historical labeled data)
            print("[predict] Searching dataset namespace")
            dataset_results = pinecone_store.search(
                query_embedding=claim_embedding.tolist(),
                top_k=request.top_k,
                namespace="dataset"
            )
            print("[predict] Dataset results:", len(dataset_results))
            
            # Search live_news namespace (recent scraped news)
            print("[predict] Searching live_news namespace")
            news_results = pinecone_store.search(
                query_embedding=claim_embedding.tolist(),
                top_k=request.top_k,
                namespace="live_news"
            )
            print("[predict] Live news results:", len(news_results))
            
            # Combine and sort by score
            all_results = dataset_results + news_results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top_k results
            evidence = all_results[:request.top_k]
            print("[predict] Total evidence after combining:", len(evidence))
            
        except Exception as e:
            print("[predict] Pinecone search failed:", str(e))
            evidence = []
    else:
        print("[predict] Pinecone not configured, skipping search")
    
    # Step 4: Reason about the claim
    print("[predict] Step 4: Reasoning about claim")
    reasoning = reasoning_agent.reason(claim_text, evidence)
    print("[predict] Reasoning complete, recommendation:", reasoning.get('verdict_recommendation'))
    
    # Step 5: Generate verdict
    print("[predict] Step 5: Generating verdict")
    verdict = verdict_agent.generate_verdict(claim, reasoning, evidence)
    print("[predict] Final verdict:", verdict.get('label'))
    print("[predict] Confidence:", verdict.get('confidence'))
    print("=" * 50)
    
    return PredictResponse(
        claim=claim,
        retrieved_evidence=evidence,
        reasoning=reasoning,
        verdict=verdict
    )


@router.get("/predict/stats")
async def get_pinecone_stats():
    """
    Get Pinecone index statistics.
    
    Returns information about the vector database including
    total vectors and namespace breakdown.
    """
    print("[predict] Stats endpoint called")
    settings = get_settings()
    
    if not settings.PINECONE_API_KEY:
        print("[predict] PINECONE_API_KEY not set")
        return {"error": "PINECONE_API_KEY not set"}
    
    try:
        from ...store.pinecone_store import PineconeVectorStore
        
        pinecone_store = PineconeVectorStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME
        )
        
        stats = pinecone_store.get_stats()
        print("[predict] Stats retrieved:", stats)
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        print("[predict] Error getting stats:", str(e))
        return {"error": str(e)}
