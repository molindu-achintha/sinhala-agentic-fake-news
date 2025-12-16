"""
Evaluation API Endpoint.

Provides endpoints to run and retrieve evaluation results.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import os
from datetime import datetime

from ..evaluation.metrics import EvaluationMetrics
from ..evaluation.benchmark import Benchmark

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# Store for evaluation results
_evaluation_results = {}
_evaluation_status = {"running": False, "last_run": None}


class EvaluationRequest(BaseModel):
    """Request to run evaluation."""
    test_samples: Optional[List[Dict]] = None  # Custom test data
    limit: int = 50
    use_dataset: bool = True  # Use preprocessed.jsonl


class QuickTestRequest(BaseModel):
    """Quick test with known labels."""
    samples: List[Dict]  # List of {"text": "...", "expected_label": "true/false/..."}


@router.get("/status")
async def get_evaluation_status():
    """Get current evaluation status."""
    return {
        "is_running": _evaluation_status["running"],
        "last_run": _evaluation_status["last_run"],
        "results_available": bool(_evaluation_results)
    }


@router.get("/results")
async def get_evaluation_results():
    """Get the most recent evaluation results."""
    if not _evaluation_results:
        raise HTTPException(status_code=404, detail="No evaluation results available. Run an evaluation first.")
    return _evaluation_results


@router.post("/quick-test")
async def quick_test(request: QuickTestRequest):
    """
    Run a quick test with provided samples.
    
    Each sample should have:
    - text: The claim text
    - expected_label: The expected verdict (true/false/misleading)
    """
    from ..agents.claim_extractor import ClaimExtractorAgent
    from ..agents.langproc_agent import LangProcAgent
    from ..agents.retrieval_agent import RetrievalAgent
    from ..agents.reasoning_agent import ReasoningAgent
    from ..agents.verdict_agent import VerdictAgent
    
    metrics = EvaluationMetrics()
    results = []
    
    # Initialize agents
    claim_extractor = ClaimExtractorAgent()
    lang_proc = LangProcAgent()
    retrieval = RetrievalAgent()
    reasoning = ReasoningAgent()
    verdict_agent = VerdictAgent()
    
    for sample in request.samples:
        text = sample.get('text', '')
        expected = sample.get('expected_label', 'unknown')
        
        try:
            # Run prediction pipeline
            claim = claim_extractor.extract(text)
            embedding = lang_proc.get_embeddings(claim['claim_text'])
            evidence = retrieval.retrieve(embedding, top_k=5)
            reasoning_result = reasoning.reason(claim['claim_text'], evidence)
            verdict = verdict_agent.generate_verdict(claim, reasoning_result, evidence)
            
            predicted = verdict['label']
            is_correct = metrics.LABEL_MAPPING.get(predicted.lower()) == metrics.LABEL_MAPPING.get(expected.lower())
            
            metrics.add_result(predicted, expected)
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'confidence': verdict.get('confidence', 0)
            })
            
        except Exception as e:
            results.append({
                'text': text[:100],
                'expected': expected,
                'predicted': 'error',
                'correct': False,
                'error': str(e)
            })
    
    report = metrics.classification_report()
    
    return {
        'summary': {
            'total_samples': len(request.samples),
            'accuracy': report['accuracy'],
            'correct': sum(1 for r in results if r.get('correct')),
            'incorrect': sum(1 for r in results if not r.get('correct'))
        },
        'metrics': {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro_f1'],
            'per_class': report['per_class']
        },
        'results': results
    }


@router.get("/sample-test-data")
async def get_sample_test_data():
    """
    Get sample test data that can be used with quick-test endpoint.
    """
    return {
        "description": "Sample test data for evaluation",
        "samples": [
            {
                "text": "ශ්‍රී ලංකාවේ ජනගහනය මිලියන 22 කි",
                "expected_label": "true",
                "note": "Verifiable fact about Sri Lanka's population"
            },
            {
                "text": "හෙට සිට පෙට්‍රල් මිල රුපියල් 500 දක්වා ඉහළ යයි",
                "expected_label": "false",
                "note": "Likely false claim about extreme price increase"
            },
            {
                "text": "කොළඹ දිසා ලේම්‌ස්ට් ගොස් ජනතාව 100 ක් අත් අඩන්ගුවේ",
                "expected_label": "needs_verification",
                "note": "Unverified news claim"
            }
        ],
        "usage": "POST /v1/evaluate/quick-test with {samples: [...]}"
    }


@router.get("/metrics-info")
async def get_metrics_info():
    """Get information about available metrics."""
    return {
        "classification_metrics": {
            "accuracy": "Proportion of correct predictions",
            "precision": "TP / (TP + FP) - How many positive predictions were correct",
            "recall": "TP / (TP + FN) - How many actual positives were found",
            "f1_score": "Harmonic mean of precision and recall",
            "macro_f1": "Average F1 across all classes",
            "weighted_f1": "F1 weighted by class support"
        },
        "benchmark_metrics": {
            "latency_ms": "Time to process single request",
            "p95_latency_ms": "95th percentile latency",
            "throughput_rps": "Requests processed per second",
            "success_rate": "Percentage of successful requests"
        },
        "labels": ["true", "false", "misleading", "needs_verification"]
    }
