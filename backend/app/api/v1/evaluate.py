"""
Evaluation API Endpoint.

Provides endpoints to run and retrieve evaluation results.
Uses the new Temporal-Hybrid Verifier for predictions.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import os
from datetime import datetime

from ...evaluation.metrics import EvaluationMetrics
from ...evaluation.benchmark import Benchmark
from ...agents.hybrid_verifier import get_hybrid_verifier

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# Store for evaluation results
_evaluation_results = {}
_evaluation_status = {"running": False, "last_run": None}


class EvaluationRequest(BaseModel):
    """Request to run evaluation."""
    test_samples: Optional[List[Dict]] = None  # Custom test data
    limit: int = 50
    use_dataset: bool = True


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
        raise HTTPException(status_code=404, detail="No evaluation results available")
    return _evaluation_results


@router.post("/quick-test")
async def quick_test(request: QuickTestRequest):
    """
    Run a quick test using Hybrid Verifier.
    
    Each sample should have:
    - text: The claim text
    - expected_label: The expected verdict (true/false/misleading)
    """
    metrics = EvaluationMetrics()
    results = []
    
    # Get hybrid verifier
    verifier = get_hybrid_verifier()
    
    for sample in request.samples:
        text = sample.get('text', '')
        expected = sample.get('expected_label', 'unknown')
        
        try:
            # Run hybrid verification
            result = verifier.verify(text)
            verdict = result.get('verdict', {})
            
            predicted = verdict.get('label', 'unknown')
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
    """Get sample test data for quick-test endpoint."""
    return {
        "description": "Sample test data for evaluation",
        "samples": [
            {
                "text": "ශ්‍රී ලංකාවේ ජනගහනය මිලියන 22 කි",
                "expected_label": "true",
                "description": "True fact about Sri Lanka population"
            },
            {
                "text": "අද රට පුරා ඇදිරි නීතිය පනවා ඇත",
                "expected_label": "false",
                "description": "False claim about curfew"
            },
            {
                "text": "පාර්ලිමේන්තුව විසුරුවා හැර ඇත",
                "expected_label": "needs_verification",
                "description": "Claim needing verification"
            }
        ]
    }


@router.get("/health")
async def evaluate_health():
    """Health check for evaluation service."""
    return {
        "status": "healthy",
        "verifier": "HybridVerifier",
        "metrics_available": True
    }
