"""
Evaluation Runner Script for Sinhala Fake News Detection System.

This script:
1. Loads test data with known labels
2. Runs predictions through the API
3. Calculates accuracy, precision, recall, F1
4. Measures latency and throughput
5. Generates a comprehensive report

Usage:
    python -m app.evaluation.run_evaluation [--dataset path] [--limit N]
"""
import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.benchmark import Benchmark
from app.agents.langproc_agent import LangProcAgent
from app.agents.claim_extractor import ClaimExtractorAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.verdict_agent import VerdictAgent


class EvaluationRunner:
    """
    Run evaluation and benchmarking for the fake news detection system.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics = EvaluationMetrics()
        self.benchmark = Benchmark()
        
        # Initialize agents for direct testing
        self.lang_proc = None
        self.claim_extractor = None
        self.retrieval = None
        self.reasoning = None
        self.verdict = None
    
    def _init_agents(self):
        """Initialize agents for direct evaluation."""
        if self.lang_proc is None:
            print("Initializing agents...")
            self.lang_proc = LangProcAgent()
            self.claim_extractor = ClaimExtractorAgent()
            self.retrieval = RetrievalAgent()
            self.reasoning = ReasoningAgent()
            self.verdict = VerdictAgent()
    
    def load_test_data(self, filepath: str, limit: int = None) -> list:
        """Load test data from JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                doc = json.loads(line)
                # Ensure we have text and label
                if doc.get('text') or doc.get('claim'):
                    data.append({
                        'text': doc.get('text') or doc.get('claim', ''),
                        'label': doc.get('label', 'unknown'),
                        'source': doc.get('source', 'dataset')
                    })
        return data
    
    def predict_via_api(self, text: str) -> dict:
        """Make prediction via API endpoint."""
        response = requests.post(
            f"{self.api_url}/v1/predict",
            json={"text": text, "top_k": 5},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def predict_direct(self, text: str) -> dict:
        """Make prediction directly using agents (no API)."""
        self._init_agents()
        
        # Step 1: Extract claim
        claim = self.claim_extractor.extract(text)
        
        # Step 2: Get embeddings
        embedding = self.lang_proc.get_embeddings(claim['claim_text'])
        
        # Step 3: Retrieve evidence
        evidence = self.retrieval.retrieve(embedding, top_k=5)
        
        # Step 4: Reasoning
        reasoning = self.reasoning.reason(claim['claim_text'], evidence)
        
        # Step 5: Verdict
        verdict = self.verdict.generate_verdict(claim, reasoning, evidence)
        
        return {
            'claim': claim,
            'evidence': evidence,
            'reasoning': reasoning,
            'verdict': verdict
        }
    
    def run_accuracy_evaluation(
        self, 
        test_data: list, 
        use_api: bool = True,
        verbose: bool = False
    ):
        """
        Run accuracy evaluation on test data.
        
        Args:
            test_data: List of dicts with 'text' and 'label'
            use_api: If True, use API; else use direct agent calls
            verbose: Print each prediction
        """
        print(f"\n{'='*60}")
        print(f"RUNNING ACCURACY EVALUATION")
        print(f"{'='*60}")
        print(f"Test samples: {len(test_data)}")
        print(f"Method: {'API' if use_api else 'Direct'}")
        
        for i, sample in enumerate(test_data):
            try:
                text = sample['text']
                actual_label = sample['label']
                
                # Make prediction
                if use_api:
                    result = self.predict_via_api(text)
                    predicted_label = result['verdict']['label']
                else:
                    result = self.predict_direct(text)
                    predicted_label = result['verdict']['label']
                
                # Record result
                self.metrics.add_result(
                    predicted=predicted_label,
                    actual=actual_label,
                    metadata={
                        'text': text[:100],
                        'confidence': result.get('verdict', {}).get('confidence', 0)
                    }
                )
                
                if verbose:
                    status = "✓" if predicted_label.lower() == actual_label.lower() else "✗"
                    print(f"  [{status}] {i+1}: Predicted={predicted_label}, Actual={actual_label}")
                
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i+1}/{len(test_data)} samples...")
                    
            except Exception as e:
                print(f"  Error on sample {i+1}: {e}")
                self.metrics.add_result(
                    predicted='error',
                    actual=sample['label'],
                    metadata={'error': str(e)}
                )
        
        # Print report
        return self.metrics.print_report()
    
    def run_latency_benchmark(
        self, 
        test_data: list, 
        use_api: bool = True,
        warmup: int = 3
    ):
        """
        Run latency and throughput benchmark.
        """
        texts = [s['text'] for s in test_data]
        
        if use_api:
            def predict_fn(text):
                return self.predict_via_api(text)
            name = "API Prediction Latency"
        else:
            def predict_fn(text):
                return self.predict_direct(text)
            name = "Direct Prediction Latency"
        
        return self.benchmark.run_benchmark(
            name=name,
            func=predict_fn,
            inputs=texts,
            warmup_runs=warmup
        )
    
    def run_embedding_benchmark(self, test_data: list):
        """Benchmark embedding generation separately."""
        self._init_agents()
        texts = [s['text'] for s in test_data]
        
        def embed_fn(text):
            return self.lang_proc.get_embeddings(text)
        
        return self.benchmark.run_benchmark(
            name="Embedding Generation",
            func=embed_fn,
            inputs=texts,
            warmup_runs=2
        )
    
    def run_full_evaluation(
        self, 
        test_data: list,
        use_api: bool = True,
        output_dir: str = None
    ):
        """
        Run complete evaluation with accuracy and benchmarking.
        """
        print("\n" + "=" * 70)
        print("FULL SYSTEM EVALUATION")
        print("=" * 70)
        
        # 1. Accuracy evaluation
        accuracy_report = self.run_accuracy_evaluation(test_data, use_api=use_api)
        
        # 2. Latency benchmark
        latency_result = self.run_latency_benchmark(test_data, use_api=use_api)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.metrics.save_report(os.path.join(output_dir, 'accuracy_report.json'))
            self.benchmark.save_results(os.path.join(output_dir, 'benchmark_results.json'))
        
        # Summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Accuracy:          {accuracy_report['accuracy']*100:.1f}%")
        print(f"Macro F1:          {accuracy_report['macro_f1']:.4f}")
        print(f"Avg Latency:       {latency_result.avg_latency_ms:.0f} ms")
        print(f"P95 Latency:       {latency_result.p95_latency_ms:.0f} ms")
        print(f"Throughput:        {latency_result.throughput_rps:.2f} req/s")
        print("=" * 70)
        
        return {
            'accuracy': accuracy_report,
            'benchmark': self.benchmark.summary()
        }


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on fake news detection system')
    parser.add_argument('--dataset', type=str, 
                        default='data/dataset/processed.jsonl',
                        help='Path to test dataset (JSONL format)')
    parser.add_argument('--limit', type=int, default=50,
                        help='Max number of samples to evaluate')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                        help='API base URL')
    parser.add_argument('--direct', action='store_true',
                        help='Use direct agent calls instead of API')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for reports')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each prediction')
    
    args = parser.parse_args()
    
    # Find dataset path
    if not os.path.exists(args.dataset):
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        alt_path = project_root / args.dataset
        if alt_path.exists():
            args.dataset = str(alt_path)
        else:
            print(f"Error: Dataset not found at {args.dataset}")
            return
    
    # Run evaluation
    runner = EvaluationRunner(api_url=args.api_url)
    
    print(f"Loading test data from: {args.dataset}")
    test_data = runner.load_test_data(args.dataset, limit=args.limit)
    print(f"Loaded {len(test_data)} test samples")
    
    runner.run_full_evaluation(
        test_data=test_data,
        use_api=not args.direct,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
