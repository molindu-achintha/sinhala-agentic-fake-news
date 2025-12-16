"""
Benchmarking Module for Fake News Detection System.

Measures response time, throughput, and latency.
"""
import time
import statistics
from typing import List, Dict, Callable
from dataclasses import dataclass, field
import json


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float  
    latencies: List[float] = field(default_factory=list)


class Benchmark:
    """
    Benchmark the fake news detection system.
    
    Measures:
    - Response time (latency)
    - Throughput (requests per second)
    - Success rate
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self, 
        name: str,
        func: Callable,
        inputs: List,
        warmup_runs: int = 3
    ) -> BenchmarkResult:
        """
        Run a benchmark on a function.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark (takes single input)
            inputs: List of inputs to test
            warmup_runs: Number of warmup runs before measurement
        
        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"\n{'='*50}")
        print(f"Running Benchmark: {name}")
        print(f"{'='*50}")
        
        # Warmup
        print(f"Warming up ({warmup_runs} runs)...")
        for i in range(min(warmup_runs, len(inputs))):
            try:
                func(inputs[i])
            except Exception:
                pass
        
        # Actual benchmark
        latencies = []
        successful = 0
        failed = 0
        
        start_total = time.time()
        
        for i, inp in enumerate(inputs):
            start = time.perf_counter()
            try:
                func(inp)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"  Request {i+1} failed: {e}")
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(inputs)} requests...")
        
        end_total = time.time()
        total_time = end_total - start_total
        
        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies)-1)]
            p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies)-1)]
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = median_latency = p95_latency = p99_latency = 0
            min_latency = max_latency = 0
        
        throughput = len(inputs) / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            total_requests=len(inputs),
            successful_requests=successful,
            failed_requests=failed,
            total_time_seconds=total_time,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_rps=throughput,
            latencies=latencies
        )
        
        self.results.append(result)
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print(f"\n{'-'*50}")
        print(f"Benchmark: {result.name}")
        print(f"{'-'*50}")
        print(f"Total Requests:    {result.total_requests}")
        print(f"Successful:        {result.successful_requests}")
        print(f"Failed:            {result.failed_requests}")
        print(f"Success Rate:      {result.successful_requests/result.total_requests*100:.1f}%")
        print(f"Total Time:        {result.total_time_seconds:.2f}s")
        print(f"Throughput:        {result.throughput_rps:.2f} req/s")
        print(f"\nLatency Statistics:")
        print(f"  Average:         {result.avg_latency_ms:.2f} ms")
        print(f"  Median:          {result.median_latency_ms:.2f} ms")
        print(f"  P95:             {result.p95_latency_ms:.2f} ms")
        print(f"  P99:             {result.p99_latency_ms:.2f} ms")
        print(f"  Min:             {result.min_latency_ms:.2f} ms")
        print(f"  Max:             {result.max_latency_ms:.2f} ms")
        print(f"{'-'*50}")
    
    def summary(self) -> Dict:
        """Get summary of all benchmark results."""
        return {
            'benchmarks': [
                {
                    'name': r.name,
                    'total_requests': r.total_requests,
                    'success_rate': r.successful_requests / r.total_requests if r.total_requests > 0 else 0,
                    'avg_latency_ms': r.avg_latency_ms,
                    'p95_latency_ms': r.p95_latency_ms,
                    'throughput_rps': r.throughput_rps
                }
                for r in self.results
            ]
        }
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON."""
        data = {
            'benchmarks': []
        }
        
        for r in self.results:
            data['benchmarks'].append({
                'name': r.name,
                'total_requests': r.total_requests,
                'successful_requests': r.successful_requests,
                'failed_requests': r.failed_requests,
                'total_time_seconds': r.total_time_seconds,
                'avg_latency_ms': r.avg_latency_ms,
                'median_latency_ms': r.median_latency_ms,
                'p95_latency_ms': r.p95_latency_ms,
                'p99_latency_ms': r.p99_latency_ms,
                'min_latency_ms': r.min_latency_ms,
                'max_latency_ms': r.max_latency_ms,
                'throughput_rps': r.throughput_rps,
                'latencies': r.latencies
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Benchmark results saved to: {filepath}")
