"""
Benchmarking utilities for performance metrics and latency profiling.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import statistics
from pathlib import Path
import json

from ..core.base_module import BaseModule
from ..simulation import OrbitalEnv, EnvConfig


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    module_name: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_hz: float
    num_samples: int
    errors: int = 0
    error_rate: float = 0.0
    memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatencyProfiler:
    """Profiles inference latency of modules."""
    
    def __init__(self, warmup_runs: int = 10):
        """
        Args:
            warmup_runs: Number of warmup runs before profiling
        """
        self.warmup_runs = warmup_runs
    
    def profile_module(
        self,
        module: BaseModule,
        inputs: Dict[str, Any],
        num_runs: int = 1000,
        device: str = "cpu",
    ) -> BenchmarkResult:
        """
        Profile a module's inference latency.
        
        Args:
            module: Module to profile
            inputs: Input data
            num_runs: Number of profiling runs
            device: Compute device
            
        Returns:
            BenchmarkResult
        """
        latencies = []
        errors = 0
        
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                module.forward(inputs)
            except Exception:
                pass
        
        # Profile
        for _ in range(num_runs):
            start = time.perf_counter()
            try:
                module.forward(inputs)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                errors += 1
                continue
        
        if not latencies:
            raise RuntimeError("All runs failed")
        
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        # Compute statistics
        avg_latency = statistics.mean(latencies)
        p50 = latencies_sorted[n // 2]
        p95 = latencies_sorted[int(n * 0.95)]
        p99 = latencies_sorted[int(n * 0.99)]
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Throughput
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        # Memory usage (if CUDA)
        memory_mb = None
        if device == "cuda" and torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        
        return BenchmarkResult(
            module_name=module.name,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_hz=throughput,
            num_samples=num_runs,
            errors=errors,
            error_rate=errors / (num_runs + errors) if (num_runs + errors) > 0 else 0.0,
            memory_mb=memory_mb,
            metadata={
                "device": device,
                "warmup_runs": self.warmup_runs,
            }
        )


class SystemBenchmark:
    """Benchmarks the entire system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Benchmark configuration
        """
        self.config = config or {}
        self.profiler = LatencyProfiler(warmup_runs=self.config.get("warmup_runs", 10))
    
    def benchmark_module(
        self,
        module: BaseModule,
        input_generator,
        num_runs: int = 1000,
        device: str = "cpu",
    ) -> BenchmarkResult:
        """
        Benchmark a single module.
        
        Args:
            module: Module to benchmark
            input_generator: Function that generates input dicts
            num_runs: Number of runs
            device: Compute device
            
        Returns:
            BenchmarkResult
        """
        # Generate sample input
        sample_input = input_generator()
        
        return self.profiler.profile_module(
            module=module,
            inputs=sample_input,
            num_runs=num_runs,
            device=device,
        )
    
    def benchmark_system(
        self,
        modules: Dict[str, BaseModule],
        input_generators: Dict[str, callable],
        num_runs: int = 1000,
        device: str = "cpu",
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark multiple modules.
        
        Args:
            modules: Dictionary of module name -> module instance
            input_generators: Dictionary of module name -> input generator function
            num_runs: Number of runs per module
            device: Compute device
            
        Returns:
            Dictionary of module name -> BenchmarkResult
        """
        results = {}
        
        for name, module in modules.items():
            print(f"Benchmarking {name}...")
            generator = input_generators.get(name)
            if generator is None:
                print(f"  Warning: No input generator for {name}, skipping")
                continue
            
            try:
                result = self.benchmark_module(
                    module=module,
                    input_generator=generator,
                    num_runs=num_runs,
                    device=device,
                )
                results[name] = result
                print(f"  Average latency: {result.avg_latency_ms:.2f} ms")
                print(f"  P95 latency: {result.p95_latency_ms:.2f} ms")
                print(f"  Throughput: {result.throughput_hz:.2f} Hz")
            except Exception as e:
                print(f"  Error benchmarking {name}: {e}")
        
        return results
    
    def save_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: str,
    ):
        """Save benchmark results to JSON."""
        output = {}
        for name, result in results.items():
            output[name] = {
                "module_name": result.module_name,
                "avg_latency_ms": result.avg_latency_ms,
                "p50_latency_ms": result.p50_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
                "min_latency_ms": result.min_latency_ms,
                "max_latency_ms": result.max_latency_ms,
                "throughput_hz": result.throughput_hz,
                "num_samples": result.num_samples,
                "errors": result.errors,
                "error_rate": result.error_rate,
                "memory_mb": result.memory_mb,
                "metadata": result.metadata,
            }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")


def benchmark_all_modules(
    output_path: str = "benchmark_results.json",
    num_runs: int = 1000,
    device: str = "auto",
):
    """
    Benchmark all modules in the system.
    
    Args:
        output_path: Path to save results
        num_runs: Number of runs per module
        device: Compute device
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from ..models import (
        CollisionAvoidanceModule,
        NavigationModule,
        AnomalyDetector,
        EnergyManagementModule,
        StatePredictionModule,
    )
    from ..core.config import SystemConfig
    
    config = SystemConfig()
    
    # Create modules
    modules = {}
    input_generators = {}
    
    # Collision avoidance
    try:
        col_module = CollisionAvoidanceModule(
            config={"lidar_points": 256},
            device=device,
        )
        modules["collision_avoidance"] = col_module
        input_generators["collision_avoidance"] = lambda: {
            "lidar": np.random.randn(256, 6).astype(np.float32),
            "radar": np.random.randn(64).astype(np.float32),
            "imu": np.random.randn(12).astype(np.float32),
        }
    except Exception as e:
        print(f"Could not create collision_avoidance module: {e}")
    
    # Navigation
    try:
        nav_module = NavigationModule(
            config={},
            device=device,
        )
        modules["navigation"] = nav_module
        input_generators["navigation"] = lambda: {
            "gps_position": np.random.randn(3).astype(np.float32) * 7000,
            "gps_velocity": np.random.randn(3).astype(np.float32) * 7.5,
            "imu_accel": np.random.randn(3).astype(np.float32),
            "imu_gyro": np.random.randn(3).astype(np.float32),
            "star_tracker": np.random.randn(4).astype(np.float32),
        }
    except Exception as e:
        print(f"Could not create navigation module: {e}")
    
    # Anomaly detection
    try:
        anomaly_module = AnomalyDetector(
            config={"input_dim": 12, "seq_len": 50},
            device=device,
        )
        modules["anomaly_detection"] = anomaly_module
        input_generators["anomaly_detection"] = lambda: {
            "telemetry": np.random.randn(50, 12).astype(np.float32),
        }
    except Exception as e:
        print(f"Could not create anomaly_detection module: {e}")
    
    # Energy management
    try:
        energy_module = EnergyManagementModule(
            config={},
            device=device,
        )
        modules["energy_management"] = energy_module
        input_generators["energy_management"] = lambda: {
            "battery_soc": 0.8,
            "solar_power": 1000.0,
            "power_requests": np.array([100, 200, 150, 50]).astype(np.float32),
            "dt": 0.1,
        }
    except Exception as e:
        print(f"Could not create energy_management module: {e}")
    
    # State prediction
    try:
        state_module = StatePredictionModule(
            config={"input_dim": 13, "output_dim": 10, "prediction_horizon": 10},
            device=device,
        )
        modules["state_prediction"] = state_module
        input_generators["state_prediction"] = lambda: {
            "state_history": np.random.randn(50, 13).astype(np.float32),
        }
    except Exception as e:
        print(f"Could not create state_prediction module: {e}")
    
    # Run benchmarks
    benchmark = SystemBenchmark()
    results = benchmark.benchmark_system(
        modules=modules,
        input_generators=input_generators,
        num_runs=num_runs,
        device=device,
    )
    
    # Save results
    benchmark.save_results(results, output_path)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Average: {result.avg_latency_ms:.2f} ms")
        print(f"  P95: {result.p95_latency_ms:.2f} ms")
        print(f"  P99: {result.p99_latency_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_hz:.2f} Hz")
        if result.memory_mb:
            print(f"  Memory: {result.memory_mb:.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Space Debris AI System")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--num-runs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    benchmark_all_modules(
        output_path=args.output,
        num_runs=args.num_runs,
        device=args.device,
    )

