"""
Profiling and metrics collection utilities.
Wraps CUDA events and Nsight ranges for granular breakdown of:
  - Draft generation time
  - Target verification time
  - Memory bandwidth utilization
"""

import torch
import time
import csv
import os
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class RunMetrics:
    experiment_name: str
    model_target: str
    model_draft: str
    dataset: str
    gpu: str
    context_length: int
    max_new_tokens: int
    gamma: int
    # Timing
    baseline_time_s: float
    spec_time_s: float
    speedup: float
    # Quality
    acceptance_rate: float
    output_tokens: int
    throughput_baseline: float
    throughput_spec: float
    # Spec decoding internals
    num_target_calls: int
    total_draft_tokens: int
    total_accepted_tokens: int


@contextmanager
def cuda_timer(label: str = ""):
    """Context manager that times a CUDA block using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    if label:
        print(f"[{label}] {elapsed_ms:.2f} ms")
    return elapsed_ms


@contextmanager
def nsight_range(label: str):
    """
    Wraps a block in an NVTX range for Nsight Systems profiling.
    No-ops if torch.cuda.nvtx is unavailable.
    """
    try:
        torch.cuda.nvtx.range_push(label)
        yield
    finally:
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def save_results(metrics: RunMetrics, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.csv")
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys())
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(metrics))
    print(f"[profiling] Saved results to {path}")


def print_summary(metrics: RunMetrics):
    print("\n" + "=" * 55)
    print(f"  Experiment : {metrics.experiment_name}")
    print(f"  Target     : {metrics.model_target}")
    print(f"  Draft      : {metrics.model_draft}")
    print(f"  GPU        : {metrics.gpu}")
    print(f"  Context    : {metrics.context_length} tokens")
    print("-" * 55)
    print(f"  Speedup             : {metrics.speedup:.3f}x")
    print(f"  Acceptance rate     : {metrics.acceptance_rate:.2%}")
    print(f"  Baseline throughput : {metrics.throughput_baseline:.1f} tok/s")
    print(f"  Spec throughput     : {metrics.throughput_spec:.1f} tok/s")
    print(f"  Target model calls  : {metrics.num_target_calls}")
    print("=" * 55 + "\n")
