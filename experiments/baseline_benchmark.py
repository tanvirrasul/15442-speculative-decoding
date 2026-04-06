"""
experiments/baseline_benchmark.py

Runs the autoregressive baseline on Modal across context lengths and datasets.
Usage:
    modal run experiments/baseline_benchmark.py
"""

import modal
import sys

app = modal.App("spec-decoding-baseline")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "datasets>=2.18.0",
    "sentencepiece>=0.2.0",
)

# Cache model weights across runs
volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]
MAX_NEW_TOKENS = 256
N_RUNS = 3  # average over N runs per config


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN for gated models
)
def run_baseline_benchmark(context_length: int, dataset_name: str, prompt: str):
    import torch
    import sys
    sys.path.insert(0, "/root")

    from src.baseline.autoregressive import load_model, run_baseline
    from src.profiling.metrics import get_gpu_name, save_results, print_summary, RunMetrics

    model, tokenizer = load_model(TARGET_MODEL)

    times = []
    for _ in range(N_RUNS):
        result = run_baseline(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        times.append(result.total_time_s)

    avg_time = sum(times) / len(times)
    throughput = MAX_NEW_TOKENS / avg_time

    print(f"[baseline] ctx={context_length} | dataset={dataset_name} | time={avg_time:.3f}s | throughput={throughput:.1f} tok/s")
    return {
        "context_length": context_length,
        "dataset": dataset_name,
        "avg_time_s": avg_time,
        "throughput": throughput,
        "gpu": get_gpu_name(),
    }


@app.local_entrypoint()
def main():
    # Sample prompts per dataset — replace with real dataset loading
    prompts = {
        "sharegpt": "User: Tell me about the history of machine learning. Assistant:",
        "cnn_dailymail": "Summarize the following article in detail: The stock market saw significant volatility today as investors reacted to new inflation data released by the Federal Reserve...",
        "humaneval": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    }

    futures = []
    for dataset_name, prompt in prompts.items():
        for ctx_len in CONTEXT_LENGTHS:
            futures.append(run_baseline_benchmark.remote(ctx_len, dataset_name, prompt))

    results = [f for f in futures]
    print("\nAll baseline results:")
    for r in results:
        print(r)
