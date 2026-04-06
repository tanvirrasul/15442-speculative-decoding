"""
experiments/baseline_benchmark.py

Runs the autoregressive baseline on Modal across context lengths and datasets.
Usage:
    modal run experiments/baseline_benchmark.py
"""

import modal

app = modal.App("spec-decoding-baseline")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "datasets>=2.18.0",
    "sentencepiece>=0.2.0",
).add_local_dir("./src", remote_path="/root/src")


volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]
MAX_NEW_TOKENS = 256
N_RUNS = 3


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_baseline_benchmark(context_length: int, dataset_name: str, prompt: str):
    import os
    from src.baseline.autoregressive import load_model, run_baseline
    from src.profiling.metrics import get_gpu_name

    model, tokenizer = load_model(TARGET_MODEL, device="cuda")

    times = []
    throughputs = []
    for _ in range(N_RUNS):
        result = run_baseline(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, device="cuda")
        times.append(result.total_time_s)
        throughputs.append(result.throughput_tokens_per_s)

    avg_time = sum(times) / len(times)
    avg_throughput = sum(throughputs) / len(throughputs)

    result_dict = {
        "context_length": context_length,
        "dataset": dataset_name,
        "avg_time_s": avg_time,
        "throughput_tokens_per_s": avg_throughput,
        "gpu": get_gpu_name(),
    }

    print(f"[baseline] ctx={context_length} | dataset={dataset_name} | time={avg_time:.3f}s | throughput={avg_throughput:.1f} tok/s")
    return result_dict


@app.local_entrypoint()
def main():
    prompts = {
        "sharegpt": "User: Tell me about the history of machine learning in detail, covering the key milestones, researchers, and breakthroughs that shaped the field. Assistant:",
        "cnn_dailymail": "Summarize the following article in detail: The stock market saw significant volatility today as investors reacted to new inflation data released by the Federal Reserve. Analysts noted that the figures came in higher than expected, prompting concerns about further interest rate hikes.",
        "humaneval": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    }

    futures = []
    for dataset_name, prompt in prompts.items():
        for ctx_len in CONTEXT_LENGTHS:
            futures.append(run_baseline_benchmark.remote(ctx_len, dataset_name, prompt))

    results = [f for f in futures]

    print("\n=== Baseline Results ===")
    for r in sorted(results, key=lambda x: (x["dataset"], x["context_length"])):
        print(f"  dataset={r['dataset']} | ctx={r['context_length']:5d} | time={r['avg_time_s']:.3f}s | throughput={r['throughput_tokens_per_s']:.1f} tok/s | gpu={r['gpu']}")