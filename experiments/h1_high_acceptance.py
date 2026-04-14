"""
experiments/h1_high_acceptance.py

Supplementary H1 experiment: test speculative decoding on high-acceptance
prompt types to confirm the speedup curve flips when acceptance rate is high.

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run experiments/h1_high_acceptance.py
"""

import modal

app = modal.App("spec-decoding-h1-high-acceptance")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONTEXT_LENGTHS = [128, 512, 1024]
GAMMA = 4
MAX_NEW_TOKENS = 256
N_RUNS = 3

HIGH_ACCEPTANCE_PROMPTS = {
    "code_repetitive": [
        """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def modulo(a, b):
    return a % b

def power(a, b):
    return a ** b

def floor_divide(a, b):
    return a // b

def absolute(a):
    return abs(a)

def negate(a):
    return -a

def increment(a):
    return a + 1

def decrement(a):
    return a - 1

def double(a):
    return a * 2

def halve(a):
    return a / 2

def square(a):""",
        """{"users": [{"id": 1, "name": "Alice", "age": 30, "email": "alice@example.com", "active": true}, {"id": 2, "name": "Bob", "age": 25, "email": "bob@example.com", "active": true}, {"id": 3, "name": "Carol", "age": 35, "email": "carol@example.com", "active": false}, {"id": 4, "name": "Dave", "age": 28, "email": "dave@example.com", "active": true}, {"id": 5, "name": "Eve", "age": 32, "email": "eve@example.com", "active": true}, {"id": 6, "name": "Frank", "age": 45, "email": "frank@example.com", "active": false}, {"id": 7, "name": "Grace", "age": 29,""",
        """| Name       | Age | Department  | Salary  | Years |
|------------|-----|-------------|---------|-------|
| Alice      | 30  | Engineering | 120000  | 5     |
| Bob        | 25  | Marketing   | 85000   | 2     |
| Carol      | 35  | Engineering | 140000  | 8     |
| Dave       | 28  | Sales       | 95000   | 3     |
| Eve        | 32  | Engineering | 130000  | 6     |
| Frank      | 45  | Management  | 160000  | 15    |
| Grace      | 29  | Marketing   | 88000   | 4     |
| Henry      | 38  | Engineering | 145000  | 10    |
| Iris       | 31  | Sales       | 98000   | 5     |
| Jack       | 27  | Engineering | 115000  |""",
    ],
}


def pad_prompt_to_length(prompt: str, target_length: int, tokenizer) -> str:
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= target_length:
        return tokenizer.decode(tokens[:target_length])
    while len(tokens) < target_length:
        tokens = tokens + tokenizer.encode(prompt)[1:]
    return tokenizer.decode(tokens[:target_length])


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h1_high_acceptance(context_length: int, prompt_type: str):
    import os
    import gc
    import torch
    from transformers import AutoTokenizer
    from src.baseline.autoregressive import load_model, run_baseline
    from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name

    tokenizer_only = AutoTokenizer.from_pretrained(
        TARGET_MODEL, token=os.environ["HF_TOKEN"]
    )

    raw_prompts = HIGH_ACCEPTANCE_PROMPTS[prompt_type]
    prompts = [pad_prompt_to_length(p, context_length, tokenizer_only) for p in raw_prompts]

    config = SpecDecodingConfig(gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, verbose_timing=True, temperature=0.0)

    all_speedups = []
    all_acceptance = []
    all_baseline_tput = []
    all_spec_tput = []
    all_draft_times = []
    all_target_times = []

    for prompt_idx, prompt in enumerate(prompts):
        actual_tokens = len(tokenizer_only.encode(prompt))
        print(f"[h1-high] ctx={context_length} | prompt_type={prompt_type} | prompt {prompt_idx+1}/{len(prompts)} | tokens={actual_tokens}")

        base_model, base_tok = load_model(TARGET_MODEL, device="cuda")
        baseline_times = []
        for _ in range(N_RUNS):
            r = run_baseline(base_model, base_tok, prompt, max_new_tokens=MAX_NEW_TOKENS, device="cuda")
            baseline_times.append(r.total_time_s)
        avg_baseline = sum(baseline_times) / len(baseline_times)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        target_model, draft_model, tokenizer = load_models(TARGET_MODEL, DRAFT_MODEL, device="cuda")
        spec_times = []
        spec_result = None
        for _ in range(N_RUNS):
            r = run_speculative(target_model, draft_model, tokenizer, prompt, config, device="cuda")
            spec_times.append(r.total_time_s)
            spec_result = r
        avg_spec = sum(spec_times) / len(spec_times)
        del target_model, draft_model
        gc.collect()
        torch.cuda.empty_cache()

        speedup = avg_baseline / avg_spec
        all_speedups.append(speedup)
        all_acceptance.append(spec_result.acceptance_rate)
        all_baseline_tput.append(MAX_NEW_TOKENS / avg_baseline)
        all_spec_tput.append(spec_result.output_tokens / avg_spec)
        all_draft_times.append(spec_result.draft_time_s)
        all_target_times.append(spec_result.target_time_s)

        print(f"[h1-high] ctx={context_length} | prompt {prompt_idx+1} | speedup={speedup:.3f}x | acceptance={spec_result.acceptance_rate:.2%} | draft={spec_result.draft_time_s:.2f}s | target={spec_result.target_time_s:.2f}s")

    avg_speedup = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "context_length": context_length,
        "prompt_type": prompt_type,
        "speedup": avg_speedup,
        "speedup_std": std,
        "acceptance_rate": avg_acceptance,
        "throughput_baseline": sum(all_baseline_tput) / len(all_baseline_tput),
        "throughput_spec": sum(all_spec_tput) / len(all_spec_tput),
        "avg_draft_time_s": sum(all_draft_times) / len(all_draft_times),
        "avg_target_time_s": sum(all_target_times) / len(all_target_times),
        "draft_target_ratio": sum(all_draft_times) / max(sum(all_target_times), 1e-8),
        "gpu": get_gpu_name(),
    }

    marker = "SLOWDOWN" if avg_speedup < 1.0 else "speedup"
    print(f"\n[h1-high] SUMMARY ctx={context_length} | type={prompt_type} | speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | draft/target={result['draft_target_ratio']:.2f}x | [{marker}]")
    return result


@app.local_entrypoint()
def main():
    futures = []
    for prompt_type in HIGH_ACCEPTANCE_PROMPTS:
        for ctx in CONTEXT_LENGTHS:
            futures.append(run_h1_high_acceptance.remote(ctx, prompt_type))

    results = [f for f in futures]

    print("\n=== H1 High-Acceptance Results ===")
    print(f"{'ctx':>6} | {'type':>16} | {'speedup':>8} | {'±std':>6} | {'accept':>8} | {'baseline':>10} | {'spec':>10} | {'draft/tgt':>10} | result")
    print("-" * 105)
    for r in sorted(results, key=lambda x: (x["prompt_type"], x["context_length"])):
        marker = "SLOWDOWN" if r["speedup"] < 1.0 else "speedup"
        print(
            f"  {r['context_length']:>4} | "
            f"{r['prompt_type']:>16} | "
            f"{r['speedup']:>7.3f}x | "
            f"{r['speedup_std']:>5.3f} | "
            f"{r['acceptance_rate']:>7.2%} | "
            f"{r['throughput_baseline']:>8.1f} t/s | "
            f"{r['throughput_spec']:>8.1f} t/s | "
            f"{r['draft_target_ratio']:>9.2f}x | "
            f"[{marker}]"
        )
