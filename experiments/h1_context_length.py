"""
experiments/h1_context_length.py

H1: Short contexts cause slowdown because verification overhead dominates.

Sweeps context lengths [128, 256, 512, 1024, 2048] and measures
speculative decoding speedup at each. We expect sub-1x speedup
at short context lengths.

Usage:
    modal run experiments/h1_context_length.py
"""

import modal

app = modal.App("spec-decoding-h1-context")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
)

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]
GAMMA = 4
MAX_NEW_TOKENS = 256
N_RUNS = 3


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h1(context_length: int, prompt: str):
    import torch
    import sys
    sys.path.insert(0, "/root")

    from src.baseline.autoregressive import load_model, run_baseline
    from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name, save_results, print_summary, RunMetrics

    # Load models
    base_model, base_tok = load_model(TARGET_MODEL)
    target_model, draft_model, tokenizer = load_models(TARGET_MODEL, DRAFT_MODEL)

    config = SpecDecodingConfig(gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS)

    # Run baseline N times
    baseline_times = []
    for _ in range(N_RUNS):
        r = run_baseline(base_model, base_tok, prompt, max_new_tokens=MAX_NEW_TOKENS)
        baseline_times.append(r.total_time_s)
    avg_baseline = sum(baseline_times) / len(baseline_times)

    # Run speculative N times
    spec_times = []
    spec_result = None
    for _ in range(N_RUNS):
        r = run_speculative(target_model, draft_model, tokenizer, prompt, config)
        spec_times.append(r.total_time_s)
        spec_result = r
    avg_spec = sum(spec_times) / len(spec_times)

    speedup = avg_baseline / avg_spec
    metrics = RunMetrics(
        experiment_name="H1_context_length",
        model_target=TARGET_MODEL,
        model_draft=DRAFT_MODEL,
        dataset="sharegpt",
        gpu=get_gpu_name(),
        context_length=context_length,
        max_new_tokens=MAX_NEW_TOKENS,
        gamma=GAMMA,
        baseline_time_s=avg_baseline,
        spec_time_s=avg_spec,
        speedup=speedup,
        acceptance_rate=spec_result.acceptance_rate,
        output_tokens=spec_result.output_tokens,
        throughput_baseline=MAX_NEW_TOKENS / avg_baseline,
        throughput_spec=spec_result.output_tokens / avg_spec,
        num_target_calls=spec_result.num_target_calls,
        total_draft_tokens=spec_result.total_draft_tokens,
        total_accepted_tokens=spec_result.total_accepted_tokens,
    )

    print_summary(metrics)
    return metrics.__dict__


@app.local_entrypoint()
def main():
    prompt_template = "User: Explain the concept of neural networks in detail, covering history, architecture, and modern applications. " * 10  # pad to hit context lengths

    futures = [run_h1.remote(ctx, prompt_template) for ctx in CONTEXT_LENGTHS]
    results = [f for f in futures]

    print("\n=== H1 Results: Speedup vs Context Length ===")
    for r in sorted(results, key=lambda x: x["context_length"]):
        marker = "SLOWDOWN" if r["speedup"] < 1.0 else "speedup"
        print(f"  ctx={r['context_length']:5d} | speedup={r['speedup']:.3f}x | acceptance={r['acceptance_rate']:.2%} | [{marker}]")
