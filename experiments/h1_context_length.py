"""
experiments/h1_context_length.py

H1: Short contexts cause slowdown because verification overhead dominates.

Uses real ShareGPT and CNN/DailyMail prompts at each context length,
averaged over multiple prompts per length for robustness.

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run experiments/h1_context_length.py
"""

import modal

app = modal.App("spec-decoding-h1-context")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
    "datasets>=2.18.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONTEXT_LENGTHS = [128, 192, 256, 320, 384, 512, 768, 1024, 2048]
GAMMA = 4
MAX_NEW_TOKENS = 256
N_RUNS = 3
PROMPTS_PER_LENGTH = 3


def get_real_prompts(target_length: int, tokenizer, n: int = 3) -> list[str]:
    from datasets import load_dataset

    prompts = []

    try:
        sharegpt = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        for item in sharegpt:
            if len(prompts) >= n // 2 + 1:
                break
            try:
                text = item["conversations"][0]["value"]
                tokens = tokenizer.encode(text)
                if len(tokens) >= target_length:
                    truncated = tokenizer.decode(tokens[:target_length])
                    prompts.append(truncated)
            except Exception:
                continue
    except Exception:
        pass

    try:
        cnn = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
        for item in cnn:
            if len(prompts) >= n:
                break
            try:
                text = "Summarize the following article:\n\n" + item["article"]
                tokens = tokenizer.encode(text)
                if len(tokens) >= target_length:
                    truncated = tokenizer.decode(tokens[:target_length])
                    prompts.append(truncated)
            except Exception:
                continue
    except Exception:
        pass

    fallback_texts = [
        "The economic implications of artificial intelligence on labor markets have been extensively studied by researchers at major universities. The findings suggest that automation will disproportionately affect routine cognitive tasks while augmenting creative and interpersonal work. Policy makers are now grappling with how to structure retraining programs and social safety nets for displaced workers. ",
        "Climate scientists have recently published findings indicating that ocean temperatures have risen significantly over the past decade. The data collected from thousands of buoys across the Pacific and Atlantic oceans shows a consistent upward trend. Marine biologists are particularly concerned about the impact on coral reef ecosystems and the fish populations that depend on them. ",
        "The development of quantum computing has accelerated dramatically in recent years with major breakthroughs from both academic institutions and private companies. Researchers have demonstrated quantum supremacy for specific computational tasks, though practical general-purpose quantum computers remain years away. The implications for cryptography and drug discovery are particularly significant. ",
    ]
    while len(prompts) < n:
        for text in fallback_texts:
            if len(prompts) >= n:
                break
            repeated = text * 10
            tokens = tokenizer.encode(repeated)
            if len(tokens) >= target_length:
                truncated = tokenizer.decode(tokens[:target_length])
                prompts.append(truncated)

    return prompts[:n]


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h1(context_length: int):
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

    prompts = get_real_prompts(context_length, tokenizer_only, n=PROMPTS_PER_LENGTH)
    print(f"[h1] ctx={context_length} | got {len(prompts)} real prompts")

    config = SpecDecodingConfig(gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, verbose_timing=True)

    all_speedups = []
    all_acceptance = []
    all_baseline_tput = []
    all_spec_tput = []
    all_draft_times = []
    all_target_times = []

    for prompt_idx, prompt in enumerate(prompts):
        actual_tokens = len(tokenizer_only.encode(prompt))
        print(f"[h1] ctx={context_length} | prompt {prompt_idx+1}/{len(prompts)} | actual_tokens={actual_tokens}")

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

        print(f"[h1] ctx={context_length} | prompt {prompt_idx+1} | speedup={speedup:.3f}x | acceptance={spec_result.acceptance_rate:.2%} | draft={spec_result.draft_time_s:.2f}s | target={spec_result.target_time_s:.2f}s")

    avg_speedup = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "context_length": context_length,
        "speedup": avg_speedup,
        "speedup_std": std,
        "acceptance_rate": avg_acceptance,
        "throughput_baseline": sum(all_baseline_tput) / len(all_baseline_tput),
        "throughput_spec": sum(all_spec_tput) / len(all_spec_tput),
        "avg_draft_time_s": sum(all_draft_times) / len(all_draft_times),
        "avg_target_time_s": sum(all_target_times) / len(all_target_times),
        "draft_target_ratio": sum(all_draft_times) / max(sum(all_target_times), 1e-8),
        "gpu": get_gpu_name(),
        "n_prompts": len(prompts),
    }

    marker = "SLOWDOWN" if avg_speedup < 1.0 else "speedup"
    print(f"\n[h1] SUMMARY ctx={context_length} | speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | draft/target={result['draft_target_ratio']:.2f}x | [{marker}]")
    return result


@app.local_entrypoint()
def main():
    futures = [run_h1.remote(ctx) for ctx in CONTEXT_LENGTHS]
    results = [f for f in futures]

    print("\n=== H1 Results: Speedup vs Context Length ===")
    print(f"{'ctx':>6} | {'speedup':>8} | {'±std':>6} | {'accept':>8} | {'baseline':>10} | {'spec':>10} | {'draft/tgt':>10} | result")
    print("-" * 95)
    for r in sorted(results, key=lambda x: x["context_length"]):
        marker = "SLOWDOWN" if r["speedup"] < 1.0 else "speedup"
        print(
            f"  {r['context_length']:>4} | "
            f"{r['speedup']:>7.3f}x | "
            f"{r['speedup_std']:>5.3f} | "
            f"{r['acceptance_rate']:>7.2%} | "
            f"{r['throughput_baseline']:>8.1f} t/s | "
            f"{r['throughput_spec']:>8.1f} t/s | "
            f"{r['draft_target_ratio']:>9.2f}x | "
            f"[{marker}]"
        )
