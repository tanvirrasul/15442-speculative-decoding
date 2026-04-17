"""
experiments/h4_hardware.py

H4: Does GPU hardware determine speculative decoding viability?

Hypothesis: A100's high memory bandwidth (~2 TB/s) saturates both draft and
target models equally, eliminating the draft model's speed advantage. On
lower-bandwidth GPUs (A10G ~600 GB/s, T4 ~300 GB/s), the parameter count
difference between draft and target should translate to a real latency gap,
potentially enabling speedup.

Design:
  Fix:  TinyLlama-1.1B + Llama-2-7B-Chat draft models, Llama-2-7B target
  Vary: GPU (A100 vs A10G vs T4), gamma (1, 4, 8), domain (3 domains)

3 GPUs x 2 draft models x 3 gamma values x 3 domains = 54 conditions
3 prompts x 3 runs each = 486 total model runs

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run --detach experiments/h4_hardware.py
"""

import modal

app = modal.App("spec-decoding-h4-hardware")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"

DRAFT_MODELS = {
    "tinyllama_1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
}

GAMMA_VALUES = [1, 4, 8]
MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 512
N_RUNS = 3

GPU_BANDWIDTH = {
    "A100": 2000,
    "A10G": 600,
    "T4":   300,
}

DOMAIN_PROMPTS = {
    "general_news": [
        "The Federal Reserve announced Wednesday that it would hold interest rates steady for the third consecutive meeting, citing continued uncertainty about the trajectory of inflation and the labor market. Fed Chair Jerome Powell told reporters that while inflation has declined significantly from its peak, policymakers want to see more evidence that price pressures are sustainably moving toward the central bank's two percent target before cutting borrowing costs.",
        "Scientists at the European Space Agency have confirmed that the James Webb Space Telescope has detected water vapor in the atmosphere of a distant exoplanet for the first time with such precision. The discovery, published in the journal Nature Astronomy, represents a significant advance in the search for potentially habitable worlds beyond our solar system.",
        "Global semiconductor manufacturers are accelerating investment in new fabrication facilities amid growing concerns about supply chain resilience and geopolitical tensions. Taiwan Semiconductor Manufacturing Company, the world's largest contract chipmaker, announced plans to expand its operations in Arizona and Japan.",
    ],
    "code_completion": [
        """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)""",
        """class LinkedList:
    class Node:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, val):
        new_node = self.Node(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1""",
        """import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)""",
    ],
    "direct_continuation": [
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,",
        "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s,",
    ],
}


def pad_or_trim(prompt: str, target_length: int, tokenizer) -> str:
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= target_length:
        return tokenizer.decode(tokens[:target_length])
    while len(tokens) < target_length:
        tokens = tokens + tokenizer.encode(prompt)[1:]
    return tokenizer.decode(tokens[:target_length])


def _run_condition(draft_label, domain, gamma, gpu_string):
    """Shared logic — called from each GPU-specific function."""
    import os
    import gc
    import torch
    from transformers import AutoTokenizer
    from src.baseline.autoregressive import load_model, run_baseline
    from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name

    gpu_name = get_gpu_name()
    draft_model_id = DRAFT_MODELS[draft_label]
    print(f"[h4] gpu={gpu_name} | draft={draft_label} | domain={domain} | gamma={gamma}")

    tokenizer_only = AutoTokenizer.from_pretrained(
        TARGET_MODEL, token=os.environ["HF_TOKEN"]
    )

    raw_prompts = DOMAIN_PROMPTS[domain]
    prompts = [pad_or_trim(p, CONTEXT_LENGTH, tokenizer_only) for p in raw_prompts]

    config = SpecDecodingConfig(
        gamma=gamma,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose_timing=True,
        temperature=1.0,
        max_sequence_length=1536,
    )

    all_speedups = []
    all_acceptance = []
    all_baseline_tput = []
    all_spec_tput = []
    all_draft_times = []
    all_target_times = []

    for prompt_idx, prompt in enumerate(prompts):
        actual_tokens = len(tokenizer_only.encode(prompt))
        print(f"[h4] gpu={gpu_name} | draft={draft_label} | domain={domain} | gamma={gamma} | prompt {prompt_idx+1}/3 | tokens={actual_tokens}")

        # Baseline
        base_model, base_tok = load_model(TARGET_MODEL, device="cuda")
        baseline_times = []
        for _ in range(N_RUNS):
            r = run_baseline(base_model, base_tok, prompt, max_new_tokens=MAX_NEW_TOKENS, device="cuda")
            baseline_times.append(r.total_time_s)
        avg_baseline = sum(baseline_times) / len(baseline_times)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Speculative
        try:
            target_model, draft_model, tokenizer = load_models(
                TARGET_MODEL, draft_model_id, device="cuda"
            )
            spec_times = []
            spec_result = None
            for _ in range(N_RUNS):
                r = run_speculative(
                    target_model, draft_model, tokenizer, prompt, config, device="cuda"
                )
                spec_times.append(r.total_time_s)
                spec_result = r
            avg_spec = sum(spec_times) / len(spec_times)
            del target_model, draft_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[h4] WARNING: failed gpu={gpu_name} draft={draft_label} domain={domain} gamma={gamma} prompt={prompt_idx+1}: {e}")
            try:
                del target_model, draft_model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            continue

        speedup = avg_baseline / avg_spec
        all_speedups.append(speedup)
        all_acceptance.append(spec_result.acceptance_rate)
        all_baseline_tput.append(MAX_NEW_TOKENS / avg_baseline)
        all_spec_tput.append(spec_result.output_tokens / avg_spec)
        all_draft_times.append(spec_result.draft_time_s)
        all_target_times.append(spec_result.target_time_s)

        print(
            f"[h4] gpu={gpu_name} | draft={draft_label} | domain={domain} | gamma={gamma} | "
            f"prompt {prompt_idx+1} | speedup={speedup:.3f}x | "
            f"acceptance={spec_result.acceptance_rate:.2%} | "
            f"d/t={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x | "
            f"base={MAX_NEW_TOKENS/avg_baseline:.1f} t/s | "
            f"spec={spec_result.output_tokens/avg_spec:.1f} t/s"
        )

    if not all_speedups:
        print(f"[h4] WARNING: all prompts failed gpu={gpu_name} draft={draft_label} domain={domain} gamma={gamma}")
        return {
            "gpu": gpu_name, "gpu_requested": gpu_string,
            "draft_label": draft_label, "domain": domain, "gamma": gamma,
            "speedup": 0.0, "speedup_std": 0.0, "acceptance_rate": 0.0,
            "throughput_baseline": 0.0, "throughput_spec": 0.0,
            "avg_draft_time_s": 0.0, "avg_target_time_s": 0.0,
            "draft_target_ratio": 0.0, "failed": True,
        }

    avg_speedup = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "gpu": gpu_name,
        "gpu_requested": gpu_string,
        "draft_label": draft_label,
        "draft_model": draft_model_id,
        "domain": domain,
        "gamma": gamma,
        "speedup": avg_speedup,
        "speedup_std": std,
        "acceptance_rate": avg_acceptance,
        "throughput_baseline": sum(all_baseline_tput) / len(all_baseline_tput),
        "throughput_spec": sum(all_spec_tput) / len(all_spec_tput),
        "avg_draft_time_s": sum(all_draft_times) / len(all_draft_times),
        "avg_target_time_s": sum(all_target_times) / len(all_target_times),
        "draft_target_ratio": sum(all_draft_times) / max(sum(all_target_times), 1e-8),
        "failed": False,
    }

    marker = "SPEEDUP" if avg_speedup >= 1.0 else "SLOWDOWN"
    print(
        f"\n[h4] SUMMARY gpu={gpu_name} | draft={draft_label} | domain={domain} | gamma={gamma} | "
        f"speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | "
        f"d/t={result['draft_target_ratio']:.2f}x | [{marker}]"
    )
    return result


# ── GPU-specific functions defined at global scope ──

@app.function(
    image=image, gpu="A100", timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h4_a100(draft_label: str, domain: str, gamma: int):
    return _run_condition(draft_label, domain, gamma, "A100")


@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h4_a10g(draft_label: str, domain: str, gamma: int):
    return _run_condition(draft_label, domain, gamma, "A10G")


@app.function(
    image=image, gpu="T4", timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h4_t4(draft_label: str, domain: str, gamma: int):
    return _run_condition(draft_label, domain, gamma, "T4")


GPU_FUNCTIONS = {
    "A100": run_h4_a100,
    "A10G": run_h4_a10g,
    "T4":   run_h4_t4,
}


@app.local_entrypoint()
def main():
    import sys

    # Spawn all jobs first before collecting any results
    # This way disconnection doesn't prevent jobs from starting
    print("Spawning all jobs...")
    job_specs = []
    for gpu_string, fn in GPU_FUNCTIONS.items():
        for draft_label in DRAFT_MODELS:
            for domain in DOMAIN_PROMPTS:
                for gamma in GAMMA_VALUES:
                    handle = fn.spawn(draft_label, domain, gamma)
                    job_specs.append((gpu_string, draft_label, domain, gamma, handle))
                    print(f"  Spawned: {gpu_string} | {draft_label} | {domain} | gamma={gamma}")

    print(f"\nAll {len(job_specs)} jobs spawned. Collecting results...")

    results = []
    failed = []
    for i, (gpu_string, draft_label, domain, gamma, handle) in enumerate(job_specs):
        try:
            r = handle.get(timeout=7200)
            if not r.get("failed", False):
                r["bandwidth_gbs"] = GPU_BANDWIDTH[gpu_string]
                results.append(r)
                marker = "SPEEDUP ✓" if r["speedup"] >= 1.0 else "slowdown"
                print(f"[{i+1}/{len(job_specs)}] {gpu_string} | {draft_label} | {domain} | γ={gamma} | {r['speedup']:.3f}x | [{marker}]")
        except Exception as e:
            print(f"[{i+1}/{len(job_specs)}] FAILED {gpu_string} | {draft_label} | {domain} | γ={gamma}: {e}")
            failed.append((gpu_string, draft_label, domain, gamma))

    if failed:
        print(f"\nWARNING: {len(failed)} jobs failed: {failed}")

    if not results:
        print("No results collected.")
        return

    # ── Full results table ──
    results_sorted = sorted(
        results,
        key=lambda x: (x["gpu_requested"], x["draft_label"], x["domain"], x["gamma"])
    )

    print("\n=== H4 Full Results: Hardware Effect on Speculative Decoding ===")
    print(f"{'gpu':>6} | {'bw':>5} | {'draft':>16} | {'domain':>18} | {'γ':>2} | {'accept':>7} | {'speedup':>8} | {'±std':>5} | {'base t/s':>8} | {'spec t/s':>8} | {'d/t':>5} | result")
    print("-" * 130)
    for r in results_sorted:
        marker = "SPEEDUP ✓" if r["speedup"] >= 1.0 else "slowdown"
        print(
            f"  {r['gpu_requested']:>4} | "
            f"{r['bandwidth_gbs']:>5} | "
            f"  {r['draft_label']:>14} | "
            f"  {r['domain']:>16} | "
            f"{r['gamma']:>2} | "
            f"{r['acceptance_rate']:>6.1%} | "
            f"{r['speedup']:>7.3f}x | "
            f"{r['speedup_std']:>5.3f} | "
            f"{r['throughput_baseline']:>7.1f}   | "
            f"{r['throughput_spec']:>7.1f}   | "
            f"{r['draft_target_ratio']:>4.2f}x | "
            f"[{marker}]"
        )

    # ── GPU x Draft summary ──
    print("\n=== GPU x Draft Summary (averaged across domains and gamma) ===")
    print(f"{'gpu':>6} | {'bw (GB/s)':>10} | {'draft':>16} | {'avg accept':>11} | {'avg speedup':>12} | {'avg d/t':>8} | {'base t/s':>9} | {'spec t/s':>9} | result")
    print("-" * 100)
    for gpu_string in GPU_FUNCTIONS:
        for draft_label in DRAFT_MODELS:
            subset = [r for r in results if r["gpu_requested"] == gpu_string and r["draft_label"] == draft_label]
            if not subset:
                continue
            avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
            avg_accept  = sum(r["acceptance_rate"] for r in subset) / len(subset)
            avg_dt      = sum(r["draft_target_ratio"] for r in subset) / len(subset)
            avg_base    = sum(r["throughput_baseline"] for r in subset) / len(subset)
            avg_spec_t  = sum(r["throughput_spec"] for r in subset) / len(subset)
            bw          = GPU_BANDWIDTH[gpu_string]
            marker      = "SPEEDUP ✓" if avg_speedup >= 1.0 else "slowdown"
            print(
                f"  {gpu_string:>4} | "
                f"{bw:>10} | "
                f"  {draft_label:>14} | "
                f"{avg_accept:>10.1%} | "
                f"{avg_speedup:>11.3f}x | "
                f"{avg_dt:>7.2f}x | "
                f"{avg_base:>8.1f}   | "
                f"{avg_spec_t:>8.1f}   | "
                f"[{marker}]"
            )

    # ── GPU bandwidth summary ──
    print("\n=== GPU Bandwidth Effect (averaged across all conditions) ===")
    print(f"{'gpu':>6} | {'bw (GB/s)':>10} | {'avg accept':>11} | {'avg speedup':>12} | {'avg d/t':>8} | {'base t/s':>9} | {'spec t/s':>9} | result")
    print("-" * 85)
    for gpu_string in GPU_FUNCTIONS:
        subset = [r for r in results if r["gpu_requested"] == gpu_string]
        if not subset:
            continue
        avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
        avg_accept  = sum(r["acceptance_rate"] for r in subset) / len(subset)
        avg_dt      = sum(r["draft_target_ratio"] for r in subset) / len(subset)
        avg_base    = sum(r["throughput_baseline"] for r in subset) / len(subset)
        avg_spec_t  = sum(r["throughput_spec"] for r in subset) / len(subset)
        bw          = GPU_BANDWIDTH[gpu_string]
        marker      = "SPEEDUP ✓" if avg_speedup >= 1.0 else "slowdown"
        print(
            f"  {gpu_string:>4} | "
            f"{bw:>10} | "
            f"{avg_accept:>10.1%} | "
            f"{avg_speedup:>11.3f}x | "
            f"{avg_dt:>7.2f}x | "
            f"{avg_base:>8.1f}   | "
            f"{avg_spec_t:>8.1f}   | "
            f"[{marker}]"
        )

    # ── Per-gamma breakdown per GPU ──
    print("\n=== Gamma Effect Per GPU ===")
    for gpu_string in GPU_FUNCTIONS:
        print(f"\n  {gpu_string}:")
        for gamma in GAMMA_VALUES:
            subset = [r for r in results if r["gpu_requested"] == gpu_string and r["gamma"] == gamma]
            if not subset:
                continue
            avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
            avg_dt = sum(r["draft_target_ratio"] for r in subset) / len(subset)
            n_speedup = len([r for r in subset if r["speedup"] >= 1.0])
            marker = f"SPEEDUP ✓ ({n_speedup}/{len(subset)} conditions)" if n_speedup else "all slowdown"
            print(f"    gamma={gamma}: avg speedup={avg_speedup:.3f}x | avg d/t={avg_dt:.2f}x | [{marker}]")

    # ── Break-even analysis ──
    print("\n=== Break-even Analysis ===")
    for gpu_string in GPU_FUNCTIONS:
        speedup_results = [r for r in results if r["gpu_requested"] == gpu_string and r["speedup"] >= 1.0]
        if speedup_results:
            best = max(speedup_results, key=lambda x: x["speedup"])
            print(
                f"  {gpu_string}: SPEEDUP in {len(speedup_results)} conditions! "
                f"Best={best['speedup']:.3f}x | draft={best['draft_label']} | "
                f"domain={best['domain']} | gamma={best['gamma']} | "
                f"accept={best['acceptance_rate']:.1%} | d/t={best['draft_target_ratio']:.2f}x"
            )
        else:
            n = len([r for r in results if r["gpu_requested"] == gpu_string])
            print(f"  {gpu_string}: no speedup observed across all {n} conditions")