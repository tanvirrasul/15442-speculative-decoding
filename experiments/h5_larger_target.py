"""
experiments/h5_larger_target.py

H5: Llama-2-7B (fp16) draft → Llama-2-70B-chat (NF4 4-bit) target on A100-80GB.
70B in NF4 ~35GB + 7B fp16 ~14GB + KV cache ~10GB = ~59GB — fits with headroom.

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run --detach experiments/h5_larger_target.py
"""

import os
import modal

app = modal.App("spec-decoding-h5-v4")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.44.0",
    "accelerate==0.33.0",
    "sentencepiece>=0.2.0",
    "bitsandbytes>=0.41.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-70b-chat-hf"   # NF4 4-bit, ~35GB
DRAFT_MODEL  = "meta-llama/Llama-2-7b-hf"          # fp16, ~14GB

GAMMA_VALUES   = [4, 8, 12]
MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 512
N_RUNS         = 3

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
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,",
        "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s,",
    ],
}


def pad_or_trim(prompt: str, target_length: int, tokenizer) -> str:
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= target_length:
        return tokenizer.decode(tokens[:target_length])
    while len(tokens) < target_length:
        tokens = tokens + tokenizer.encode(prompt)[1:]
    return tokenizer.decode(tokens[:target_length])


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h5(domain: str, gamma: int):
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from src.baseline.autoregressive import run_baseline
    from src.speculative.spec_decode import run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_name = get_gpu_name()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[h5] gpu={gpu_name} ({total_vram:.0f}GB) | domain={domain} | gamma={gamma}")
    print(f"[h5] Target: {TARGET_MODEL} (NF4 4-bit) | Draft: {DRAFT_MODEL} (fp16)")
    print(f"[h5] Expected VRAM: ~35GB target + ~14GB draft + ~10GB KV = ~59GB")

    # NF4 4-bit config for 70B target — ~35GB on A100-80GB
    bnb_4bit_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL, token=os.environ["HF_TOKEN"]
    )
    raw_prompts = DOMAIN_PROMPTS[domain]
    prompts = [pad_or_trim(p, CONTEXT_LENGTH, tokenizer) for p in raw_prompts]

    config = SpecDecodingConfig(
        gamma=gamma,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose_timing=True,
        temperature=1.0,
        max_sequence_length=1536,
    )

    all_speedups, all_acceptance = [], []
    all_baseline_tput, all_spec_tput = [], []
    all_draft_times, all_target_times = [], []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[h5] prompt {prompt_idx+1}/3 | domain={domain} | gamma={gamma}")

        # ── Baseline: 70B NF4 alone ──────────────────────────────────────
        print(f"[h5] Loading target (70B NF4)...")
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            quantization_config=bnb_4bit_config,
            device_map="cuda",
            token=os.environ["HF_TOKEN"],
        )
        baseline_times = []
        for _ in range(N_RUNS):
            r = run_baseline(target_model, tokenizer, prompt,
                             max_new_tokens=MAX_NEW_TOKENS, device="cuda")
            baseline_times.append(r.total_time_s)
        avg_baseline = sum(baseline_times) / len(baseline_times)
        baseline_tput = MAX_NEW_TOKENS / avg_baseline
        print(f"[h5] Baseline: {baseline_tput:.1f} t/s ({avg_baseline:.1f}s)")

        del target_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ── Speculative: 70B NF4 target + 7B fp16 draft ─────────────────
        try:
            print(f"[h5] Loading target (70B NF4) + draft (7B fp16)...")
            target_model = AutoModelForCausalLM.from_pretrained(
                TARGET_MODEL,
                quantization_config=bnb_4bit_config,
                device_map="cuda",
                token=os.environ["HF_TOKEN"],
            )
            draft_model = AutoModelForCausalLM.from_pretrained(
                DRAFT_MODEL,
                torch_dtype=torch.float16,
                device_map="cuda",
                token=os.environ["HF_TOKEN"],
            )

            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            total_gb = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[h5] VRAM after loading both models: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

            spec_times, spec_result = [], None
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
            torch.cuda.synchronize()

        except Exception as e:
            print(f"[h5] WARNING: {e}")
            try:
                del target_model
            except Exception:
                pass
            try:
                del draft_model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            continue

        speedup = avg_baseline / avg_spec
        all_speedups.append(speedup)
        all_acceptance.append(spec_result.acceptance_rate)
        all_baseline_tput.append(baseline_tput)
        all_spec_tput.append(spec_result.output_tokens / avg_spec)
        all_draft_times.append(spec_result.draft_time_s)
        all_target_times.append(spec_result.target_time_s)

        marker = "SPEEDUP ✓" if speedup >= 1.0 else "slowdown"
        print(
            f"[h5] prompt {prompt_idx+1} | speedup={speedup:.3f}x | "
            f"accept={spec_result.acceptance_rate:.2%} | "
            f"d/t={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x | "
            f"base={baseline_tput:.1f} t/s | spec={spec_result.output_tokens/avg_spec:.1f} t/s | [{marker}]"
        )

    if not all_speedups:
        return {"domain": domain, "gamma": gamma, "failed": True,
                "fail_reason": "all prompts failed"}

    avg_speedup    = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "domain": domain,
        "gamma": gamma,
        "speedup": avg_speedup,
        "speedup_std": std,
        "acceptance_rate": avg_acceptance,
        "throughput_baseline": sum(all_baseline_tput) / len(all_baseline_tput),
        "throughput_spec":     sum(all_spec_tput) / len(all_spec_tput),
        "avg_draft_time_s":    sum(all_draft_times) / len(all_draft_times),
        "avg_target_time_s":   sum(all_target_times) / len(all_target_times),
        "draft_target_ratio":  sum(all_draft_times) / max(sum(all_target_times), 1e-8),
        "target_quantization": "NF4-4bit",
        "draft_quantization":  "fp16",
        "target_model": TARGET_MODEL,
        "draft_model":  DRAFT_MODEL,
        "failed": False,
    }

    marker = "SPEEDUP ✓" if avg_speedup >= 1.0 else "SLOWDOWN"
    print(
        f"\n[h5] SUMMARY domain={domain} | gamma={gamma} | "
        f"speedup={avg_speedup:.3f}x±{std:.3f} | "
        f"accept={avg_acceptance:.2%} | "
        f"d/t={result['draft_target_ratio']:.2f}x | [{marker}]"
    )
    return result


@app.local_entrypoint()
def main():
    import time

    print("H5: Llama-2-7B (fp16) draft → Llama-2-70B-chat (NF4 4-bit) target")
    print("VRAM plan: ~35GB target + ~14GB draft + ~10GB KV = ~59GB on A100-80GB\n")

    all_conditions = [
        ("code_completion", 8),
        ("direct_continuation", 8),
        ("general_news", 12),
    ]

    MAX_RETRIES = 3
    results = {}
    failed_final = []

    pending = []
    for domain, gamma in all_conditions:
        handle = run_h5.spawn(domain, gamma)
        pending.append((domain, gamma, handle, 1))
        print(f"  Spawned (attempt 1): domain={domain} | gamma={gamma}")

    print(f"\nAll {len(pending)} jobs spawned. Collecting with retry logic...\n")

    while pending:
        still_pending = []
        for domain, gamma, handle, attempt in pending:
            try:
                r = handle.get(timeout=10800)
                if r.get("failed"):
                    reason = r.get("fail_reason", "unknown")
                    print(f"  FAILED {domain} γ={gamma} attempt {attempt}: {reason}")
                    if attempt < MAX_RETRIES:
                        new_handle = run_h5.spawn(domain, gamma)
                        still_pending.append((domain, gamma, new_handle, attempt + 1))
                        print(f"  → Retrying (attempt {attempt+1}): {domain} γ={gamma}")
                    else:
                        print(f"  → Giving up after {MAX_RETRIES} attempts: {domain} γ={gamma}")
                        failed_final.append((domain, gamma))
                else:
                    results[(domain, gamma)] = r
                    marker = "SPEEDUP ✓" if r["speedup"] >= 1.0 else "slowdown"
                    print(
                        f"  ✓ {domain} | γ={gamma} | "
                        f"{r['speedup']:.3f}x | accept={r['acceptance_rate']:.1%} | "
                        f"d/t={r['draft_target_ratio']:.2f}x | [{marker}]"
                    )
            except Exception as e:
                err = str(e)
                print(f"  EXCEPTION {domain} γ={gamma} attempt {attempt}: {err[:120]}")
                is_transient = any(x in err for x in [
                    "No CUDA GPUs are available",
                    "driver version",
                    "provisioning",
                    "timeout",
                    "Connection lost",
                    "preemption",
                ])
                if attempt < MAX_RETRIES and is_transient:
                    time.sleep(5)
                    new_handle = run_h5.spawn(domain, gamma)
                    still_pending.append((domain, gamma, new_handle, attempt + 1))
                    print(f"  → Retrying (attempt {attempt+1}): {domain} γ={gamma}")
                else:
                    print(f"  → Not retrying: {domain} γ={gamma}")
                    failed_final.append((domain, gamma))

        pending = still_pending
        if pending:
            print(f"\n  {len(pending)} jobs still pending (retries in flight)...\n")

    # ── Print results ──────────────────────────────────────────────────────
    all_results = list(results.values())
    if not all_results:
        print("No results collected.")
        return

    all_results.sort(key=lambda x: (x["domain"], x["gamma"]))

    print("\n=== H5 Results: Llama-2-7B (fp16) → Llama-2-70B-chat (NF4) ===")
    print(f"{'domain':>18} | {'γ':>2} | {'accept':>7} | {'speedup':>8} | {'±std':>5} | "
          f"{'base t/s':>8} | {'spec t/s':>8} | {'d/t':>5} | result")
    print("-" * 100)
    for r in all_results:
        marker = "SPEEDUP ✓" if r["speedup"] >= 1.0 else "slowdown"
        print(
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

    print("\n=== Summary by Gamma ===")
    for gamma in GAMMA_VALUES:
        subset = [r for r in all_results if r["gamma"] == gamma]
        if not subset:
            continue
        avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
        avg_accept  = sum(r["acceptance_rate"] for r in subset) / len(subset)
        avg_dt      = sum(r["draft_target_ratio"] for r in subset) / len(subset)
        n_speedup   = len([r for r in subset if r["speedup"] >= 1.0])
        marker = f"SPEEDUP ✓ ({n_speedup}/{len(subset)})" if n_speedup else "all slowdown"
        print(f"  gamma={gamma:>2}: avg speedup={avg_speedup:.3f}x | "
              f"avg accept={avg_accept:.1%} | avg d/t={avg_dt:.2f}x | [{marker}]")

    print("\n=== Summary by Domain ===")
    for domain in DOMAIN_PROMPTS:
        subset = [r for r in all_results if r["domain"] == domain]
        if not subset:
            continue
        avg_speedup = sum(r["speedup"] for r in subset) / len(subset)
        avg_accept  = sum(r["acceptance_rate"] for r in subset) / len(subset)
        n_speedup   = len([r for r in subset if r["speedup"] >= 1.0])
        marker = f"SPEEDUP ✓ ({n_speedup}/{len(subset)})" if n_speedup else "all slowdown"
        print(f"  {domain}: avg speedup={avg_speedup:.3f}x | avg accept={avg_accept:.1%} | [{marker}]")

    print("\n=== Overall ===")
    avg_speedup = sum(r["speedup"] for r in all_results) / len(all_results)
    avg_accept  = sum(r["acceptance_rate"] for r in all_results) / len(all_results)
    avg_dt      = sum(r["draft_target_ratio"] for r in all_results) / len(all_results)
    n_speedup   = len([r for r in all_results if r["speedup"] >= 1.0])
    print(f"  avg speedup={avg_speedup:.3f}x | avg accept={avg_accept:.1%} | avg d/t={avg_dt:.2f}x")
    print(f"  speedups: {n_speedup}/{len(all_results)} conditions")

    if n_speedup:
        best = max(all_results, key=lambda x: x["speedup"])
        print(f"\n  BEST: {best['speedup']:.3f}x | domain={best['domain']} | "
              f"gamma={best['gamma']} | accept={best['acceptance_rate']:.1%} | "
              f"d/t={best['draft_target_ratio']:.2f}x")
    else:
        print("\n  No speedup observed.")

    if failed_final:
        print(f"\nPermanently failed: {failed_final}")
