"""
experiments/h4_a10g_remaining.py

A10G llama2_7b_chat: loads draft in int8 to fit both 7B models in 22GB VRAM.
Llama-2-7B (target, fp16) ~13GB + Llama-2-7B-Chat (draft, int8) ~7GB = ~20GB.

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run --detach experiments/h4_a10g_remaining.py
"""

import os
import modal

app = modal.App("spec-decoding-h4-a10g-remaining-v2")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
    "bitsandbytes>=0.41.0",  # Required for int8 quantization
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL    = "meta-llama/Llama-2-7b-hf"
DRAFT_MODEL_ID  = "meta-llama/Llama-2-7b-chat-hf"
DRAFT_LABEL     = "llama2_7b_chat"
GAMMA_VALUES    = [1, 4, 8]
MAX_NEW_TOKENS  = 256
CONTEXT_LENGTH  = 512
N_RUNS          = 3

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
    gpu="A10G",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_a10g_llama2_chat(domain: str, gamma: int):
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from src.baseline.autoregressive import load_model, run_baseline
    from src.speculative.spec_decode import run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_name = get_gpu_name()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[h4-a10g] gpu={gpu_name} ({total_vram:.0f}GB) | draft={DRAFT_LABEL} | domain={domain} | gamma={gamma}")

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

    # Load draft in int8 to fit both 7B models in 22GB
    # Target (fp16) ~13GB + Draft (int8) ~7GB = ~20GB — fits A10G
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    all_speedups, all_acceptance = [], []
    all_baseline_tput, all_spec_tput = [], []
    all_draft_times, all_target_times = [], []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"[h4-a10g] prompt {prompt_idx+1}/3")

        # Baseline (fp16 target)
        base_model, base_tok = load_model(TARGET_MODEL, device="cuda")
        baseline_times = []
        for _ in range(N_RUNS):
            r = run_baseline(base_model, base_tok, prompt, max_new_tokens=MAX_NEW_TOKENS, device="cuda")
            baseline_times.append(r.total_time_s)
        avg_baseline = sum(baseline_times) / len(baseline_times)
        baseline_tput = MAX_NEW_TOKENS / avg_baseline
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Speculative (fp16 target + int8 draft)
        try:
            # Manual load: target fp16, draft int8
            target_model = AutoModelForCausalLM.from_pretrained(
                TARGET_MODEL,
                torch_dtype=torch.float16,
                device_map="cuda",
                token=os.environ["HF_TOKEN"],
            )
            draft_model = AutoModelForCausalLM.from_pretrained(
                DRAFT_MODEL_ID,
                quantization_config=bnb_config,
                device_map="cuda",
                token=os.environ["HF_TOKEN"],
            )

            spec_times, spec_result = [], None
            for _ in range(N_RUNS):
                r = run_speculative(
                    target_model, draft_model, tokenizer_only, prompt, config, device="cuda"
                )
                spec_times.append(r.total_time_s)
                spec_result = r
            avg_spec = sum(spec_times) / len(spec_times)

            del target_model, draft_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[h4-a10g] WARNING: {e}")
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
        all_baseline_tput.append(baseline_tput)
        all_spec_tput.append(spec_result.output_tokens / avg_spec)
        all_draft_times.append(spec_result.draft_time_s)
        all_target_times.append(spec_result.target_time_s)

        marker = "SPEEDUP" if speedup >= 1.0 else "slowdown"
        print(
            f"[h4-a10g] {domain} γ={gamma} prompt {prompt_idx+1} | "
            f"speedup={speedup:.3f}x | accept={spec_result.acceptance_rate:.2%} | "
            f"d/t={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x | [{marker}]"
        )

    if not all_speedups:
        return {"gpu": gpu_name, "draft_label": DRAFT_LABEL, "domain": domain,
                "gamma": gamma, "failed": True, "fail_reason": "all prompts failed"}

    avg_speedup    = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "gpu": gpu_name,
        "draft_label": DRAFT_LABEL,
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
        "quantized_draft": True,  # Note: draft loaded in int8
        "failed": False,
    }

    marker = "SPEEDUP" if avg_speedup >= 1.0 else "SLOWDOWN"
    print(
        f"\n[h4-a10g] SUMMARY {domain} γ={gamma} | "
        f"speedup={avg_speedup:.3f}x±{std:.3f} | "
        f"accept={avg_acceptance:.2%} | "
        f"d/t={result['draft_target_ratio']:.2f}x | [{marker}]"
    )
    return result


@app.local_entrypoint()
def main():
    print("NOTE: llama2_7b_chat draft loaded in int8 to fit A10G (22GB).")
    print("      Target (fp16 ~13GB) + Draft (int8 ~7GB) = ~20GB\n")

    job_specs = []
    for domain in DOMAIN_PROMPTS:
        for gamma in GAMMA_VALUES:
            handle = run_a10g_llama2_chat.spawn(domain, gamma)
            job_specs.append((domain, gamma, handle))
            print(f"  Spawned: A10G | llama2_7b_chat (int8) | {domain} | gamma={gamma}")

    print(f"\nAll {len(job_specs)} jobs spawned. Collecting...\n")

    results, failed = [], []
    for i, (domain, gamma, handle) in enumerate(job_specs):
        try:
            r = handle.get(timeout=7200)
            if r.get("failed"):
                print(f"[{i+1}/{len(job_specs)}] FAILED | {domain} | γ={gamma} | {r.get('fail_reason','')}")
                failed.append((domain, gamma))
            else:
                results.append(r)
                marker = "SPEEDUP" if r["speedup"] >= 1.0 else "slowdown"
                print(
                    f"[{i+1}/{len(job_specs)}] A10G | llama2_7b_chat | {domain} | γ={gamma} | "
                    f"{r['speedup']:.3f}x | accept={r['acceptance_rate']:.1%} | "
                    f"d/t={r['draft_target_ratio']:.2f}x | [{marker}]"
                )
        except Exception as e:
            print(f"[{i+1}/{len(job_specs)}] EXCEPTION | {domain} | γ={gamma}: {e}")
            failed.append((domain, gamma))

    if not results:
        print("No results.")
        return

    results.sort(key=lambda x: (x["domain"], x["gamma"]))

    print("\n=== A10G llama2_7b_chat (int8 draft) Results ===")
    print(f"{'domain':>18} | {'γ':>2} | {'accept':>7} | {'speedup':>8} | {'±std':>5} | {'base t/s':>8} | {'spec t/s':>8} | {'d/t':>5} | result")
    print("-" * 92)
    for r in results:
        marker = "SPEEDUP" if r["speedup"] >= 1.0 else "slowdown"
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

    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        avg_dt      = sum(r["draft_target_ratio"] for r in results) / len(results)
        avg_accept  = sum(r["acceptance_rate"] for r in results) / len(results)
        n_speedup   = len([r for r in results if r["speedup"] >= 1.0])
        print(f"\n  Summary: avg speedup={avg_speedup:.3f}x | avg d/t={avg_dt:.2f}x | "
              f"avg accept={avg_accept:.1%} | speedups={n_speedup}/{len(results)}")
        print("  NOTE: results use int8 draft — d/t ratio reflects quantized inference speed")