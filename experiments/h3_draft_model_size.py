"""
experiments/h3_draft_model_size.py

H3: Does draft model size/family alignment fix the slowdown observed in H1/H2?

Fix: target=Llama-2-7B, ctx=512, gamma=4, A100
Vary: draft model (size + family alignment)

Draft models tested:
  - TinyLlama-1.1B-Chat     (1.1B, different family) -- baseline already measured
  - Llama-2-7B itself        (7B, same model)         -- oracle upper bound
  - Sheared-LLaMA-2.7B      (2.7B, pruned from Llama-2) -- same family, medium size
  - Llama-2-7B-Chat         (7B, same family, RLHF)   -- same family, chat-tuned

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run experiments/h3_draft_model_size.py
"""

import modal

app = modal.App("spec-decoding-h3-draft-model")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.2.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "meta-llama/Llama-2-7b-hf"
CONTEXT_LENGTH = 512
MAX_NEW_TOKENS = 256
N_RUNS = 3
GAMMA = 4  # fixed at gamma=4 based on H2 findings

DRAFT_MODELS = {
    "tinyllama_1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "sheared_llama_2b": "princeton-nlp/Sheared-LLaMA-2.7B",
    "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_7b_self": "meta-llama/Llama-2-7b-hf",  # oracle: same model as target
}

# Use one domain per acceptance tier based on H2 findings
# general_news: ~60% acceptance (realistic workload)
# direct_continuation: ~50% acceptance (high-repetition)
TEST_PROMPTS = {
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
    gpu="A100",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h3(draft_label: str, domain: str):
    import os
    import gc
    import torch
    from transformers import AutoTokenizer
    from src.baseline.autoregressive import load_model, run_baseline
    from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name

    draft_model_id = DRAFT_MODELS[draft_label]
    print(f"[h3] draft={draft_label} | domain={domain} | target={TARGET_MODEL}")

    tokenizer_only = AutoTokenizer.from_pretrained(
        TARGET_MODEL, token=os.environ["HF_TOKEN"]
    )

    raw_prompts = TEST_PROMPTS[domain]
    prompts = [pad_or_trim(p, CONTEXT_LENGTH, tokenizer_only) for p in raw_prompts]

    config = SpecDecodingConfig(
        gamma=GAMMA,
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
        print(f"[h3] draft={draft_label} | domain={domain} | prompt {prompt_idx+1}/{len(prompts)} | tokens={actual_tokens}")

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
            print(f"[h3] WARNING: spec run failed draft={draft_label} domain={domain} prompt={prompt_idx+1}: {e}")
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

        print(f"[h3] draft={draft_label} | domain={domain} | prompt {prompt_idx+1} | speedup={speedup:.3f}x | acceptance={spec_result.acceptance_rate:.2%} | draft/target={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x")

    if not all_speedups:
        print(f"[h3] WARNING: all prompts failed for draft={draft_label} domain={domain}")
        return {
            "draft_label": draft_label,
            "draft_model": draft_model_id,
            "domain": domain,
            "speedup": 0.0, "speedup_std": 0.0,
            "acceptance_rate": 0.0, "throughput_baseline": 0.0, "throughput_spec": 0.0,
            "avg_draft_time_s": 0.0, "avg_target_time_s": 0.0, "draft_target_ratio": 0.0,
            "gpu": get_gpu_name(), "failed": True,
        }

    avg_speedup = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
        "draft_label": draft_label,
        "draft_model": draft_model_id,
        "domain": domain,
        "gamma": GAMMA,
        "speedup": avg_speedup,
        "speedup_std": std,
        "acceptance_rate": avg_acceptance,
        "throughput_baseline": sum(all_baseline_tput) / len(all_baseline_tput),
        "throughput_spec": sum(all_spec_tput) / len(all_spec_tput),
        "avg_draft_time_s": sum(all_draft_times) / len(all_draft_times),
        "avg_target_time_s": sum(all_target_times) / len(all_target_times),
        "draft_target_ratio": sum(all_draft_times) / max(sum(all_target_times), 1e-8),
        "gpu": get_gpu_name(),
        "failed": False,
    }

    marker = "SPEEDUP" if avg_speedup >= 1.0 else "SLOWDOWN"
    print(f"\n[h3] SUMMARY draft={draft_label} | domain={domain} | speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | d/t={result['draft_target_ratio']:.2f}x | [{marker}]")
    return result


@app.local_entrypoint()
def main():
    futures = []
    for draft_label in DRAFT_MODELS:
        for domain in TEST_PROMPTS:
            futures.append(run_h3.remote(draft_label, domain))

    results = [f for f in futures]
    results = [r for r in results if not r.get("failed", False)]
    results.sort(key=lambda x: x["acceptance_rate"])

    print("\n=== H3 Results: Draft Model Size / Family Alignment ===")
    print(f"{'draft':>18} | {'domain':>14} | {'accept':>8} | {'speedup':>8} | {'±std':>6} | {'baseline':>10} | {'spec':>10} | {'d/t ratio':>10} | result")
    print("-" * 120)
    for r in results:
        marker = "SPEEDUP" if r["speedup"] >= 1.0 else "SLOWDOWN"
        print(
            f"  {r['draft_label']:>16} | "
            f"{r['domain']:>14} | "
            f"{r['acceptance_rate']:>7.2%} | "
            f"{r['speedup']:>7.3f}x | "
            f"{r['speedup_std']:>5.3f} | "
            f"{r['throughput_baseline']:>8.1f} t/s | "
            f"{r['throughput_spec']:>8.1f} t/s | "
            f"{r['draft_target_ratio']:>9.2f}x | "
            f"[{marker}]"
        )

    print("\n=== Draft Model Summary (averaged across domains) ===")
    for draft_label in DRAFT_MODELS:
        dr = [r for r in results if r["draft_label"] == draft_label]
        if not dr:
            continue
        avg_speedup = sum(r["speedup"] for r in dr) / len(dr)
        avg_acceptance = sum(r["acceptance_rate"] for r in dr) / len(dr)
        avg_dt_ratio = sum(r["draft_target_ratio"] for r in dr) / len(dr)
        marker = "SPEEDUP" if avg_speedup >= 1.0 else "SLOWDOWN"
        print(f"  {draft_label:>18} | avg speedup={avg_speedup:.3f}x | avg acceptance={avg_acceptance:.2%} | avg d/t={avg_dt_ratio:.2f}x | [{marker}]")
