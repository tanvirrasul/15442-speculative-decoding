"""
experiments/h2_acceptance_rate.py

H2: Find the acceptance rate threshold where speculative decoding transitions
from slowdown to speedup, and how gamma affects that threshold.

Usage:
    /Users/tanvirrasul/Downloads/speculative-decoding/venv/bin/python -m modal run experiments/h2_acceptance_rate.py
"""

import modal

app = modal.App("spec-decoding-h2-acceptance")

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
CONTEXT_LENGTH = 512
MAX_NEW_TOKENS = 256
N_RUNS = 3
GAMMA_VALUES = [1, 2, 4, 6, 8]

DOMAIN_PROMPTS = {
    "low_acceptance_news": [
        "The pathophysiology of septic shock involves a dysregulated host response to infection leading to life-threatening organ dysfunction. The release of pathogen-associated molecular patterns triggers toll-like receptor activation, initiating a cascade of pro-inflammatory cytokines including tumor necrosis factor alpha, interleukin-1 beta, and interleukin-6. This systemic inflammatory response syndrome results in increased vascular permeability, distributive shock, and ultimately multi-organ failure if not treated aggressively with broad-spectrum antibiotics, fluid resuscitation, and vasopressor support.",
        "Quantitative easing represents an unconventional monetary policy tool whereby a central bank purchases government securities or other financial assets to inject liquidity directly into the economy when conventional monetary policy has become ineffective. The transmission mechanism operates through several channels: the portfolio balance effect whereby investors shift toward riskier assets, the signaling channel whereby forward guidance anchors expectations, and the wealth effect whereby rising asset prices stimulate consumption. Critics contend that prolonged QE distorts asset valuations and creates moral hazard.",
        "The Hohfeldian framework for analyzing legal relations distinguishes between four fundamental jural relations: rights and duties, privileges and no-rights, powers and liabilities, and immunities and disabilities. A right in the strict sense correlates with a duty in another party to perform some action or refrain from action. A privilege, conversely, represents the absence of a duty and correlates with a no-right in another party to demand performance.",
    ],
    "foreign_language": [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括视觉感知、语音识别、决策制定和语言翻译。人工智能研究始于20世纪50年代，当时的研究人员开始探索机器是否可以模拟人类思维过程。早期的里程碑包括阿兰·图灵提出的图灵测试，该测试评估机器是否能够表现出与人类无法区分的智能行为。",
        "La inteligencia artificial (IA) es una rama de la informática que se ocupa de crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Estas tareas incluyen el reconocimiento visual, el reconocimiento de voz, la toma de decisiones y la traducción de idiomas. La investigación en IA comenzó en la década de 1950, cuando los investigadores comenzaron a explorar si las máquinas podían simular los procesos de pensamiento humano.",
        "L'intelligence artificielle (IA) est une branche de l'informatique qui s'intéresse à la création de systèmes capables d'effectuer des tâches qui nécessitent normalement une intelligence humaine. Ces tâches comprennent la perception visuelle, la reconnaissance vocale, la prise de décision et la traduction linguistique.",
    ],
    "general_news": [
        "The Federal Reserve announced Wednesday that it would hold interest rates steady for the third consecutive meeting, citing continued uncertainty about the trajectory of inflation and the labor market. Fed Chair Jerome Powell told reporters that while inflation has declined significantly from its peak, policymakers want to see more evidence that price pressures are sustainably moving toward the central bank's two percent target before cutting borrowing costs.",
        "Scientists at the European Space Agency have confirmed that the James Webb Space Telescope has detected water vapor in the atmosphere of a distant exoplanet for the first time with such precision. The discovery, published in the journal Nature Astronomy, represents a significant advance in the search for potentially habitable worlds beyond our solar system.",
        "Global semiconductor manufacturers are accelerating investment in new fabrication facilities amid growing concerns about supply chain resilience and geopolitical tensions. Taiwan Semiconductor Manufacturing Company, the world's largest contract chipmaker, announced plans to expand its operations in Arizona and Japan, while Intel and Samsung are also pursuing major construction projects in the United States and Europe.",
    ],
    "code_completion": [
        """import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    ],
    "repetitive_template": [
        """Name: Alice Johnson | Age: 28 | Department: Engineering | Salary: $125,000 | Start Date: 2020-03-15 | Manager: Bob Smith | Location: San Francisco
Name: Bob Smith | Age: 45 | Department: Management | Salary: $185,000 | Start Date: 2015-01-10 | Manager: Carol White | Location: San Francisco
Name: Carol White | Age: 52 | Department: Executive | Salary: $250,000 | Start Date: 2010-06-01 | Manager: None | Location: New York
Name: Dave Brown | Age: 31 | Department: Marketing | Salary: $95,000 | Start Date: 2021-08-20 | Manager: Eve Davis | Location: Chicago
Name: Eve Davis | Age: 38 | Department: Marketing | Salary: $140,000 | Start Date: 2018-04-12 | Manager: Carol White | Location: Chicago""",
        """Step 1: Open the application and navigate to the Settings menu.
Step 2: Click on the Account tab and select Personal Information.
Step 3: Update your first name in the First Name field.
Step 4: Update your last name in the Last Name field.
Step 5: Enter your new email address in the Email field.
Step 6: Confirm your email address in the Confirm Email field.
Step 7: Update your phone number in the Phone Number field.
Step 8: Select your country from the Country dropdown menu.
Step 9: Enter your city in the City field.
Step 10: Enter your postal code in the Postal Code field.""",
        """Product: Laptop Pro X1 | SKU: LP-X1-001 | Price: $1299.99 | Stock: 45 | Category: Electronics | Brand: TechCorp
Product: Wireless Mouse M200 | SKU: WM-200-002 | Price: $29.99 | Stock: 230 | Category: Peripherals | Brand: TechCorp
Product: USB-C Hub H400 | SKU: UCH-400-003 | Price: $49.99 | Stock: 180 | Category: Accessories | Brand: TechCorp
Product: Monitor 27inch K1 | SKU: MN-27K-004 | Price: $399.99 | Stock: 62 | Category: Displays | Brand: TechCorp
Product: Keyboard Mech K500 | SKU: KB-M-005 | Price: $89.99 | Stock: 95 | Category: Peripherals | Brand: TechCorp""",
    ],
    "direct_continuation": [
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
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


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_h2(domain: str, gamma: int):
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

    raw_prompts = DOMAIN_PROMPTS[domain]
    prompts = [pad_or_trim(p, CONTEXT_LENGTH, tokenizer_only) for p in raw_prompts]

    config = SpecDecodingConfig(
        gamma=gamma,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose_timing=False,
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
        print(f"[h2] domain={domain} | gamma={gamma} | prompt {prompt_idx+1}/{len(prompts)} | tokens={actual_tokens}")

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

        # Speculative — wrapped in try/except to handle GPU errors gracefully
        try:
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
        except Exception as e:
            print(f"[h2] WARNING: spec run failed domain={domain} gamma={gamma} prompt={prompt_idx+1}: {e}")
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

        print(f"[h2] domain={domain} | gamma={gamma} | prompt {prompt_idx+1} | speedup={speedup:.3f}x | acceptance={spec_result.acceptance_rate:.2%} | draft/target={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x")

    if not all_speedups:
        print(f"[h2] WARNING: all prompts failed for domain={domain} gamma={gamma}, skipping")
        return {
            "domain": domain, "gamma": gamma, "speedup": 0.0, "speedup_std": 0.0,
            "acceptance_rate": 0.0, "throughput_baseline": 0.0, "throughput_spec": 0.0,
            "avg_draft_time_s": 0.0, "avg_target_time_s": 0.0, "draft_target_ratio": 0.0,
            "gpu": get_gpu_name(), "failed": True,
        }

    avg_speedup = sum(all_speedups) / len(all_speedups)
    avg_acceptance = sum(all_acceptance) / len(all_acceptance)
    std = (sum((s - avg_speedup)**2 for s in all_speedups) / len(all_speedups)) ** 0.5

    result = {
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
        "gpu": get_gpu_name(),
        "failed": False,
    }

    marker = "SPEEDUP" if avg_speedup >= 1.0 else "SLOWDOWN"
    print(f"\n[h2] SUMMARY domain={domain} | gamma={gamma} | speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | [{marker}]")
    return result


@app.local_entrypoint()
def main():
    futures = []
    for domain in DOMAIN_PROMPTS:
        for gamma in GAMMA_VALUES:
            futures.append(run_h2.remote(domain, gamma))

    results = [f for f in futures]
    results = [r for r in results if not r.get("failed", False)]
    results.sort(key=lambda x: x["acceptance_rate"])

    print("\n=== H2 Results: Speedup vs Acceptance Rate x Gamma ===")
    print(f"{'domain':>22} | {'gamma':>5} | {'accept':>8} | {'speedup':>8} | {'±std':>6} | {'baseline':>10} | {'spec':>10} | {'d/t ratio':>10} | result")
    print("-" * 115)
    for r in results:
        marker = "SPEEDUP" if r["speedup"] >= 1.0 else "SLOWDOWN"
        print(
            f"  {r['domain']:>20} | "
            f"{r['gamma']:>5} | "
            f"{r['acceptance_rate']:>7.2%} | "
            f"{r['speedup']:>7.3f}x | "
            f"{r['speedup_std']:>5.3f} | "
            f"{r['throughput_baseline']:>8.1f} t/s | "
            f"{r['throughput_spec']:>8.1f} t/s | "
            f"{r['draft_target_ratio']:>9.2f}x | "
            f"[{marker}]"
        )

    print("\n=== Gamma Sweep Summary ===")
    for gamma in GAMMA_VALUES:
        gamma_results = [r for r in results if r["gamma"] == gamma]
        gamma_results.sort(key=lambda x: x["acceptance_rate"])
        speedups_above_1 = [r for r in gamma_results if r["speedup"] >= 1.0]
        if speedups_above_1:
            threshold = min(speedups_above_1, key=lambda x: x["acceptance_rate"])
            print(f"  gamma={gamma}: break-even at ~{threshold['acceptance_rate']:.0%} acceptance ({threshold['domain']})")
        else:
            print(f"  gamma={gamma}: no speedup observed across all domains")
