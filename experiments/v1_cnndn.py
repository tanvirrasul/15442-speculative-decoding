"""
experiments/.py

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
    "datasets>=2.18.0",
).add_local_dir("./src", remote_path="/root/src")

volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TARGET_MODEL = "google/t5-v1_1-xxl"
CONTEXT_LENGTH = 512
MAX_NEW_TOKENS = 256
N_RUNS = 3
GAMMA_TEMP_CONFIG = {
    "t5-large": [(3, 0.0), (3, 1.0)],
    "t5-base": [(3, 0.0), (5, 1.0)],
    "t5-small": [(5, 0.0), (5, 1.0)]
}
DRAFT_MODELS = {
    "t5-large": "google-t5/t5-large",
    "t5-base": "google-t5/t5-base",
    "t5-small": "google-t5/t5-small",
}
PROMPTS_PER_LENGTH = 5

def get_real_prompts(target_length: int, tokenizer, n: int = 3) -> list[str]:
    from datasets import load_dataset

    prompts = []

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
    gpu="H100",
    timeout=7200,
    volumes={"/model-weights": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_v1(draft_label : str, gamma : int, temp : float):
    import os
    import gc
    import torch
    from transformers import AutoTokenizer
    from src.baseline.encdec import load_s2s_model, run_s2s_baseline
    from src.speculative.s2s_spec_decode import load_s2s_models, run_s2s_speculative
    from src.speculative.spec_decode import SpecDecodingConfig
    from src.profiling.metrics import get_gpu_name
    
    draft_model_id = DRAFT_MODELS[draft_label]
    print(f"[v1] draft={draft_label} | gamma={gamma} | temp={temp} | target model={TARGET_MODEL}")
    
    tokenizer_only = AutoTokenizer.from_pretrained(
        TARGET_MODEL, token=os.environ["HF_TOKEN"]
    )

    prompts = get_real_prompts(CONTEXT_LENGTH, tokenizer_only, n=PROMPTS_PER_LENGTH)
    
    config = SpecDecodingConfig(
        gamma=gamma,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose_timing=True,
        temperature=temp,
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
        print(f"[v1] draft={draft_label} | prompt {prompt_idx+1}/{len(prompts)} | tokens={actual_tokens}")

        # Baseline
        base_model, base_tok = load_s2s_model(TARGET_MODEL, device="cuda")
        baseline_times = []
        for _ in range(N_RUNS):
            r = run_s2s_baseline(base_model, base_tok, prompt, max_new_tokens=MAX_NEW_TOKENS, device="cuda")
            baseline_times.append(r.total_time_s)
        avg_baseline = sum(baseline_times) / len(baseline_times)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Speculative
        try:
            target_model, draft_model, tokenizer = load_s2s_models(
                TARGET_MODEL, draft_model_id, device="cuda"
            )
            spec_times = []
            spec_result = None
            for _ in range(N_RUNS):
                r = run_s2s_speculative(
                    target_model, draft_model, tokenizer, prompt, config, device="cuda"
                )
                spec_times.append(r.total_time_s)
                spec_result = r
            avg_spec = sum(spec_times) / len(spec_times)
            del target_model, draft_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[v1] WARNING: spec run failed draft={draft_label} gamma={gamma} temp={temp} prompt={prompt_idx+1}: {e}")
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

        print(f"[v1] draft={draft_label} | gamma={gamma} | temp={temp} | prompt {prompt_idx+1} | speedup={speedup:.3f}x | acceptance={spec_result.acceptance_rate:.2%} | draft/target={spec_result.draft_time_s/max(spec_result.target_time_s,1e-8):.2f}x")
    
    if not all_speedups:
        print(f"[v1] WARNING: all prompts failed for draft={draft_label} gamma={gamma} | temp={temp}")
        return {
            "draft_label": draft_label,
            "draft_model": draft_model_id,
            "gamma": gamma,
            "temp": temp,
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
        "gamma": gamma,
        "temp": temp,
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
    print(f"\n[v1] SUMMARY draft={draft_label} | gamma={gamma} | temp={temp} | speedup={avg_speedup:.3f}x±{std:.3f} | acceptance={avg_acceptance:.2%} | d/t={result['draft_target_ratio']:.2f}x | [{marker}]")
    return result

@app.local_entrypoint()
def main():
    futures = []
    for draft_label in DRAFT_MODELS:
        for gamma, temp in GAMMA_TEMP_CONFIG[draft_label]:
            futures.append(run_v1.remote(draft_label, gamma, temp))
    results = [f for f in futures]
    results = [r for r in results if not r.get("failed", False)]
    
    print("\n=== V1 Results: Verification of T5X acceleration with CNN/DN ===")
    print(f"{'draft':>18} | {'gamma':>7} | {'temp':>7} | {'accept':>8} | {'speedup':>8} | {'±std':>6} | {'baseline':>10} | {'spec':>10} | {'d/t ratio':>10} | result")
    print("-" * 120)
    for r in results:
        marker = "SPEEDUP" if r["speedup"] >= 1.0 else "SLOWDOWN"
        print(
            f"  {r['draft_label']:>16} | "
            f"{r['gamma']:>7} | "
            f"{r['temp']:>7} | "
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