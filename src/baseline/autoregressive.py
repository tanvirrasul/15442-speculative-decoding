"""
Baseline autoregressive inference for Llama-2-7B.
Used as the performance reference for all speculative decoding experiments.
"""

import time
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class BaselineResult:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    total_time_s: float
    time_to_first_token_s: float
    throughput_tokens_per_s: float


def load_model(model_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def run_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> BaselineResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warm up
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    first_token_time = None

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    # Approximate TTFT as time for 1 token generation
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        first_token_time = time.perf_counter() - t0

    total_time = end - start
    output_tokens = output_ids.shape[1] - input_len
    output_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return BaselineResult(
        prompt=prompt,
        output=output_text,
        input_tokens=input_len,
        output_tokens=output_tokens,
        total_time_s=total_time,
        time_to_first_token_s=first_token_time,
        throughput_tokens_per_s=output_tokens / total_time,
    )
