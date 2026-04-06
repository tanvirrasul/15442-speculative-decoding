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
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
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
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model(**inputs)
    sync()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    sync()
    end = time.perf_counter()

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        sync()
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
