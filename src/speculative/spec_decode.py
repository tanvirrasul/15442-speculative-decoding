"""
Core speculative decoding implementation.

Follows the algorithm from Leviathan et al. (2023):
  - Draft model proposes `gamma` tokens autoregressively
  - Target model verifies all gamma tokens in a single forward pass
  - Tokens are accepted/rejected based on the ratio of target/draft probabilities
  - On rejection, resample from a corrected distribution and discard remaining draft tokens
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SpecDecodingResult:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    total_time_s: float
    time_to_first_token_s: float
    throughput_tokens_per_s: float
    # Speculative decoding specific
    acceptance_rate: float
    total_draft_tokens: int
    total_accepted_tokens: int
    num_target_calls: int
    speedup_vs_baseline: float = 0.0  # filled in externally after baseline run


@dataclass
class SpecDecodingConfig:
    gamma: int = 4                  # number of draft tokens to propose per step
    temperature: float = 1.0        # sampling temperature (1.0 = greedy-equivalent for top-1)
    max_new_tokens: int = 256


def load_models(
    target_model_id: str,
    draft_model_id: str,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    target_model.eval()
    draft_model.eval()
    return target_model, draft_model, tokenizer


def _sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample a token from a probability distribution."""
    return torch.multinomial(probs, num_samples=1)


def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    config: SpecDecodingConfig,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """
    Run speculative decoding and return (output_ids, stats).

    Stats dict contains:
        - total_draft_tokens: total number of tokens proposed by draft
        - total_accepted_tokens: total accepted by target
        - num_target_calls: number of target model forward passes
    """
    generated = input_ids.clone()
    total_draft = 0
    total_accepted = 0
    num_target_calls = 0

    with torch.no_grad():
        while generated.shape[1] - input_ids.shape[1] < config.max_new_tokens:
            # --- Draft phase: propose gamma tokens ---
            draft_ids = generated.clone()
            draft_probs = []

            for _ in range(config.gamma):
                out = draft_model(draft_ids)
                logits = out.logits[:, -1, :]
                probs = F.softmax(logits / config.temperature, dim=-1)
                next_token = _sample(probs)
                draft_probs.append(probs)
                draft_ids = torch.cat([draft_ids, next_token], dim=-1)

            total_draft += config.gamma

            # --- Verify phase: single target forward pass over all draft tokens ---
            target_out = target_model(draft_ids)
            num_target_calls += 1

            # Target probabilities for each position (offset by 1 since we want p(x_{t+1}|x_{1:t}))
            target_logits = target_out.logits[:, generated.shape[1] - 1 : generated.shape[1] + config.gamma - 1, :]
            target_probs_all = F.softmax(target_logits / config.temperature, dim=-1)

            # --- Acceptance / rejection ---
            accepted_count = 0
            for i in range(config.gamma):
                draft_token = draft_ids[:, generated.shape[1] + i]
                p = target_probs_all[:, i, draft_token].squeeze()
                q = draft_probs[i][:, draft_token].squeeze()

                acceptance_prob = torch.clamp(p / (q + 1e-8), max=1.0)
                u = torch.rand(1, device=device)

                if u <= acceptance_prob:
                    accepted_count += 1
                else:
                    # Rejection: resample from corrected distribution and stop
                    corrected = torch.clamp(target_probs_all[:, i, :] - draft_probs[i], min=0)
                    corrected = corrected / corrected.sum()
                    resampled = _sample(corrected)
                    generated = torch.cat([generated, draft_ids[:, generated.shape[1]: generated.shape[1] + i], resampled], dim=-1)
                    total_accepted += accepted_count
                    break
            else:
                # All gamma tokens accepted: append + sample one bonus token from target
                bonus_logits = target_out.logits[:, generated.shape[1] + config.gamma - 1, :]
                bonus_probs = F.softmax(bonus_logits / config.temperature, dim=-1)
                bonus_token = _sample(bonus_probs)
                generated = torch.cat([generated, draft_ids[:, generated.shape[1]: generated.shape[1] + config.gamma], bonus_token], dim=-1)
                total_accepted += config.gamma

            # Check for EOS
            if generated[0, -1].item() == draft_model.config.eos_token_id:
                break

    stats = {
        "total_draft_tokens": total_draft,
        "total_accepted_tokens": total_accepted,
        "num_target_calls": num_target_calls,
        "acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0.0,
    }
    return generated, stats


def run_speculative(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    config: SpecDecodingConfig,
    device: str = "cuda",
) -> SpecDecodingResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warm up
    _ = target_model(inputs["input_ids"])
    _ = draft_model(inputs["input_ids"])
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    output_ids, stats = speculative_decode(
        target_model, draft_model, inputs["input_ids"], config, device
    )
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    # TTFT: time for target model single forward pass
    t0 = time.perf_counter()
    _ = target_model(inputs["input_ids"])
    if device == "cuda":
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    total_time = end - start
    output_tokens = output_ids.shape[1] - input_len
    output_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return SpecDecodingResult(
        prompt=prompt,
        output=output_text,
        input_tokens=input_len,
        output_tokens=output_tokens,
        total_time_s=total_time,
        time_to_first_token_s=ttft,
        throughput_tokens_per_s=output_tokens / total_time,
        acceptance_rate=stats["acceptance_rate"],
        total_draft_tokens=stats["total_draft_tokens"],
        total_accepted_tokens=stats["total_accepted_tokens"],
        num_target_calls=stats["num_target_calls"],
    )
