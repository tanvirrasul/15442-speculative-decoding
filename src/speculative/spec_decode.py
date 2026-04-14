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
from dataclasses import dataclass
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
    acceptance_rate: float
    total_draft_tokens: int
    total_accepted_tokens: int
    num_target_calls: int
    speedup_vs_baseline: float = 0.0
    draft_time_s: float = 0.0
    target_time_s: float = 0.0
    overhead_time_s: float = 0.0


@dataclass
class SpecDecodingConfig:
    gamma: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 256
    verbose_timing: bool = False


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


def _sample(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample a token. Input: [1, vocab_size]. Output: [1, 1].
    If temperature == 0.0, use greedy (argmax) instead of multinomial.
    """
    if temperature == 0.0:
        return probs.argmax(dim=-1, keepdim=True)  # [1, 1]
    return torch.multinomial(probs, num_samples=1)  # [1, 1]


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    config: SpecDecodingConfig,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    generated = input_ids.clone()
    total_draft = 0
    total_accepted = 0
    num_target_calls = 0
    total_draft_time = 0.0
    total_target_time = 0.0
    total_overhead_time = 0.0

    with torch.no_grad():
        # Pre-fill draft KV cache with the prompt
        t0 = time.perf_counter()
        draft_out = draft_model(generated, use_cache=True)
        draft_past_kv = draft_out.past_key_values
        _sync(device)
        total_draft_time += time.perf_counter() - t0

        while generated.shape[1] - input_ids.shape[1] < config.max_new_tokens:

            # --- Draft phase: propose gamma tokens using KV cache ---
            t_draft_start = time.perf_counter()
            draft_probs = []
            draft_tokens = []  # each [1, 1]
            current_past_kv = draft_past_kv
            last_token = generated[:, -1:]  # [1, 1]

            for _ in range(config.gamma):
                out = draft_model(
                    last_token,
                    past_key_values=current_past_kv,
                    use_cache=True
                )
                logits = out.logits[:, -1, :]  # [1, vocab_size]

                if config.temperature == 0.0:
                    # Greedy: sharp distribution centered on argmax
                    next_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
                    # Build a one-hot prob vector for acceptance check
                    probs = torch.zeros_like(logits)
                    probs.scatter_(1, next_token, 1.0)
                else:
                    probs = F.softmax(logits / config.temperature, dim=-1)
                    next_token = _sample(probs, config.temperature)  # [1, 1]

                draft_probs.append(probs)
                draft_tokens.append(next_token)
                last_token = next_token
                current_past_kv = out.past_key_values

            _sync(device)
            total_draft_time += time.perf_counter() - t_draft_start
            total_draft += config.gamma

            # Assemble draft sequence
            t_overhead_start = time.perf_counter()
            draft_token_tensor = torch.cat(draft_tokens, dim=1)  # [1, gamma]
            draft_ids = torch.cat([generated, draft_token_tensor], dim=1)  # [1, seq+gamma]
            total_overhead_time += time.perf_counter() - t_overhead_start

            # --- Verify phase: single target forward pass ---
            t_target_start = time.perf_counter()
            target_out = target_model(draft_ids)
            _sync(device)
            total_target_time += time.perf_counter() - t_target_start
            num_target_calls += 1

            # Extract target probs at each draft position
            t_overhead_start = time.perf_counter()
            target_logits = target_out.logits[
                :,
                generated.shape[1] - 1: generated.shape[1] + config.gamma - 1,
                :
            ]  # [1, gamma, vocab_size]

            if config.temperature == 0.0:
                # Greedy target: one-hot at argmax
                target_probs_all = torch.zeros_like(target_logits)
                argmax_ids = target_logits.argmax(dim=-1, keepdim=True)  # [1, gamma, 1]
                target_probs_all.scatter_(2, argmax_ids, 1.0)
            else:
                target_probs_all = F.softmax(
                    target_logits / config.temperature, dim=-1
                )

            # --- Acceptance / rejection ---
            accepted_count = 0
            rejected_at = None

            for i in range(config.gamma):
                token_idx = draft_tokens[i].squeeze().item()
                p = target_probs_all[0, i, token_idx]
                q = draft_probs[i][0, token_idx]

                if config.temperature == 0.0:
                    # Greedy: accept iff both models agree on the same token
                    accepted = (p.item() == 1.0)
                else:
                    acceptance_prob = torch.clamp(p / (q + 1e-8), max=1.0)
                    u = torch.rand(1, device=device).item()
                    accepted = (u <= acceptance_prob.item())

                if accepted:
                    accepted_count += 1
                else:
                    rejected_at = i
                    if config.temperature == 0.0:
                        # Greedy: take target's argmax at this position
                        resampled = target_logits[0, i].argmax(dim=-1).reshape(1, 1)
                    else:
                        corrected = torch.clamp(
                            target_probs_all[0, i, :] - draft_probs[i][0, :], min=0
                        )
                        corrected = corrected / (corrected.sum() + 1e-8)
                        resampled = torch.multinomial(
                            corrected.unsqueeze(0), num_samples=1
                        )
                    accepted_draft = draft_token_tensor[:, :i]
                    generated = torch.cat([generated, accepted_draft, resampled], dim=1)
                    total_accepted += accepted_count
                    break

            if rejected_at is None:
                bonus_logits = target_out.logits[
                    :, generated.shape[1] + config.gamma - 1, :
                ]
                if config.temperature == 0.0:
                    bonus_token = bonus_logits.argmax(dim=-1, keepdim=True)
                else:
                    bonus_probs = F.softmax(
                        bonus_logits / config.temperature, dim=-1
                    )
                    bonus_token = _sample(bonus_probs, config.temperature)
                generated = torch.cat(
                    [generated, draft_token_tensor, bonus_token], dim=1
                )
                total_accepted += config.gamma

            total_overhead_time += time.perf_counter() - t_overhead_start

            # --- KV cache update ---
            t_draft_start = time.perf_counter()
            if rejected_at is None:
                # All accepted: just append bonus token to existing cache
                draft_out = draft_model(
                    bonus_token,
                    past_key_values=current_past_kv,
                    use_cache=True
                )
                draft_past_kv = draft_out.past_key_values
            else:
                # Rejected: full reprocess (unavoidable)
                draft_out = draft_model(generated, use_cache=True)
                draft_past_kv = draft_out.past_key_values
            _sync(device)
            total_draft_time += time.perf_counter() - t_draft_start

            # Check EOS
            last_token_id = generated[0, -1].item()
            eos_ids = [
                draft_model.config.eos_token_id,
                target_model.config.eos_token_id,
            ]
            if last_token_id in eos_ids:
                break

    if config.verbose_timing:
        total = total_draft_time + total_target_time + total_overhead_time
        print(f"\n[spec_decode timing]")
        print(f"  Draft time   : {total_draft_time:.3f}s ({100*total_draft_time/max(total,1e-8):.1f}%)")
        print(f"  Target time  : {total_target_time:.3f}s ({100*total_target_time/max(total,1e-8):.1f}%)")
        print(f"  Overhead     : {total_overhead_time:.3f}s ({100*total_overhead_time/max(total,1e-8):.1f}%)")
        print(f"  Draft/Target ratio: {total_draft_time/max(total_target_time,1e-8):.2f}x")

    stats = {
        "total_draft_tokens": total_draft,
        "total_accepted_tokens": total_accepted,
        "num_target_calls": num_target_calls,
        "acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0.0,
        "draft_time_s": total_draft_time,
        "target_time_s": total_target_time,
        "overhead_time_s": total_overhead_time,
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
    with torch.no_grad():
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

    with torch.no_grad():
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
        draft_time_s=stats["draft_time_s"],
        target_time_s=stats["target_time_s"],
        overhead_time_s=stats["overhead_time_s"],
    )
