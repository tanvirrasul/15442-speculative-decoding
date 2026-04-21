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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.speculative.spec_decode import SpecDecodingResult, SpecDecodingConfig


# @dataclass
# class SpecDecodingResult:
#     prompt: str
#     output: str
#     input_tokens: int
#     output_tokens: int
#     total_time_s: float
#     time_to_first_token_s: float
#     throughput_tokens_per_s: float
#     acceptance_rate: float
#     total_draft_tokens: int
#     total_accepted_tokens: int
#     num_target_calls: int
#     speedup_vs_baseline: float = 0.0
#     draft_time_s: float = 0.0
#     target_time_s: float = 0.0
#     overhead_time_s: float = 0.0


# @dataclass
# class SpecDecodingConfig:
#     gamma: int = 4
#     temperature: float = 1.0
#     max_new_tokens: int = 256
#     verbose_timing: bool = False
#     max_sequence_length: int = 1536  # safety cap to prevent OOM on long sequences


def load_s2s_models(
    target_model_id: str,
    draft_model_id: str,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32

    target_model = AutoModelForSeq2SeqLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    draft_model = AutoModelForSeq2SeqLM.from_pretrained(
        draft_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    target_model.eval()
    draft_model.eval()
    return target_model, draft_model, tokenizer


def _safe_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with NaN/inf guards for numerical stability."""
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    return F.softmax(logits / max(temperature, 1e-8), dim=-1)


def _sample(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample a token. Input: [1, vocab_size]. Output: [1, 1].
    If temperature == 0.0, use greedy (argmax).
    """
    if temperature == 0.0:
        return probs.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def s2s_speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: SpecDecodingConfig,
    decoder_start_token_id: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    generated = torch.full(
        (input_ids.shape[0], 1), decoder_start_token_id, device=device, dtype=torch.long
    )
    total_draft = 0
    total_accepted = 0
    num_target_calls = 0
    total_draft_time = 0.0
    total_target_time = 0.0
    total_overhead_time = 0.0

    with torch.no_grad():
        # Encode source prompt for each model separately
        t0 = time.perf_counter()
        target_encoder_outputs = target_model.get_encoder()(input_ids, attention_mask=attention_mask, return_dict=True)
        draft_encoder_outputs = draft_model.get_encoder()(input_ids, attention_mask=attention_mask, return_dict=True)
        _sync(device)
        total_draft_time += time.perf_counter() - t0

        # Pre-fill draft KV cache with the decoder start token
        t0 = time.perf_counter()
        draft_out = draft_model(
            encoder_outputs=draft_encoder_outputs,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=generated,
            use_cache=True,
        )
        draft_past_kv = draft_out.past_key_values
        _sync(device)
        total_draft_time += time.perf_counter() - t0

        while generated.shape[1] - 1 < config.max_new_tokens:

            # Safety cap: prevent OOM from excessively long sequences
            if generated.shape[1] >= config.max_sequence_length:
                break

            # --- Draft phase: propose gamma tokens using KV cache ---
            t_draft_start = time.perf_counter()
            draft_probs = []
            draft_tokens = []  # each [1, 1]
            current_past_kv = draft_past_kv
            last_token = generated[:, -1:]

            for _ in range(config.gamma):
                out = draft_model(
                    encoder_outputs=draft_encoder_outputs,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=last_token,
                    past_key_values=current_past_kv,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]

                if config.temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    probs = torch.zeros_like(logits)
                    probs.scatter_(1, next_token, 1.0)
                else:
                    probs = _safe_softmax(logits, config.temperature)
                    next_token = _sample(probs, config.temperature)

                draft_probs.append(probs)
                draft_tokens.append(next_token)
                last_token = next_token
                current_past_kv = out.past_key_values

            _sync(device)
            total_draft_time += time.perf_counter() - t_draft_start
            total_draft += config.gamma

            # Assemble draft sequence
            t_overhead_start = time.perf_counter()
            draft_token_tensor = torch.cat(draft_tokens, dim=1)
            draft_ids = torch.cat([generated, draft_token_tensor], dim=1)
            total_overhead_time += time.perf_counter() - t_overhead_start

            # --- Verify phase: single target forward pass ---
            t_target_start = time.perf_counter()
            target_out = target_model(
                encoder_outputs=target_encoder_outputs,
                encoder_attention_mask=attention_mask,
                decoder_input_ids=draft_ids,
            )
            _sync(device)
            total_target_time += time.perf_counter() - t_target_start
            num_target_calls += 1

            # Extract target probs at each draft position
            t_overhead_start = time.perf_counter()
            target_logits = target_out.logits[
                :,
                generated.shape[1] - 1: generated.shape[1] + config.gamma - 1,
                :
            ]

            if config.temperature == 0.0:
                target_probs_all = torch.zeros_like(target_logits)
                argmax_ids = target_logits.argmax(dim=-1, keepdim=True)
                target_probs_all.scatter_(2, argmax_ids, 1.0)
            else:
                target_probs_all = torch.stack([
                    _safe_softmax(target_logits[:, i, :], config.temperature)
                    for i in range(config.gamma)
                ], dim=1)

            # --- Acceptance / rejection ---
            accepted_count = 0
            rejected_at = None

            for i in range(config.gamma):
                token_idx = draft_tokens[i].squeeze().item()
                p = target_probs_all[0, i, token_idx]
                q = draft_probs[i][0, token_idx]

                if config.temperature == 0.0:
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
                        resampled = target_logits[0, i].argmax(dim=-1).reshape(1, 1)
                    else:
                        corrected = torch.clamp(
                            target_probs_all[0, i, :] - draft_probs[i][0, :], min=0
                        )
                        corrected = corrected / (corrected.sum() + 1e-8)
                        corrected = torch.nan_to_num(corrected, nan=0.0)
                        if corrected.sum() < 1e-8:
                            corrected = target_probs_all[0, i, :]
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
                    bonus_probs = _safe_softmax(bonus_logits, config.temperature)
                    bonus_token = _sample(bonus_probs, config.temperature)
                generated = torch.cat(
                    [generated, draft_token_tensor, bonus_token], dim=1
                )
                total_accepted += config.gamma

            total_overhead_time += time.perf_counter() - t_overhead_start

            # --- KV cache update ---
            t_draft_start = time.perf_counter()
            if rejected_at is None:
                # All accepted: just feed bonus token to update cache
                draft_out = draft_model(
                    encoder_outputs=draft_encoder_outputs,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=bonus_token,
                    past_key_values=current_past_kv,
                    use_cache=True,
                )
                draft_past_kv = draft_out.past_key_values
            else:
                # Rejection: full reprocess of current sequence
                draft_out = draft_model(
                    encoder_outputs=draft_encoder_outputs,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=generated,
                    use_cache=True,
                )
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


def run_s2s_speculative(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    config: SpecDecodingConfig,
    device: str = "cuda",
) -> SpecDecodingResult:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    decoder_start_token_id = getattr(target_model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        _ = target_model.get_encoder()(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        _ = draft_model.get_encoder()(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    output_ids, stats = s2s_speculative_decode(
        target_model,
        draft_model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
        config,
        decoder_start_token_id,
        device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    with torch.no_grad():
        t0 = time.perf_counter()
        _ = target_model(
            encoder_outputs=target_model.get_encoder()(inputs["input_ids"], attention_mask=inputs.get("attention_mask"), return_dict=True),
            decoder_input_ids=torch.tensor([[decoder_start_token_id]], device=device),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

    total_time = end - start
    output_tokens = output_ids.shape[1] - 1
    output_text = tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)

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
