import torch
from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig

target, draft, tokenizer = load_models("gpt2", "gpt2", device="cpu")
config = SpecDecodingConfig(gamma=4, max_new_tokens=20)
result = run_speculative(target, draft, tokenizer, "The history of AI is", config, device="cpu")
print("Output:", result.output)
print("Acceptance rate:", result.acceptance_rate)
print("Target model calls:", result.num_target_calls)
print("Spec decoding works!")
