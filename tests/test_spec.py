import torch
from src.speculative.spec_decode import load_models, run_speculative, SpecDecodingConfig

def test_speculative():
    target, draft, tokenizer = load_models("gpt2", "gpt2", device="cpu")
    config = SpecDecodingConfig(gamma=4, max_new_tokens=20)
    result = run_speculative(target, draft, tokenizer, "The history of AI is", config, device="cpu")
    print("Output:", result.output)
    print("Acceptance rate:", result.acceptance_rate)
    print("Target model calls:", result.num_target_calls)
    assert result.output_tokens > 0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.num_target_calls > 0
