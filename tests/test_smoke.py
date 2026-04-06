import torch
from src.baseline.autoregressive import load_model, run_baseline

def test_baseline():
    model, tokenizer = load_model("gpt2", device="cpu")
    result = run_baseline(model, tokenizer, "The history of AI is", max_new_tokens=20, device="cpu")
    print("Output:", result.output)
    print("Throughput:", result.throughput_tokens_per_s)
    assert result.output_tokens > 0
    assert result.throughput_tokens_per_s > 0
