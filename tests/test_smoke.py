import torch
from src.baseline.autoregressive import load_model, run_baseline

model, tokenizer = load_model("gpt2", device="cpu")  # tiny, no GPU needed
result = run_baseline(model, tokenizer, "The history of AI is", max_new_tokens=20, device="cpu")
print("Output:", result.output)
print("Throughput:", result.throughput_tokens_per_s)
print("Baseline works!")
