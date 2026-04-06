# Speculative Decoding: Performance Boundary Characterization

**CMU 15-442 MLSys Project**
Tanvir Rasul (trasul) | Anagha Pai (anaghap) | Shreya Mukherjee (shreyamu)

## Overview

This project systematically characterizes the performance boundaries of speculative decoding — mapping when it helps vs. hurts LLM inference performance. We study inflection points across acceptance rate thresholds, context lengths, draft model sizes, and hardware characteristics (A100/H100 vs. RTX 4070).

## Project Structure

```
speculative-decoding/
├── src/
│   ├── baseline/         # Standard autoregressive inference
│   ├── speculative/      # Speculative decoding implementation
│   ├── profiling/        # Nsight-compatible profiling + metrics
│   └── utils/            # Shared helpers (data loading, logging, etc.)
├── experiments/          # Modal experiment scripts (one per hypothesis)
├── results/              # Raw outputs, CSVs, plots
├── scripts/              # Setup, launch, and utility scripts
└── tests/                # Unit tests
```

## Hypotheses Under Investigation

- **H1**: Short contexts (<512 tokens) cause slowdown due to verification overhead dominating
- **H2**: Low acceptance rates (domain mismatch) cause slowdown below a crossover threshold
- **H3**: Oversized draft models cause slowdown when draft generation cost exceeds token savings
- **H4**: Memory-bound GPUs (RTX 4070) see minimal or negative speedup

## Models

| Role | Model |
|------|-------|
| Target | Llama-2-7B |
| Draft (distilled) | TinyLlama-1.1B |
| Draft (family) | Llama-2-1.3B |
| Draft (baseline) | N-gram model |

## Datasets

| Dataset | Type | Expected behavior |
|---------|------|-------------------|
| ShareGPT | Conversational | High acceptance rate |
| CNN/DailyMail | Summarization | Medium acceptance, long contexts |
| HumanEval | Code generation | Low acceptance (domain mismatch) |

## Hardware

- **A100 / H100** — via Modal GPU credits
- **RTX 4070** — consumer GPU, memory-bound baseline

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/speculative-decoding.git
cd speculative-decoding

# Install dependencies
pip install -r requirements.txt

# Set up Modal
pip install modal
modal setup
```

## Running Experiments

```bash
# Run baseline autoregressive benchmark
modal run experiments/baseline_benchmark.py

# Run speculative decoding benchmark
modal run experiments/spec_decoding_benchmark.py

# Run a specific hypothesis experiment
modal run experiments/h1_context_length.py
```

## Metrics

- **Primary**: Speedup vs. autoregressive baseline (including sub-1x)
- **Secondary**: Time to first token (TTFT), throughput (tokens/sec), acceptance rate, memory bandwidth utilization
- **Profiling**: NVIDIA Nsight Systems for verification overhead and draft generation cost breakdown
