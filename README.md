# Medusa PDC Optimization
### CS-3006 Parallel and Distributed Computing — Semester Project
**Spring 2026**

---

## Overview

This project is a performance optimization study of the **Medusa LLM Inference Acceleration Framework**, conducted as part of the CS-3006 Parallel and Distributed Computing course.

**Base Paper:**
> Cai et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.*
> https://arxiv.org/pdf/2401.10774v2

**Supporting Literature:**
> He et al. (2024). *FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines.*
> https://arxiv.org/pdf/2403.11421

> Artetxe et al. (2025). *NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference.*
> https://proceedings.mlsys.org/paper_files/paper/2025/file/66a026c0d17040889b50f0dfa650e5e0-Paper-Conference.pdf

---

## Identified Gap

Medusa already performs parallel candidate generation and tree-based verification. However, the candidate verification stage (`evaluate_posterior`) contains a hidden redundancy:

The original Medusa pipeline expands logits from shape `(1, T, V)` to `(C, T, V)` before scoring, then computes:

```
softmax( logits[C, T-1, V] )   →   C × (T-1) × V operations
```

Since all `C` candidates are evaluated against the **same model probability distribution**, computing the softmax `C` times is pure redundant memory work. This is a memory-bandwidth bottleneck identified by **FastDecode** as the dominant cost in LLM inference.

---

## Proposed Optimization: Shared-Softmax Vectorization

Compute softmax **once** on `(1, T-1, V)`, then broadcast to all `C` candidates via a **zero-copy memory view** (`expand()`), and apply a single batched `torch.gather`.

```
Baseline:  O(C × T × V)        — softmax reads vocab C times
Optimized: O(T × V  +  C × T) — softmax reads vocab once + C lightweight gathers
```

**Measured speedup: 3.5× to 41× depending on candidate count C.**

### PDC Concepts Demonstrated

| Concept | Where Applied |
|---|---|
| **Data Parallelism** | All `C` candidates scored in one batched tensor op — no Python loop |
| **SIMD Vectorization** | `torch.gather` over `(C, T, V)` = single vectorized lookup for all candidates |
| **Memory Optimization** | `C`-fold reduction in softmax memory reads (FastDecode principle) |
| **Zero-Copy Broadcast** | `expand()` creates a stride-based view — zero allocation, zero copy |
| **Vectorized Reduction** | `argmax` over score vector replaces sequential best-candidate search |

---

## Project Structure

```
PDC/
│
├── core/
│   ├── medusa_original/          Original Medusa code (baseline — NOT modified)
│   │   ├── utils.py              Contains evaluate_posterior() — baseline function
│   │   ├── medusa_model.py
│   │   ├── medusa_choices.py
│   │   └── kv_cache.py
│   │
│   └── medusa_optimized/         PDC-optimized versions
│       ├── pdc_posterior.py      ← CORE CONTRIBUTION: shared-softmax optimization
│       ├── utils.py
│       ├── medusa_model.py
│       ├── medusa_choices.py
│       ├── kv_cache.py
│       ├── scoring_cpu.py
│       ├── modeling_llama_kv.py
│       └── modeling_mistral_kv.py
│
├── experiments/                  All benchmark scripts (same folder, modular)
│   ├── benchmark.py              ← MASTER ENTRY POINT: runs all 4 sections
│   ├── config.py                 Central configuration (model, params, paths)
│   ├── utils_bench.py            Shared helpers: load_model, timed, build_candidates
│   ├── compare_outputs.py        Section 1 — correctness verification
│   ├── microbenchmark.py         Section 2 — isolated timing
│   ├── analysis.py               Section 3 — Amdahl's Law + complexity table
│   └── plots.py                  Section 4 — all figures saved to results/
│
├── analysis/
│   ├── kv_cache_analysis.py      KV-cache access pattern instrumentation
│   └── scheduling_analysis.py    Load-aware scheduling analysis
│
├── results/                      Auto-generated output (do not edit manually)
│   ├── benchmark_results.json    Combined results from all sections
│   ├── benchmark_results.png     4-panel benchmark figure
│   ├── amdahl_projection.png     End-to-end speedup projection
│   └── output_correctness.png    Probability delta correctness chart
│
├── tests/
│   └── sanity_check.py           Quick sanity check before running full benchmark
│
├── config.py                     Project-level configuration
├── requirements.txt              Python dependencies
└── README.md                     This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Windows / Linux / macOS
- No GPU required — all experiments run on CPU

### Step 1 — Clone or extract the project

```bash
cd path/to/your/project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
```
torch
numpy
tqdm
psutil
matplotlib
transformers
huggingface_hub
```

### Step 4 — Verify installation

```bash
python -m tests.sanity_check
```

This runs a quick check that imports work and the model can be loaded before committing to the full benchmark.

---

## Running the Benchmark

### Run everything (recommended)

```bash
python -m experiments.benchmark
```

This runs all four sections in order and saves all results and plots to `results/`.

### Run individual sections

Each section can also be run independently:

```bash
# Section 1: Correctness — proves optimization produces identical outputs
python -m experiments.compare_outputs

# Section 2: Timing — isolated latency measurement
python -m experiments.microbenchmark

# Section 3: Analysis — Amdahl projection + complexity table
python -m experiments.analysis

# Section 4: Plots — regenerate all figures from saved JSON
python -m experiments.plots
```

> **Note:** `analysis.py` and `plots.py` when run standalone will load results from `results/*.json` saved by earlier runs. You can regenerate plots without re-running model inference.

---

## Configuration

All settings are in `experiments/config.py`. Change values there and they update everywhere:

```python
MODEL_NAME      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE          = "cpu"
NUM_TRIALS      = 15          # timing runs per config; first discarded as warmup
CANDIDATE_SIZES = [4, 8, 16, 32, 64]
TEMPERATURE     = 0.7
POST_THRESHOLD  = 0.3
POST_ALPHA      = 0.09
SAMPLING        = "typical"
```

---

## Expected Output

Running `python -m experiments.benchmark` produces output in four sections:

### Section 1 — Correctness Verification
```
OUTPUT COMPARISON — Correctness Verification
P    C   BestCand_B  BestCand_O  AL_B  AL_O  Tokens  ProbDelta   Status
1    4            0           0     8     8     YES   0.00e+00   PASS ✓
1    8            0           0     8     8     YES   0.00e+00   PASS ✓
...
✓ ALL COMPARISONS PASSED
```

### Section 2 — Microbenchmark
```
C=  4 | Base:  5.410ms(±1.10) | Opt:  1.549ms(±0.57) | Speedup:  3.49x | AL: 9/9 ✓
C=  8 | Base: 10.128ms(±1.26) | Opt:  1.624ms(±0.25) | Speedup:  6.24x | AL: 9/9 ✓
C= 16 | Base: 18.565ms(±4.72) | Opt:  1.525ms(±0.34) | Speedup: 12.18x | AL: 9/9 ✓
C= 32 | Base: 36.075ms(±2.04) | Opt:  1.535ms(±0.31) | Speedup: 23.50x | AL: 9/9 ✓
C= 64 | Base: 70.981ms(±3.11) | Opt:  1.709ms(±0.45) | Speedup: 41.54x | AL: 9/9 ✓
```

### Section 3 — Amdahl's Law Projection
```
C    Kernel Speedup   10% of pipeline   20% of pipeline   35% of pipeline
4          3.49x             1.028x            1.059x            1.113x
64        41.54x             1.108x            1.242x            1.519x
```

### Section 4 — Plots saved to `results/`
- `benchmark_results.png` — 4-panel: latency bars, speedup curve, log-scale, correctness
- `amdahl_projection.png` — projected end-to-end speedup via Amdahl's Law
- `output_correctness.png` — probability delta bar chart per candidate count

---

## Key Results Summary

| Candidates (C) | Baseline (ms) | Optimized (ms) | Speedup | Output Match |
|---|---|---|---|---|
| 4  | 5.41  | 1.55 | 3.49×  | ✓ YES |
| 8  | 10.13 | 1.62 | 6.24×  | ✓ YES |
| 16 | 18.57 | 1.53 | 12.18× | ✓ YES |
| 32 | 36.08 | 1.54 | 23.50× | ✓ YES |
| 64 | 70.98 | 1.71 | 41.54× | ✓ YES |

**Important framing:** The kernel speedup (3.5×–41×) reflects the isolated scoring stage. Using Amdahl's Law with `evaluate_posterior` estimated at 10–35% of total Medusa runtime, the projected end-to-end pipeline speedup is **1.1×–1.5×**, consistent with gains reported by FastDecode (1.9×–5×) and NEO (up to 7.5×).

---

## How Baseline and Optimized Differ

| Aspect | Baseline (`core/medusa_original/utils.py`) | Optimized (`core/medusa_optimized/pdc_posterior.py`) |
|---|---|---|
| Input contract | `logits (C, T, V)` — pre-expanded | `logits (1, T, V)` — raw unexpanded |
| Softmax passes | C passes over `(T-1, V)` | 1 pass over `(T-1, V)` |
| Memory read volume | `C × T × V` floats | `T × V` floats (softmax) + `C × T` (gather) |
| Broadcast method | Pre-expanded by caller | `expand()` zero-copy view inside function |
| Gather operation | On pre-expanded `(C, T, V)` | On broadcast view — same result |
| Prefix mask | `torch.cumprod` | `torch.cumprod` — identical semantics |
| Output | `(best_candidate, accept_length)` | `(best_candidate, accept_length)` — identical |

---

## LLM Usage Disclosure

This project was developed with assistance from Claude (Anthropic). LLM assistance was used for:

- Explaining PyTorch tensor broadcasting mechanics
- Debugging shape mismatches between baseline and optimized function contracts
- Structuring the benchmark into modular files
- Reviewing complexity analysis and Amdahl's Law calculations
- Drafting report sections and identifying gaps in the framing

All code was reviewed, understood, and verified by the project authors. The core optimization idea — shared-softmax vectorization — was identified by the authors through analysis of the Medusa codebase. LLM assistance was used to implement and validate it, not to generate ideas independently.

---

## References

1. Cai et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* arXiv:2401.10774. https://arxiv.org/pdf/2401.10774v2

2. He et al. (2024). *FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines.* arXiv:2403.11421. https://arxiv.org/pdf/2403.11421

3. Artetxe et al. (2025). *NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference.* MLSys 2025. https://proceedings.mlsys.org/paper_files/paper/2025/file/66a026c0d17040889b50f0dfa650e5e0-Paper-Conference.pdf

4. Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* arXiv:1911.02150.

5. Stern et al. (2018). *Blockwise Parallel Decoding for Deep Autoregressive Models.* NeurIPS 2018.