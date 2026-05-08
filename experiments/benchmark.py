"""
benchmark.py
============
Master entry point for the CS-3006 PDC Medusa Optimization benchmark suite.

Runs all four sections in order and saves a combined JSON result:
  Section 1 — compare_outputs   : correctness verification
  Section 2 — microbenchmark    : isolated timing
  Section 3 — analysis          : Amdahl projection + complexity
  Section 4 — plots             : all figures saved to results/

Run:
    python -m experiments.benchmark

Individual sections can also be run standalone:
    python -m experiments.compare_outputs
    python -m experiments.microbenchmark
    python -m experiments.analysis
    python -m experiments.plots
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config        import MODEL_NAME, DEVICE, RESULTS_DIR
from experiments.utils_bench   import load_model

from experiments.compare_outputs import run_comparison
from experiments.microbenchmark  import run_microbenchmark
from experiments.analysis        import run_amdahl, run_complexity
from experiments.plots           import plot_benchmark, plot_amdahl, plot_correctness


# ─────────────────────────────────────────────────────────────────────────────
def run_all():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print()
    print("╔" + "═" * 70 + "╗")
    print("║   Medusa PDC Optimization — Complete Benchmark Suite" + " " * 18 + "║")
    print("║   CS-3006 Parallel and Distributed Computing" + " " * 25 + "║")
    print("╠" + "═" * 70 + "╣")
    print(f"║   Model  : {MODEL_NAME:<58}║")
    print(f"║   Device : {DEVICE:<58}║")
    print("╠" + "═" * 70 + "╣")
    print("║   Files in experiments/:" + " " * 45 + "║")
    print("║     config.py          — all shared settings" + " " * 25 + "║")
    print("║     utils_bench.py     — shared helpers (loader, timer, builder)" + " " * 3 + "║")
    print("║     compare_outputs.py — Section 1: correctness check" + " " * 17 + "║")
    print("║     microbenchmark.py  — Section 2: isolated timing" + " " * 19 + "║")
    print("║     analysis.py        — Section 3: Amdahl + complexity" + " " * 14 + "║")
    print("║     plots.py           — Section 4: all figures" + " " * 22 + "║")
    print("║     benchmark.py       — this file: runs all sections" + " " * 17 + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    # ── Load model once — shared across all sections ──────────────────────────
    tokenizer, model = load_model()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Output Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("[ 1 / 4 ]  Output Comparison\n")
    comparisons, all_correct = run_comparison(model, tokenizer)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Microbenchmark
    # ══════════════════════════════════════════════════════════════════════════
    print("[ 2 / 4 ]  Microbenchmark\n")
    summary = run_microbenchmark(model, tokenizer)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Analysis
    # ══════════════════════════════════════════════════════════════════════════
    print("[ 3 / 4 ]  Analysis\n")
    theoretical = run_amdahl(summary)
    complexity  = run_complexity()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Plots
    # ══════════════════════════════════════════════════════════════════════════
    print("[ 4 / 4 ]  Generating plots\n")
    plot_benchmark(summary)
    plot_amdahl(theoretical)
    plot_correctness(comparisons)

    # ── Save combined JSON ────────────────────────────────────────────────────
    combined = {
        "model" : MODEL_NAME,
        "device": DEVICE,
        "optimization": {
            "name"     : "Shared-Softmax Vectorization",
            "baseline" : "logits (C, T, V) — softmax O(C·T·V)",
            "optimized": "logits (1, T, V) — softmax O(T·V), zero-copy broadcast",
            "pdc_concepts": [
                "Data Parallelism: all C candidates in one batched tensor op",
                "SIMD Vectorization: torch.gather over (C,T,V) in single call",
                "Memory Optimization: C-fold reduction in softmax memory reads",
                "Zero-Copy Broadcast: expand() creates stride view, no allocation",
                "Vectorized Reduction: argmax replaces sequential candidate search",
            ],
            "literature": {
                "FastDecode": "Reducing memory bandwidth is the primary lever for LLM throughput",
                "NEO"       : "Decoupled scoring stage operating on pre-computed logits",
            },
        },
        "section_1_correctness"  : {"all_passed": all_correct, "comparisons": comparisons},
        "section_2_microbenchmark": summary,
        "section_3_amdahl"       : theoretical,
        "section_3_complexity"   : complexity,
    }

    out = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(combined, f, indent=4)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 70 + "╗")
    print("║   BENCHMARK COMPLETE" + " " * 49 + "║")
    print("╠" + "═" * 70 + "╣")

    correct_str = "ALL PASSED ✓" if all_correct else "SOME FAILED ✗"
    print(f"║   Correctness  : {correct_str:<52}║")

    best = max(summary, key=lambda r: r["speedup"])
    print(f"║   Peak speedup : {best['speedup']:.2f}x  at C={best['candidate_count']:<47}║")

    worst = min(summary, key=lambda r: r["speedup"])
    print(f"║   Min speedup  : {worst['speedup']:.2f}x  at C={worst['candidate_count']:<47}║")

    print("╠" + "═" * 70 + "╣")
    print(f"║   results/benchmark_results.json" + " " * 37 + "║")
    print(f"║   results/benchmark_results.png" + " " * 38 + "║")
    print(f"║   results/amdahl_projection.png" + " " * 38 + "║")
    print(f"║   results/output_correctness.png" + " " * 37 + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    return combined


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()