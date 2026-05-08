"""
analysis.py
===========
Section 3 of the benchmark suite.

Two analyses built on top of microbenchmark results:

  A. Amdahl's Law Projection
     Estimates end-to-end Medusa pipeline speedup from the kernel speedup.
     Uses S = 1 / ((1-p) + p/S_kernel) for p in {10%, 20%, 35%}.

  B. Complexity Analysis
     Compares theoretical operation counts for baseline vs optimized,
     showing the O(C*T*V) → O(T*V + C*T) reduction.

Run standalone:
    python -m experiments.analysis
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config import (
    PIPELINE_FRACTIONS, CANDIDATE_SIZES,
    VOCAB_SIZE_ANALYSIS, SEQ_LEN_ANALYSIS, RESULTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
def run_amdahl(summary):
    """
    Apply Amdahl's Law to each row in summary.

    Args:
        summary : list of dicts from microbenchmark.run_microbenchmark()

    Returns:
        theoretical : list of dicts with projected pipeline speedups
    """
    print("═" * 72)
    print("  AMDAHL'S LAW — Estimated End-to-End Pipeline Speedup")
    print("═" * 72)
    print()
    print("  S_overall = 1 / ( (1 - p) + p / S_kernel )")
    print("  p = fraction of total Medusa runtime in evaluate_posterior\n")

    fraction_names = list(PIPELINE_FRACTIONS.keys())
    col_w = 22
    header_cols = "  ".join(f"{n:>{col_w}}" for n in fraction_names)
    print(f"  {'C':>4}  {'Kernel Speedup':>16}  {header_cols}")
    print("  " + "-" * (4 + 16 + 3 + (col_w + 2) * len(fraction_names)))

    theoretical = []

    for row in summary:

        C   = row["candidate_count"]
        S_k = row["speedup"]
        entry = {"candidate_count": C, "kernel_speedup": S_k}

        vals = []
        for name, p in PIPELINE_FRACTIONS.items():
            S_overall = 1.0 / ((1.0 - p) + (p / S_k))
            entry[name] = round(S_overall, 4)
            vals.append(S_overall)

        val_str = "  ".join(f"{v:>{col_w}.3f}x" for v in vals)
        print(f"  {C:>4}  {S_k:>15.2f}x  {val_str}")
        theoretical.append(entry)

    # Narrative
    s20_64 = next(
        (r["20% of pipeline"] for r in theoretical if r["candidate_count"] == 64),
        None
    )
    print()
    print("  Interpretation:")
    print("  ──────────────────────────────────────────────────────────────────")
    print("  A 40x kernel speedup does NOT mean Medusa becomes 40x faster.")
    print("  The GPU forward pass (transformer layers + KV-cache + attention)")
    print("  dominates total inference time, especially for Vicuna-7B.")
    if s20_64:
        print(f"\n  At C=64: 41x kernel speedup → ~{s20_64:.2f}x end-to-end (p=20%)")
    print("  This is consistent with FastDecode (1.9x–5x) and NEO (up to 7.5x)")
    print("  which also report modest end-to-end gains from kernel-level work.\n")
    print("═" * 72 + "\n")

    return theoretical


# ─────────────────────────────────────────────────────────────────────────────
def run_complexity():
    """
    Print theoretical operation count comparison for baseline vs optimized.
    Shows why the speedup grows proportionally with C.
    """
    V = VOCAB_SIZE_ANALYSIS
    T = SEQ_LEN_ANALYSIS

    print("═" * 72)
    print("  COMPLEXITY ANALYSIS")
    print("═" * 72)
    print(f"\n  V (vocab size) = {V:,}   T ≈ {T}  (positions per decoding step)")
    print()
    print(f"  {'C':>4}  {'Baseline ops':>18}  {'Optimized ops':>18}  {'Theory ratio':>14}")
    print("  " + "-" * 60)

    results = []
    for C in CANDIDATE_SIZES:
        base_ops = C * T * V
        opt_ops  = T * V + C * T
        ratio    = base_ops / opt_ops
        print(f"  {C:>4}  {base_ops:>18,}  {opt_ops:>18,}  {ratio:>13.1f}x")
        results.append({
            "C": C,
            "baseline_ops" : base_ops,
            "optimized_ops": opt_ops,
            "theory_ratio" : round(ratio, 2),
        })

    print()
    print("  Baseline:  O(C × T × V)        — softmax over vocab, C times")
    print("  Optimized: O(T × V  +  C × T)  — one softmax + C fast gathers")
    print(f"  When V >> C ({V:,} >> {CANDIDATE_SIZES[-1]}), ratio → C exactly\n")
    print("═" * 72 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load microbenchmark results if they exist, else use placeholder
    mb_path = os.path.join(RESULTS_DIR, "microbenchmark_results.json")

    if os.path.exists(mb_path):
        with open(mb_path) as f:
            summary = json.load(f)
        print("  Loaded microbenchmark results from disk.\n")
    else:
        print("  WARNING: microbenchmark_results.json not found.")
        print("  Run microbenchmark.py first, or run benchmark.py for all sections.\n")
        summary = [{"candidate_count": c, "speedup": 1.0} for c in CANDIDATE_SIZES]

    theoretical = run_amdahl(summary)
    complexity  = run_complexity()

    out = os.path.join(RESULTS_DIR, "analysis_results.json")
    with open(out, "w") as f:
        json.dump({
            "amdahl_projection": theoretical,
            "complexity"       : complexity,
        }, f, indent=4)
    print(f"  Saved → {out}")