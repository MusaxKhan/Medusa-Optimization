"""
microbenchmark.py
=================
Section 2 of the benchmark suite.

Measures isolated timing of baseline vs optimized evaluate_posterior()
across all candidate sizes and all prompts. Reports:
  - Mean latency ± std per configuration
  - Speedup ratio
  - Acceptance length (correctness check)
  - CPU utilization (if psutil installed)

Run standalone:
    python -m experiments.microbenchmark
"""

import sys
import os
import json

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config       import (
    PROMPTS, CANDIDATE_SIZES, NUM_TRIALS,
    TEMPERATURE, POST_THRESHOLD, POST_ALPHA, TOP_P, SAMPLING,
    RESULTS_DIR,
)
from experiments.utils_bench  import (
    load_model, build_realistic_candidates, get_logits, timed,
)
from core.medusa_original.utils          import evaluate_posterior          as baseline_fn
from core.medusa_optimized.pdc_posterior import optimized_evaluate_posterior as optimized_fn

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
def run_microbenchmark(model, tokenizer):
    """
    Run timing comparison across all (prompt, C) configurations.

    Returns:
        summary : list of aggregated result dicts (one per C value)
    """
    print("═" * 72)
    print("  MICROBENCHMARK — evaluate_posterior() Isolated Timing")
    print(f"  {NUM_TRIALS} trials per config, first discarded as warmup")
    print("═" * 72 + "\n")

    # Per-prompt raw results
    all_base = {c: [] for c in CANDIDATE_SIZES}
    all_opt  = {c: [] for c in CANDIDATE_SIZES}
    all_al_b = {c: [] for c in CANDIDATE_SIZES}
    all_al_o = {c: [] for c in CANDIDATE_SIZES}

    if HAS_PSUTIL:
        cpu_base = {c: [] for c in CANDIDATE_SIZES}
        cpu_opt  = {c: [] for c in CANDIDATE_SIZES}

    for p_idx, prompt in enumerate(PROMPTS):

        print(f"  Prompt {p_idx+1}/{len(PROMPTS)}: \"{prompt[:58]}\"")

        raw_logits, input_ids = get_logits(model, tokenizer, prompt)
        T, V = raw_logits.shape[1], raw_logits.shape[2]
        print(f"    seq_len={T}  vocab_size={V}")

        for C in CANDIDATE_SIZES:

            candidates      = build_realistic_candidates(raw_logits, input_ids, C)
            expanded_logits = raw_logits.expand(C, -1, -1).contiguous()

            # ── Baseline ──────────────────────────────────────────────────────
            if HAS_PSUTIL:
                psutil.cpu_percent(interval=None)   # reset counter

            b_mean, b_std, b_res = timed(
                baseline_fn,
                logits=expanded_logits, candidates=candidates,
                temperature=TEMPERATURE, posterior_threshold=POST_THRESHOLD,
                posterior_alpha=POST_ALPHA, top_p=TOP_P,
                sampling=SAMPLING, fast=True,
            )

            if HAS_PSUTIL:
                cpu_base[C].append(psutil.cpu_percent(interval=None))

            # ── Optimized ─────────────────────────────────────────────────────
            if HAS_PSUTIL:
                psutil.cpu_percent(interval=None)

            o_mean, o_std, o_res = timed(
                optimized_fn,
                logits=raw_logits, candidates=candidates,
                temperature=TEMPERATURE, posterior_threshold=POST_THRESHOLD,
                posterior_alpha=POST_ALPHA, top_p=TOP_P,
                sampling=SAMPLING, fast=True,
            )

            if HAS_PSUTIL:
                cpu_opt[C].append(psutil.cpu_percent(interval=None))

            _, al_b = b_res
            _, al_o = o_res
            speedup = b_mean / o_mean if o_mean > 0 else float("inf")
            mark    = "✓" if int(al_b) == int(al_o) else "✗ MISMATCH"

            all_base[C].append(b_mean)
            all_opt[C].append(o_mean)
            all_al_b[C].append(int(al_b))
            all_al_o[C].append(int(al_o))

            print(
                f"    C={C:3d} | "
                f"Base: {b_mean*1000:7.3f}ms(±{b_std*1000:.2f}) | "
                f"Opt:  {o_mean*1000:7.3f}ms(±{o_std*1000:.2f}) | "
                f"Speedup: {speedup:6.2f}x | "
                f"AL: {int(al_b)}/{int(al_o)} {mark}"
            )
        print()

    # ── Aggregate across prompts ───────────────────────────────────────────────
    print("═" * 72)
    print("  AGGREGATED RESULTS  (mean ± std across all prompts)")
    print("═" * 72)
    print(
        f"\n  {'C':>4}  {'Baseline(ms)':>16}  {'Optimized(ms)':>16}  "
        f"{'Speedup':>10}  {'AL_B':>6}  {'AL_O':>6}  {'Match':>6}"
    )
    print("  " + "-" * 72)

    summary = []

    for C in CANDIDATE_SIZES:

        b_avg  = float(np.mean(all_base[C]))
        b_std_ = float(np.std(all_base[C]))
        o_avg  = float(np.mean(all_opt[C]))
        o_std_ = float(np.std(all_opt[C]))
        sp     = b_avg / o_avg if o_avg > 0 else float("inf")
        al_b   = float(np.mean(all_al_b[C]))
        al_o   = float(np.mean(all_al_o[C]))
        match  = "YES" if abs(al_b - al_o) < 0.01 else "NO"

        print(
            f"  {C:>4}  "
            f"{b_avg*1000:>9.3f}±{b_std_*1000:>5.2f}  "
            f"{o_avg*1000:>9.3f}±{o_std_*1000:>5.2f}  "
            f"{sp:>10.2f}x  {al_b:>6.2f}  {al_o:>6.2f}  {match:>6}"
        )

        row = {
            "candidate_count"     : C,
            "baseline_ms"         : round(b_avg * 1000, 4),
            "baseline_std_ms"     : round(b_std_ * 1000, 4),
            "optimized_ms"        : round(o_avg * 1000, 4),
            "optimized_std_ms"    : round(o_std_ * 1000, 4),
            "speedup"             : round(sp, 4),
            "accept_len_baseline" : round(al_b, 2),
            "accept_len_optimized": round(al_o, 2),
            "outputs_match"       : match,
        }

        if HAS_PSUTIL:
            row["cpu_util_baseline_pct"]  = round(float(np.mean(cpu_base[C])), 2)
            row["cpu_util_optimized_pct"] = round(float(np.mean(cpu_opt[C])),  2)

        summary.append(row)

    print()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tokenizer, model = load_model()
    summary = run_microbenchmark(model, tokenizer)
    out = os.path.join(RESULTS_DIR, "microbenchmark_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Saved → {out}")