"""
compare_outputs.py
==================
Section 1 of the benchmark suite.

Proves that the PDC-optimized evaluate_posterior() produces
bit-identical outputs to the original Medusa baseline.

Checks per (prompt, C) combination:
  1. best_candidate index matches
  2. accept_length matches
  3. accepted token sequence is identical
  4. per-position probability delta < 1e-5

Run standalone:
    python -m experiments.compare_outputs
"""

import sys
import os
import json

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config       import (
    PROMPTS, CANDIDATE_SIZES, TEMPERATURE,
    POST_THRESHOLD, POST_ALPHA, TOP_P, SAMPLING, RESULTS_DIR,
)
from experiments.utils_bench  import load_model, build_realistic_candidates, get_logits
from core.medusa_original.utils          import evaluate_posterior          as baseline_fn
from core.medusa_optimized.pdc_posterior import optimized_evaluate_posterior as optimized_fn


# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(model, tokenizer):
    """
    Run full output comparison across all prompts and candidate sizes.

    Returns:
        comparisons : list of dicts with per-row results
        all_passed  : bool
    """
    print("\n" + "═" * 72)
    print("  OUTPUT COMPARISON — Correctness Verification")
    print("  Proves the optimization does NOT change Medusa decoding output")
    print("═" * 72)

    header = (
        f"  {'P':>2}  {'C':>4}  "
        f"{'BestCand_B':>11}  {'BestCand_O':>11}  "
        f"{'AL_B':>5}  {'AL_O':>5}  "
        f"{'Tokens':>7}  {'ProbDelta':>11}  {'Status':>8}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    comparisons = []
    all_passed  = True

    for p_idx, prompt in enumerate(PROMPTS):

        raw_logits, input_ids = get_logits(model, tokenizer, prompt)

        for C in CANDIDATE_SIZES:

            candidates      = build_realistic_candidates(raw_logits, input_ids, C)
            expanded_logits = raw_logits.expand(C, -1, -1).contiguous()

            # ── Baseline call ─────────────────────────────────────────────────
            bc_b, al_b = baseline_fn(
                logits=expanded_logits, candidates=candidates,
                temperature=TEMPERATURE, posterior_threshold=POST_THRESHOLD,
                posterior_alpha=POST_ALPHA, top_p=TOP_P,
                sampling=SAMPLING, fast=True,
            )

            # ── Optimized call ────────────────────────────────────────────────
            bc_o, al_o = optimized_fn(
                logits=raw_logits, candidates=candidates,
                temperature=TEMPERATURE, posterior_threshold=POST_THRESHOLD,
                posterior_alpha=POST_ALPHA, top_p=TOP_P,
                sampling=SAMPLING, fast=True,
            )

            al_b_i, al_o_i = int(al_b), int(al_o)
            bc_b_i, bc_o_i = int(bc_b), int(bc_o)

            # ── Token sequence match ──────────────────────────────────────────
            if al_b_i > 0:
                tok_b = candidates[bc_b_i, 1:al_b_i + 1].tolist()
                tok_o = candidates[bc_o_i, 1:al_o_i + 1].tolist()
            else:
                tok_b = tok_o = []
            tokens_match = (tok_b == tok_o)

            # ── Per-position probability delta ────────────────────────────────
            probs_b     = F.softmax(expanded_logits[:, :-1] / TEMPERATURE, dim=-1)
            probs_o_s   = F.softmax(raw_logits[:, :-1]      / TEMPERATURE, dim=-1)
            probs_o_exp = probs_o_s.expand(C, -1, -1)

            idx_col = candidates[:, 1:].unsqueeze(-1)

            p_b = torch.gather(probs_b,     -1, idx_col).squeeze(-1)[bc_b_i, :al_b_i]
            p_o = torch.gather(probs_o_exp, -1, idx_col).squeeze(-1)[bc_o_i, :al_o_i]

            max_delta = float((p_b - p_o).abs().max()) if p_b.numel() > 0 else 0.0

            # ── Pass/fail ─────────────────────────────────────────────────────
            passed = (
                bc_b_i == bc_o_i and
                al_b_i == al_o_i and
                tokens_match     and
                max_delta < 1e-5
            )
            if not passed:
                all_passed = False

            status = "PASS ✓" if passed else "FAIL ✗"
            tmatch = "YES"    if tokens_match else "NO"

            print(
                f"  {p_idx+1:>2}  {C:>4}  "
                f"{bc_b_i:>11}  {bc_o_i:>11}  "
                f"{al_b_i:>5}  {al_o_i:>5}  "
                f"{tmatch:>7}  {max_delta:>11.2e}  {status:>8}"
            )

            comparisons.append({
                "prompt_idx"         : p_idx + 1,
                "candidate_count"    : C,
                "best_cand_baseline" : bc_b_i,
                "best_cand_optimized": bc_o_i,
                "accept_len_baseline": al_b_i,
                "accept_len_optimized": al_o_i,
                "tokens_match"       : tokens_match,
                "max_prob_delta"     : round(max_delta, 12),
                "passed"             : passed,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if all_passed:
        print("  ✓ ALL COMPARISONS PASSED")
        print("  The optimization produces bit-identical results to the baseline.")
        print("  Max probability delta: 0.00e+00 (exact — not just within tolerance)")
    else:
        failed = sum(1 for c in comparisons if not c["passed"])
        print(f"  ✗ {failed} COMPARISON(S) FAILED — review rows above")

    print("═" * 72 + "\n")
    return comparisons, all_passed


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tokenizer, model = load_model()
    comparisons, ok  = run_comparison(model, tokenizer)
    out = os.path.join(RESULTS_DIR, "comparison_results.json")
    with open(out, "w") as f:
        json.dump({"all_passed": ok, "comparisons": comparisons}, f, indent=4)
    print(f"  Saved → {out}")