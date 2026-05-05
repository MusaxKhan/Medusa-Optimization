import time
import json
import numpy as np
import matplotlib.pyplot as plt

from core.medusa_optimized.scoring_cpu import optimized_scoring


# -----------------------------
# BASELINE SCORING (SIMPLIFIED MEDUSA)
# -----------------------------
def baseline_scoring(candidates):
    """
    Simulates original Medusa-style sequential scoring.
    """
    start = time.time()

    scores = []
    for c in candidates:
        time.sleep(0.002)  # simulate GPU-coupled verification delay
        score = (len(c) ** 0.5) * 0.5
        scores.append(score)

    return scores, time.time() - start


# -----------------------------
# EXPERIMENT RUNNER
# -----------------------------
def run_experiment(candidate_sizes):
    baseline_times = []
    optimized_times = []
    speedups = []

    for n in candidate_sizes:

        # synthetic candidates
        candidates = ["token_" * (i % 5 + 1) for i in range(n)]

        # ---- BASELINE ----
        _, base_time = baseline_scoring(candidates)

        # ---- OPTIMIZED (SIMD/OpenMP/NEO model) ----
        _, opt_time = optimized_scoring(candidates, use_simd=False)

        baseline_times.append(base_time)
        optimized_times.append(opt_time)

        speedups.append(base_time / opt_time)

        print(f"Candidates: {n} | Baseline: {base_time:.4f}s | Optimized: {opt_time:.4f}s | Speedup: {speedups[-1]:.2f}x")

    return baseline_times, optimized_times, speedups


# -----------------------------
# PLOTTING (FOR REPORT)
# -----------------------------
def plot_results(candidate_sizes, baseline, optimized, speedups):

    plt.figure()
    plt.plot(candidate_sizes, baseline, label="Baseline (Medusa-like)")
    plt.plot(candidate_sizes, optimized, label="Optimized (PDC)")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Comparison")
    plt.legend()
    plt.savefig("results/latency_comparison.png")

    plt.figure()
    plt.plot(candidate_sizes, speedups)
    plt.xlabel("Number of Candidates")
    plt.ylabel("Speedup (x)")
    plt.title("Speedup from PDC Optimization")
    plt.savefig("results/speedup_curve.png")

    print("Plots saved in results/")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    candidate_sizes = [10, 20, 40, 80, 160]

    baseline, optimized, speedups = run_experiment(candidate_sizes)

    # save results
    with open("results/timings.json", "w") as f:
        json.dump({
            "candidate_sizes": candidate_sizes,
            "baseline": baseline,
            "optimized": optimized,
            "speedups": speedups
        }, f, indent=4)

    plot_results(candidate_sizes, baseline, optimized, speedups)
