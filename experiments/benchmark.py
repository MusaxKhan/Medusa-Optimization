import time
import json
import numpy as np
import matplotlib.pyplot as plt

import config
from core.medusa_optimized.scoring_cpu import optimized_scoring


# -----------------------------
# BASELINE (Medusa-like sequential scoring)
# -----------------------------
def baseline_scoring(candidates):
    start = time.time()

    scores = []
    for c in candidates:
        time.sleep(0.002)  # simulate GPU-coupled verification delay
        scores.append(len(c) ** 0.5)

    return time.time() - start


# -----------------------------
# OPTIMIZED SCORING WRAPPER
# -----------------------------
def optimized_scoring_run(candidates):
    _, exec_time = optimized_scoring(
        candidates,
        use_simd=config.USE_SIMD_VECTORIZATION
    )
    return exec_time


# -----------------------------
# STABLE EXPERIMENT RUNNER
# -----------------------------
def run_stable_experiment(candidate_sizes):

    baseline_avg = []
    optimized_avg = []
    speedups = []
    std_dev_opt = []

    for n in candidate_sizes:

        # synthetic Medusa-like candidates
        candidates = ["token_" * (i % 5 + 1) for i in range(n)]

        baseline_times = []
        optimized_times = []

        for _ in range(config.NUM_RUNS):

            # baseline run
            baseline_times.append(baseline_scoring(candidates))

            # optimized run
            optimized_times.append(optimized_scoring_run(candidates))

        # statistics
        b_mean = np.mean(baseline_times)
        o_mean = np.mean(optimized_times)

        baseline_avg.append(b_mean)
        optimized_avg.append(o_mean)
        speedups.append(b_mean / o_mean)
        std_dev_opt.append(np.std(optimized_times))

        print("\n==============================")
        print(f"Candidates: {n}")
        print(f"Baseline Time   : {b_mean:.4f} s")
        print(f"Optimized Time  : {o_mean:.4f} s")
        print(f"Speedup         : {b_mean / o_mean:.2f}x")
        print(f"Std Dev (Opt)   : {np.std(optimized_times):.5f}")
        print("==============================")

    return baseline_avg, optimized_avg, speedups, std_dev_opt


# -----------------------------
# PLOTTING (REPORT READY)
# -----------------------------
def plot_results(n_values, baseline, optimized, speedups):

    # Latency comparison
    plt.figure()
    plt.plot(n_values, baseline, marker='o', label="Baseline (Medusa-like)")
    plt.plot(n_values, optimized, marker='o', label="Optimized (PDC)")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Latency (seconds)")
    plt.title("Medusa Inference Latency Comparison")
    plt.legend()
    plt.grid()
    plt.savefig("results/latency_comparison.png")

    # Speedup curve
    plt.figure()
    plt.plot(n_values, speedups, marker='o')
    plt.xlabel("Number of Candidates")
    plt.ylabel("Speedup (x)")
    plt.title("PDC Optimization Speedup (OpenMP + SIMD)")
    plt.grid()
    plt.savefig("results/speedup_curve.png")

    print("\nPlots saved in results/")


# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(n_values, baseline, optimized, speedups, std_dev):

    data = {
        "project": config.PROJECT_NAME,
        "description": config.DESCRIPTION,
        "candidate_sizes": n_values,
        "baseline": baseline,
        "optimized": optimized,
        "speedups": speedups,
        "std_dev_optimized": std_dev,
        "pdc_mapping": config.PDC_MAPPING
    }

    with open("results/timings.json", "w") as f:
        json.dump(data, f, indent=4)


# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == "__main__":

    print(f"Running: {config.PROJECT_NAME}")
    print(config.DESCRIPTION)

    candidate_sizes = config.CANDIDATE_SIZES

    baseline, optimized, speedups, std_dev = run_stable_experiment(candidate_sizes)

    save_results(candidate_sizes, baseline, optimized, speedups, std_dev)

    plot_results(candidate_sizes, baseline, optimized, speedups)
