import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.medusa_optimized.medusa_model import MedusaModel


# =========================================================
# GPU SAFE TIMER
# =========================================================
def measure_time(func, *args, **kwargs):
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    output = func(*args, **kwargs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    return end - start, output


# =========================================================
# RUN SINGLE EXPERIMENT
# =========================================================
def run_experiment(model, input_ids, use_pdc, label):
    times = []

    print(f"\nRunning: {label}")

    for _ in range(3):  # small repeats for stability

        def run():
            return model.medusa_generate(
                input_ids=input_ids,
                max_steps=30,          # controlled for fairness
                temperature=0.0,
                sampling="typical",
                fast=True,
                use_pdc_opt=use_pdc
            )

        t, _ = measure_time(run)
        times.append(t)

    return np.mean(times), np.std(times)


# =========================================================
# MAIN BENCHMARK
# =========================================================
def main():

    print("\nLoading Medusa Model...")

    model = MedusaModel.from_pretrained(
        "lmsys/vicuna-7b-v1.3"  # change if needed
    )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = model.get_tokenizer()

    prompt = "Explain parallel computing in simple terms."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # =====================================================
    # BASELINE
    # =====================================================
    baseline_time, baseline_std = run_experiment(
        model,
        input_ids,
        use_pdc=False,
        label="Baseline Medusa"
    )

    # =====================================================
    # OPTIMIZED
    # =====================================================
    opt_time, opt_std = run_experiment(
        model,
        input_ids,
        use_pdc=True,
        label="PDC Optimized Medusa"
    )

    # =====================================================
    # RESULTS
    # =====================================================
    speedup = baseline_time / opt_time if opt_time > 0 else 0

    print("\n==============================")
    print(f"Baseline Time   : {baseline_time:.4f} s ± {baseline_std:.4f}")
    print(f"Optimized Time  : {opt_time:.4f} s ± {opt_std:.4f}")
    print(f"Speedup         : {speedup:.2f}x")
    print("==============================")

    # =====================================================
    # PLOT
    # =====================================================
    labels = ["Baseline", "Optimized"]
    values = [baseline_time, opt_time]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Latency (seconds)")
    plt.title("Medusa Inference Performance Comparison")
    plt.savefig("results/benchmark_plot.png")

    print("\nPlot saved to results/benchmark_plot.png")


if __name__ == "__main__":
    main()
