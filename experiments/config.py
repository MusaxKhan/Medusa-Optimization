"""
config.py
=========
Central configuration for all benchmark modules.
All experiments/ files import from here — change one value, it updates everywhere.
"""

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE     = "cpu"

# ── Benchmark settings ────────────────────────────────────────────────────────
NUM_TRIALS      = 15          # timing runs per config; first is discarded as warmup
CANDIDATE_SIZES = [4, 8, 16, 32, 64]

# ── Posterior parameters — MUST match between baseline and optimized ──────────
TEMPERATURE     = 0.7
POST_THRESHOLD  = 0.3
POST_ALPHA      = 0.09
TOP_P           = 0.8
SAMPLING        = "typical"

# ── Amdahl fractions ──────────────────────────────────────────────────────────
# What fraction of total Medusa runtime is evaluate_posterior?
# Three conservative estimates used for Amdahl projection.
PIPELINE_FRACTIONS = {
    "10% of pipeline": 0.10,
    "20% of pipeline": 0.20,
    "35% of pipeline": 0.35,
}

# ── Complexity analysis constants ─────────────────────────────────────────────
VOCAB_SIZE_ANALYSIS = 32000   # TinyLlama / Vicuna vocab size
SEQ_LEN_ANALYSIS    = 10      # approximate T used for op-count display

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = [
    "Explain parallel computing in simple terms.",
    "What is the difference between a process and a thread?",
    "Describe how a CPU cache works.",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR     = os.path.join(EXPERIMENTS_DIR, "..", "results")