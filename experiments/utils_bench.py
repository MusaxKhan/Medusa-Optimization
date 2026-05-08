"""
utils_bench.py
==============
Shared utility functions used by all benchmark modules.

  - build_realistic_candidates()  : builds candidates from model top-k predictions
  - timed()                       : multi-trial timing with warmup discard
  - load_model()                  : loads tokenizer + model once and returns them
"""

import sys
import os
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running any experiments/ file directly or via -m
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config import (
    MODEL_NAME, DEVICE, NUM_TRIALS,
    TEMPERATURE, POST_THRESHOLD, POST_ALPHA, TOP_P, SAMPLING,
    CANDIDATE_SIZES,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """
    Load tokenizer and model. Called once at the start of each benchmark script.

    Returns:
        tokenizer : HuggingFace tokenizer
        model     : LM in eval mode on DEVICE
    """
    print(f"  Loading model: {MODEL_NAME}  (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.to(DEVICE)
    model.eval()
    print("  Model loaded.\n")
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
def build_realistic_candidates(raw_logits, input_ids, num_candidates):
    """
    Build candidate token sequences that have realistic acceptance probability.

    Why this matters:
        evaluate_posterior checks whether candidates[:,1:] match the model's
        own predictions from logits[:,:-1]. Random token sequences always give
        AcceptLen = 0, making the benchmark measure nothing meaningful.

    Strategy:
        Candidate 0  : exact greedy (argmax) sequence  → guaranteed max AcceptLen
        Candidates 1+: i-th best alternative at each position via top-k

    Args:
        raw_logits    : (1, T, V) raw model logits
        input_ids     : (1, T)   input token ids
        num_candidates: int

    Returns:
        candidates : (C, T) long tensor
    """
    V = raw_logits.shape[2]
    k = min(num_candidates, V)

    greedy       = torch.argmax(raw_logits[0, :-1], dim=-1)              # (T-1,)
    topk_per_pos = torch.topk(raw_logits[0, :-1], k=k, dim=-1).indices   # (T-1, k)

    candidates = []
    for i in range(num_candidates):
        cand      = input_ids[0].clone()
        cand[1:]  = greedy if i == 0 else topk_per_pos[:, i % k]
        candidates.append(cand)

    return torch.stack(candidates, dim=0)   # (C, T)


# ─────────────────────────────────────────────────────────────────────────────
def timed(fn, *args, trials=NUM_TRIALS, **kwargs):
    """
    Run fn(*args, **kwargs) trials+1 times.
    Discards the first run (warmup — eliminates JIT / OS scheduling artifacts).
    Returns (mean_sec, std_sec, last_result).
    """
    times, result = [], None
    for i in range(trials + 1):
        t0      = time.perf_counter()
        result  = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)
    return float(np.mean(times)), float(np.std(times)), result


# ─────────────────────────────────────────────────────────────────────────────
def get_logits(model, tokenizer, prompt):
    """
    Tokenize prompt and run one forward pass.
    Returns (raw_logits, input_ids) where raw_logits is (1, T, V).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits, inputs["input_ids"]