import torch


def cpu_score_candidates(candidate_probs):
    """
    Simulated CPU-side scoring stage.

    PDC concept:
    - OpenMP-style reduction (vectorized CPU ops)
    - KV-cache locality approximation (FastDecode idea)
    """

    candidate_probs = candidate_probs.contiguous()
    return torch.sum(candidate_probs, dim=1)