import torch
import multiprocessing as mp
import torch.nn.functional as F

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

# =========================================================
# SINGLE CANDIDATE SCORING (CPU WORKER)
# =========================================================
def score_single_candidate(args):
    """
    Computes accept_length + likelihood for one candidate.
    This is the ONLY part we parallelize (safe PDC region).
    """

    idx, candidates, candidates_prob, posterior_mask = args

    mask = posterior_mask[idx]
    prob = candidates_prob[idx]

    # ---------------------------------------------
    # Accept length (prefix validity)
    # ---------------------------------------------
    accept_length = torch.cumprod(mask, dim=0).sum().item()

    # ---------------------------------------------
    # Likelihood computation (log probability sum)
    # ---------------------------------------------
    likelihood = 0.0
    if accept_length > 0:
        likelihood = torch.sum(
            torch.log(prob[:accept_length] + 1e-8)
        ).item()

    return idx, accept_length, likelihood


# =========================================================
# OPTIMIZED POSTERIOR EVALUATION (PDC VERSION)
# =========================================================
def optimized_evaluate_posterior(
    logits,
    candidates,
    temperature,
    posterior_threshold=0.3,
    posterior_alpha=0.09,
    top_p=0.8,
    sampling="typical",
    fast=True,
    num_workers=4
):
    """
    Drop-in replacement of Medusa evaluate_posterior()
    Only optimizes candidate-level scoring using CPU parallelism.
    """

    # =====================================================
    # CASE 1: GREEDY (UNCHANGED MEDUSA LOGIC)
    # =====================================================
    if temperature == 0:
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()

        candidates_accept_length = torch.cumprod(
            posterior_mask, dim=1
        ).sum(dim=1)

        accept_length = candidates_accept_length.max()

        if accept_length == 0:
            best_candidate = torch.tensor(
                0, dtype=torch.long, device=candidates.device
            )
        else:
            best_candidate = torch.argmax(
                candidates_accept_length
            ).to(torch.long)

        return best_candidate, accept_length


    # =====================================================
    # CASE 2: TYPICAL SAMPLING (OPTIMIZED REGION)
    # =====================================================
    if sampling == "typical":

        # -------------------------------------------------
        # Step 1: probability computation (GPU - unchanged)
        # -------------------------------------------------
        posterior_prob = torch.softmax(
            logits[:, :-1] / temperature, dim=-1
        )

        candidates_prob = torch.gather(
            posterior_prob,
            dim=-1,
            index=candidates[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        posterior_entropy = -torch.sum(
            posterior_prob * torch.log(posterior_prob + 1e-5),
            dim=-1
        )

        threshold = torch.minimum(
            torch.ones_like(posterior_entropy) * posterior_threshold,
            torch.exp(-posterior_entropy) * posterior_alpha,
        )

        posterior_mask = candidates_prob > threshold

        # =================================================
        # STEP 2: PDC PARALLEL REGION (CORE CONTRIBUTION)
        # =================================================
        args = [
            (i, candidates, candidates_prob, posterior_mask)
            for i in range(len(candidates))
        ]

        with mp.Pool(num_workers) as pool:
            results = pool.map(score_single_candidate, args)

        # -------------------------------------------------
        # unpack results
        # -------------------------------------------------
        idxs, lengths, likelihoods = zip(*results)

        accept_length = max(lengths)

        # -------------------------------------------------
        # Step 3: selection logic (same as Medusa)
        # -------------------------------------------------
        if accept_length == 0:
            best_candidate = torch.tensor(
                0, dtype=torch.long, device=candidates.device
            )
        else:
            valid = [
                i for i, l in enumerate(lengths)
                if l == accept_length
            ]

            best_candidate = valid[
                max(range(len(valid)), key=lambda i: likelihoods[valid[i]])
            ]

            best_candidate = torch.tensor(
                best_candidate, device=candidates.device
            )

        return best_candidate, accept_length


    # =====================================================
    # CASE 3: NUCLEUS (UNCHANGED MEDUSA LOGIC)
    # =====================================================
    if sampling == "nucleus":

        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"

        posterior_mask = get_nucleus_posterior_mask(
            logits, candidates, temperature, top_p
        )

        candidates_accept_length = torch.cumprod(
            posterior_mask, dim=1
        ).sum(dim=1)

        accept_length = candidates_accept_length.max()

        if accept_length == 0:
            best_candidate = torch.tensor(
                0, dtype=torch.long, device=candidates.device
            )
        else:
            best_candidate = torch.argmax(
                candidates_accept_length
            ).to(torch.long)

        return best_candidate, accept_length


    raise NotImplementedError("Unsupported sampling mode")