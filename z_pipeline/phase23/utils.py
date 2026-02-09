from __future__ import annotations

import random
from typing import List, Tuple, Optional

import torch

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
FIRST_PART_PROMPT = """<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
<|im_start|>user
"""
SECOND_PART_PROMPT = """<|im_end|>
<|im_start|>assistant
"""


def build_prompt(tokenizer, question: str) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return FIRST_PART_PROMPT + question + SECOND_PART_PROMPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suffix_pad(
    sequences: List[torch.Tensor],
    pad_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Suffix-pad a list of 1D tensors to the same length.
    Returns (padded, attention_mask).
    """
    max_len = max(int(x.numel()) for x in sequences)
    batch = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = int(seq.numel())
        batch[i, :n] = seq
        mask[i, :n] = 1
    return batch, mask


def effective_vocab_size(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute exp(entropy(mean_p)).
    Accepts:
      - p: [B, V]
      - p: [B, K, V]  (averages over K first)
    """
    if p.dim() == 3:
        p = p.mean(dim=1)
    mean_p = p.mean(dim=0).clamp_min(eps)
    entropy = -(mean_p * mean_p.log()).sum(dim=-1)
    return entropy.exp()


def safe_softmax(logits: torch.Tensor, tau: float = 1.0, dim: int = -1) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return torch.softmax(logits / tau, dim=dim)


def gumbel_tau_at_step(
    *,
    step: int,
    tau_start: float,
    tau_end: float,
    tau_anneal_steps: int,
) -> float:
    """
    Linear anneal with floor at tau_end.
    """
    if tau_anneal_steps <= 0:
        return float(tau_end)
    t = float(step) / float(tau_anneal_steps)
    tau = float(tau_start) - (float(tau_start) - float(tau_end)) * t
    return float(max(float(tau_end), tau))


def permute_slots(p_z: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Permute order of Z distributions across the first k slots.
    p_z: [K, V] or [k, V]
    """
    if k is None:
        k = int(p_z.size(0))
    if k <= 1:
        return p_z
    idx = torch.randperm(k, device=p_z.device)
    out = p_z.clone()
    out[:k] = out[:k][idx]
    return out


def random_mixture_like(
    p_z: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Create random mixtures for the first k slots.
    Returns tensor of same shape as p_z, padded slots zeroed.
    """
    Kmax, V = p_z.shape
    out = torch.zeros_like(p_z)
    if k <= 0:
        return out
    samples = torch.rand(
        (k, V),
        device=p_z.device,
        dtype=p_z.dtype,
        generator=generator,
    )
    if alpha != 1.0:
        samples = samples.pow(1.0 / max(float(alpha), 1e-6))
    samples = samples / samples.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    out[:k] = samples
    return out


def k_to_bucket(k: int) -> str:
    if k == 1:
        return "K1"
    if k == 2:
        return "K2"
    if k == 3:
        return "K3"
    if 4 <= k <= 7:
        return "K4_7"
    if 8 <= k <= 12:
        return "K8_12"
    if 13 <= k <= 20:
        return "K13_20"
    raise ValueError(f"K out of range: {k}")
