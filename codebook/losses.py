from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    vq_loss: torch.Tensor
    commit_loss: torch.Tensor
    kl_loss: torch.Tensor
    adjacent_overlap: torch.Tensor
    adjacent_overlap_loss: torch.Tensor
    perplexity: torch.Tensor
    dead_fraction: torch.Tensor


class UsageKLLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        alpha: float = 1.0,
        eta: float = 0.99,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not (0.0 < eta < 1.0):
            raise ValueError("eta must be in (0, 1)")

        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.eta = float(eta)

        self.register_buffer(
            "p_ema",
            torch.full((self.vocab_size,), 1.0 / float(self.vocab_size), dtype=torch.float32),
        )

    @torch.no_grad()
    def update(self, z_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_ids.ndim != 1:
            raise ValueError("z_ids must have shape [N]")
        if z_ids.numel() == 0:
            zero = z_ids.new_zeros((), dtype=torch.float32)
            return zero, torch.exp(zero), zero

        counts = torch.bincount(z_ids.to(dtype=torch.long), minlength=self.vocab_size).to(torch.float32)
        total = counts.sum()
        p_batch = (counts + self.alpha) / (total + self.alpha * self.vocab_size)

        self.p_ema.mul_(self.eta).add_(p_batch, alpha=1.0 - self.eta)
        self.p_ema.div_(self.p_ema.sum().clamp_min(1e-12))

        p = self.p_ema.clamp_min(1e-12)
        kl = torch.sum(p * (torch.log(p) + math.log(float(self.vocab_size))))
        entropy = -torch.sum(p * torch.log(p))
        perplexity = torch.exp(entropy)
        dead_fraction = (self.p_ema < 1e-5).to(torch.float32).mean()
        return kl, perplexity, dead_fraction


def compute_losses(
    *,
    latents: torch.Tensor,
    quantized_vectors: torch.Tensor,
    z_ids: torch.Tensor,
    usage_kl: UsageKLLoss,
    beta: float = 0.25,
    lambda_kl: float = 0.01,
    similarity: torch.Tensor | None = None,
    sequence_lengths: Sequence[int] | None = None,
    tau: float = 0.1,
    lambda_adjacent_overlap: float = 0.0,
) -> LossOutput:
    if latents.shape != quantized_vectors.shape:
        raise ValueError("latents and quantized_vectors must have the same shape")

    latents = latents.to(dtype=torch.float32)
    quantized_vectors = quantized_vectors.to(dtype=torch.float32)

    # || sg(latent) - embedding[z] ||^2
    vq_loss = torch.mean((latents.detach() - quantized_vectors) ** 2)
    # beta * || latent - sg(embedding[z]) ||^2
    commit_loss = float(beta) * torch.mean((latents - quantized_vectors.detach()) ** 2)

    kl_raw, perplexity, dead_fraction = usage_kl.update(z_ids)
    kl_loss = float(lambda_kl) * kl_raw

    adjacent_overlap = latents.new_zeros(())
    if similarity is not None and sequence_lengths is not None:
        adjacent_overlap = compute_soft_adjacent_overlap(
            similarity=similarity,
            sequence_lengths=sequence_lengths,
            tau=tau,
        )
    adjacent_overlap_loss = float(lambda_adjacent_overlap) * adjacent_overlap

    total_loss = vq_loss * 0.01 + commit_loss* 0.01 + kl_loss + adjacent_overlap_loss
    return LossOutput(
        total_loss=total_loss,
        vq_loss=vq_loss,
        commit_loss=commit_loss,
        kl_loss=kl_loss,
        adjacent_overlap=adjacent_overlap,
        adjacent_overlap_loss=adjacent_overlap_loss,
        perplexity=perplexity,
        dead_fraction=dead_fraction,
    )


def compute_soft_adjacent_overlap(
    *,
    similarity: torch.Tensor,
    sequence_lengths: Sequence[int],
    tau: float,
) -> torch.Tensor:
    if similarity.ndim != 2:
        raise ValueError("similarity must have shape [N, V]")
    if tau <= 0.0:
        raise ValueError("tau must be > 0")

    n = int(similarity.shape[0])
    expected = sum(int(k) for k in sequence_lengths)
    if expected != n:
        raise ValueError(f"sum(sequence_lengths)={expected} must equal similarity.shape[0]={n}")

    if n == 0:
        return similarity.new_zeros(())

    probs = torch.softmax(similarity / float(tau), dim=-1)  # [N, V]
    overlap_sum = similarity.new_zeros(())
    pair_count = 0
    start = 0
    for k_raw in sequence_lengths:
        k = int(k_raw)
        if k < 0:
            raise ValueError("sequence lengths must be non-negative")
        if k >= 2:
            seq_probs = probs[start : start + k]
            overlaps = torch.sum(seq_probs[1:] * seq_probs[:-1], dim=-1)  # [k-1]
            overlap_sum = overlap_sum + overlaps.sum()
            pair_count += k - 1
        start += k

    if pair_count == 0:
        return similarity.new_zeros(())
    return overlap_sum / float(pair_count)
