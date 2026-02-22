from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class AnswerLossOutput:
    loss: torch.Tensor
    contributing_positions: int
    fallback_used: bool


class AnswerLoss:
    """
    Masked autoregressive CE over the five digit targets after <ANSWER>.
    Trailing-zero downsampling:
    - non-zero digits are always kept
    - zero digits are kept with probability keep_prob[i]
    """

    def __init__(self, keep_prob: Sequence[float]) -> None:
        probs = [float(x) for x in keep_prob]
        if len(probs) != 5:
            raise ValueError("keep_prob must contain 5 probabilities.")
        for p in probs:
            if not (0.0 <= p <= 1.0):
                raise ValueError("keep_prob values must be in [0,1].")
        self.keep_prob = probs

    def compute(
        self,
        *,
        logits: torch.Tensor,
        digit_values: torch.Tensor,
        downsample_zeros: bool,
        seed: Optional[int] = None,
    ) -> AnswerLossOutput:
        if logits.ndim != 3:
            raise ValueError("logits must have shape [B,5,10].")
        if logits.shape[1] != 5 or logits.shape[2] != 10:
            raise ValueError(f"logits must have shape [B,5,10], got {tuple(logits.shape)}.")
        if digit_values.ndim != 2 or digit_values.shape[-1] != 5:
            raise ValueError("digit_values must have shape [B,5].")
        if logits.shape[0] != digit_values.shape[0]:
            raise ValueError("Batch size mismatch between logits and digit_values.")
        if bool((digit_values < 0).any().item()) or bool((digit_values > 9).any().item()):
            raise ValueError("digit_values must contain integers in [0,9].")

        targets = digit_values.to(device=logits.device, dtype=torch.long)

        ce = F.cross_entropy(
            logits.reshape(-1, 10),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

        keep_mask_5 = self.sample_digit_keep_mask(
            digit_values=digit_values,
            downsample_zeros=downsample_zeros,
            seed=seed,
            device=logits.device,
        )

        effective_mask = keep_mask_5.to(device=logits.device, dtype=torch.bool)
        contributing = int(effective_mask.sum().item())
        if contributing > 0:
            return AnswerLossOutput(
                loss=ce[effective_mask].mean(),
                contributing_positions=contributing,
                fallback_used=False,
            )

        # Fallback: mean over original 5 digit positions (without downsampling).
        base_mask = torch.ones_like(effective_mask, dtype=torch.bool)
        base_count = int(base_mask.sum().item())
        if base_count == 0:
            return AnswerLossOutput(
                loss=ce.new_zeros(()),
                contributing_positions=0,
                fallback_used=True,
            )
        return AnswerLossOutput(
            loss=ce[base_mask].mean(),
            contributing_positions=base_count,
            fallback_used=True,
        )

    def sample_digit_keep_mask(
        self,
        *,
        digit_values: torch.Tensor,
        downsample_zeros: bool,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Returns boolean [B,5]:
        - non-zero labels always True
        - zero labels sampled with Bernoulli(keep_prob[i]) when downsample_zeros=True
        """
        if digit_values.ndim != 2 or digit_values.shape[1] != 5:
            raise ValueError("digit_values must have shape [B,5].")

        keep_mask = torch.ones_like(digit_values, dtype=torch.bool)
        if not downsample_zeros:
            return keep_mask

        if seed is None:
            seed = 0
        target_device = device if device is not None else digit_values.device
        gen = torch.Generator(device=target_device)
        gen.manual_seed(int(seed))

        bsz = int(digit_values.shape[0])
        for i in range(5):
            zero_mask = digit_values[:, i] == 0
            if not bool(zero_mask.any().item()):
                continue
            draws = torch.rand(
                (bsz,),
                generator=gen,
                device=target_device,
                dtype=torch.float,
            ) < float(self.keep_prob[i])
            draws = draws.to(device=digit_values.device)
            keep_mask[:, i] = (~zero_mask) | draws
        return keep_mask

def permutation_sensitivity_loss(
    *,
    logits_orig: torch.Tensor,
    logits_aux: torch.Tensor,
    eligible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric-KL on digit distributions, then exp(-sym_kl).
    Applies only to samples where eligible_mask is True (typically latent_count >= 2).
    """
    if logits_aux is None:
        return logits_orig.new_zeros(())

    if eligible_mask.ndim != 1:
        raise ValueError("eligible_mask must be shape [B].")

    valid = eligible_mask.to(torch.bool)
    if not bool(valid.any().item()):
        return logits_orig.new_zeros(())

    if logits_orig.ndim != 3 or logits_aux.ndim != 3:
        raise ValueError("logits_orig/logits_aux must have shape [B,5,10].")
    if logits_orig.shape[1:] != (5, 10):
        raise ValueError("logits_orig must have shape [B,5,10].")
    if logits_aux.shape != logits_orig.shape:
        raise ValueError("logits_aux must match logits_orig shape [B,5,10].")

    d_orig = logits_orig[valid]
    d_aux = logits_aux[valid]

    p = F.softmax(d_orig, dim=-1).clamp_min(eps)
    q = F.softmax(d_aux, dim=-1).clamp_min(eps)

    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    sym_kl = (kl_pq + kl_qp).mean(dim=1)
    return torch.exp(-sym_kl).mean()
