from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import permute_slots, random_mixture_like, safe_softmax


# ─────────────────────────────────────────────
# Loss outputs container
# ─────────────────────────────────────────────

@dataclass
class LossOutputs:
    total: torch.Tensor
    answer: torch.Tensor
    softz: torch.Tensor
    counterfactual: torch.Tensor
    usage: torch.Tensor


# ─────────────────────────────────────────────
# Answer digit loss
# ─────────────────────────────────────────────

class AnswerDigitLoss(nn.Module):
    def __init__(self, keep_prob: Optional[List[float]] = None) -> None:
        super().__init__()
        if keep_prob is None:
            keep_prob = [1.0] * 5
        try:
            n = len(keep_prob)
        except Exception as e:
            raise ValueError("keep_prob must be an iterable of length 5") from e
        if n != 5:
            raise ValueError("keep_prob must have length 5")
        for p in keep_prob:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError("keep_prob values must be in [0,1]")
        self.keep_prob = [float(x) for x in keep_prob]
        self._warned_fallback = False

    def forward(
        self,
        digit_logits: torch.Tensor,   # [B,5,10]
        digit_labels: torch.Tensor,   # [B,5]
    ) -> torch.Tensor:
        if digit_logits.ndim != 3 or digit_logits.shape[1] != 5 or digit_logits.shape[2] != 10:
            raise ValueError("digit_logits must be [B, 5, 10]")
        if digit_labels.ndim != 2 or digit_labels.shape[1] != 5:
            raise ValueError("digit_labels must be [B, 5]")

        total_loss = digit_logits.new_zeros(())
        contributed = 0
        for i in range(5):
            li = digit_logits[:, i, :]
            yi = digit_labels[:, i]
            is_zero = (yi == 0)
            if self.keep_prob[i] >= 1.0:
                include_zero = torch.ones_like(is_zero, dtype=torch.bool)
            elif self.keep_prob[i] <= 0.0:
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
            else:
                probs = torch.full(yi.shape, self.keep_prob[i], device=yi.device, dtype=torch.float)
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
                if is_zero.any():
                    include_zero[is_zero] = torch.bernoulli(probs[is_zero]).bool()
            include_mask = (~is_zero) | (is_zero & include_zero)
            if include_mask.any():
                total_loss = total_loss + F.cross_entropy(li[include_mask], yi[include_mask], reduction="mean")
                contributed += 1

        if contributed == 0:
            if not self._warned_fallback:
                warnings.warn("AnswerDigitLoss fallback activated: no digits included after masking", stacklevel=2)
                self._warned_fallback = True
            per_digit = [
                F.cross_entropy(digit_logits[:, i, :], digit_labels[:, i], reduction="mean")
                for i in range(5)
            ]
            return torch.stack(per_digit).mean()

        return total_loss / float(contributed)


# ─────────────────────────────────────────────
# Self-distilled Z KL loss
# ─────────────────────────────────────────────

def self_distill_z_kl_loss(
    p_student: torch.Tensor,         # [B,K,V]
    q_teacher: torch.Tensor,          # [B,K,V]
    mask: Optional[torch.Tensor] = None,  # [B,K]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL(q_teacher || p_student), averaged over valid latent slots.
    """
    q = q_teacher.clamp_min(eps)
    p = p_student.clamp_min(eps)

    kl = (q * (q.log() - p.log())).sum(dim=-1)  # [B,K]

    if mask is not None:
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = kl.numel()

    return kl.sum() / denom


# ─────────────────────────────────────────────
# JS divergence helper
# ─────────────────────────────────────────────

def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


# ─────────────────────────────────────────────
# Counterfactual answer loss
# ─────────────────────────────────────────────

class CounterfactualAnswerLoss(nn.Module):
    def __init__(
        self,
        *,
        permute_prob: Dict[int, float],
        digit_temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.permute_prob = {int(k): float(v) for k, v in permute_prob.items()}
        self.digit_temperature = float(digit_temperature)
        self.seed = seed

    def _build_counterfactual_pz(
        self,
        p_z: torch.Tensor,     # [B,Kmax,V]
        k_vals: torch.Tensor, # [B]
    ) -> torch.Tensor:
        """
        Build counterfactual Z distributions.

        Only the first k slots are modified for each example.
        Padded slots are always zero.

        Strategy per example:
          - with probability permute_prob[k]: permute the first k slot distributions
          - otherwise: replace the first k slots with random mixtures
        """
        B, Kmax, V = p_z.shape
        device = p_z.device
        out = torch.zeros_like(p_z)

        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        for b in range(B):
            k = int(k_vals[b].item())
            if k <= 0:
                continue
            if k > Kmax:
                raise RuntimeError(f"k_vals[{b}]={k} exceeds p_z Kmax={Kmax}")

            active = p_z[b, :k]  # [k,V]

            prob = self.permute_prob.get(k, 0.0)
            if k > 1 and torch.rand((), device=device, generator=gen) < prob:
                out[b, :k] = permute_slots(active, k)
            else:
                # sample random mixtures for exactly k slots
                samples = torch.rand(
                    (k, V),
                    device=device,
                    dtype=p_z.dtype,
                    generator=gen,
                )
                samples = samples / samples.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                out[b, :k] = samples

        return out

    def forward(
        self,
        *,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_logits_ref: torch.Tensor,  # [B,5,10]
        p_z: torch.Tensor,               # [B,Kmax,V]
        k_vals: torch.Tensor,            # [B]
    ) -> torch.Tensor:
        # Reference digit distributions (detached)
        p_ref = safe_softmax(
            digit_logits_ref,
            tau=self.digit_temperature,
            dim=-1,
        ).detach()

        # Counterfactual Z distributions
        p_cf_z = self._build_counterfactual_pz(p_z, k_vals)

        # Forward with fixed Z mixtures
        digit_logits_cf = model.forward_with_fixed_z_distributions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            p_z=p_cf_z,
        )

        p_cf = safe_softmax(
            digit_logits_cf,
            tau=self.digit_temperature,
            dim=-1,
        )

        # Maximize divergence
        js = js_divergence(p_ref, p_cf).mean()
        # Negative because we want to MAXIMIZE divergence
        return -js


# ─────────────────────────────────────────────
# Usage shaping (stub)
# ─────────────────────────────────────────────

def usage_shaping_loss_stub(
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(0.0, device=device)
