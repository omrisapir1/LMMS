from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_softmax


@dataclass
class LossOutputs:
    total: torch.Tensor
    answer: torch.Tensor
    answer_sft: torch.Tensor
    softz: torch.Tensor
    counterfactual: torch.Tensor
    usage: torch.Tensor


class AnswerDigitLoss(nn.Module):
    """Digit CE with optional per-position keep-prob masking for label==0."""

    def __init__(self, keep_prob: Optional[Union[Mapping[int, float], Sequence[float]]] = None) -> None:
        super().__init__()
        self.keep_prob = self._normalize_keep_prob(keep_prob)
        self._warned_fallback = False

    @staticmethod
    def _normalize_keep_prob(
        keep_prob: Optional[Union[Mapping[int, float], Sequence[float]]]
    ) -> Sequence[float]:
        if keep_prob is None:
            return [1.0] * 5

        if isinstance(keep_prob, Mapping):
            keys = set(int(k) for k in keep_prob.keys())
            if keys == {0, 1, 2, 3, 4}:
                vals = [float(keep_prob[i]) for i in range(5)]
            elif keys == {1, 2, 3, 4, 5}:
                vals = [float(keep_prob[i]) for i in range(1, 6)]
            else:
                raise ValueError("keep_prob mapping keys must be 0..4 or 1..5")
        else:
            vals = [float(x) for x in keep_prob]
            if len(vals) != 5:
                raise ValueError("keep_prob sequence must have length 5")

        for p in vals:
            if not (0.0 <= p <= 1.0):
                raise ValueError("keep_prob values must be in [0,1]")
        return vals

    def forward(
        self,
        digit_logits: torch.Tensor,  # [B,5,10]
        digit_labels: torch.Tensor,  # [B,5]
    ) -> torch.Tensor:
        if digit_logits.ndim != 3 or digit_logits.shape[1:] != (5, 10):
            raise ValueError("digit_logits must be [B,5,10]")
        if digit_labels.ndim != 2 or digit_labels.shape[1] != 5:
            raise ValueError("digit_labels must be [B,5]")

        total = digit_logits.new_zeros(())
        contributed = 0
        for i in range(5):
            logits_i = digit_logits[:, i, :]
            labels_i = digit_labels[:, i]
            is_zero = (labels_i == 0)

            kp = float(self.keep_prob[i])
            if kp >= 1.0:
                include_zero = torch.ones_like(is_zero, dtype=torch.bool)
            elif kp <= 0.0:
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
            else:
                probs = torch.full(labels_i.shape, kp, device=labels_i.device, dtype=torch.float)
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
                if is_zero.any():
                    include_zero[is_zero] = torch.bernoulli(probs[is_zero]).bool()

            include = (~is_zero) | (is_zero & include_zero)
            if include.any():
                total = total + F.cross_entropy(logits_i[include], labels_i[include], reduction="mean")
                contributed += 1

        if contributed == 0:
            if not self._warned_fallback:
                warnings.warn(
                    "AnswerDigitLoss fallback activated: no digit contributed after keep-prob mask.",
                    stacklevel=2,
                )
                self._warned_fallback = True
            per_digit = [
                F.cross_entropy(digit_logits[:, i, :], digit_labels[:, i], reduction="mean")
                for i in range(5)
            ]
            return torch.stack(per_digit).mean()

        return total / float(contributed)


class AnswerTokenSFTLoss(nn.Module):
    """CE on the next-token logits right before <ANSWER>, target is <ANSWER>."""

    def __init__(self, answer_token_id: int) -> None:
        super().__init__()
        self.answer_token_id = int(answer_token_id)

    def forward(self, answer_next_logits: torch.Tensor) -> torch.Tensor:
        if answer_next_logits.ndim != 2:
            raise ValueError("answer_next_logits must be [B,V]")
        labels = torch.full(
            (answer_next_logits.size(0),),
            self.answer_token_id,
            dtype=torch.long,
            device=answer_next_logits.device,
        )
        return F.cross_entropy(answer_next_logits, labels)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


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
        p_z: torch.Tensor,      # [B,Kmax,V]
        k_vals: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        bsz, kmax, vocab = p_z.shape
        device = p_z.device
        out = p_z.clone()

        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        for b in range(bsz):
            k = int(k_vals[b].item())
            if k <= 0:
                continue
            if k > kmax:
                raise RuntimeError(f"k_vals[{b}]={k} exceeds p_z slots={kmax}")

            # Counterfactual operates on discrete latent actions (slot-wise argmax indices).
            active_idx = p_z[b, :k].argmax(dim=-1)  # [k]
            prob = self.permute_prob.get(k, 0.0)

            if k > 1 and torch.rand((), device=device, generator=gen) < prob:
                perm = torch.randperm(k, device=device, generator=gen)
                cf_idx = active_idx[perm]
            else:
                cf_idx = torch.randint(
                    low=0,
                    high=vocab,
                    size=(k,),
                    device=device,
                    generator=gen,
                )
            out[b, :k] = F.one_hot(cf_idx, num_classes=vocab).to(dtype=p_z.dtype)

        return out

    def forward(
        self,
        *,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        p_z: torch.Tensor,
        k_vals: torch.Tensor,
    ) -> torch.Tensor:
        # Reference path is deterministic and must not apply extra gumbel noise.
        digit_logits_ref_det = model.forward_with_fixed_z_distributions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            p_z=p_z.detach(),
        )
        p_ref = safe_softmax(digit_logits_ref_det, tau=self.digit_temperature, dim=-1).detach()

        p_cf_z = self._build_counterfactual_pz(p_z, k_vals)
        digit_logits_cf = model.forward_with_fixed_z_distributions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            p_z=p_cf_z,
        )
        p_cf = safe_softmax(digit_logits_cf, tau=self.digit_temperature, dim=-1)
        mean_abs_logit_diff = (digit_logits_ref_det.detach() - digit_logits_cf.detach()).abs().mean().item()
        # also useful: argmax disagreement rate across all digits
        ref_arg = digit_logits_ref_det.detach().argmax(dim=-1)  # [B,5]
        cf_arg = digit_logits_cf.detach().argmax(dim=-1)  # [B,5]
        disagree = (ref_arg != cf_arg).float().mean().item()
        print(
            f"[cf_loss] |logit_diff|={mean_abs_logit_diff:.6f} "
            f"disagree={disagree:.3f} tau={self.digit_temperature:.3f}"
        )
        js = js_divergence(p_ref, p_cf).mean()
        # Negative because we want to maximize divergence.
        return 1 -js


def usage_shaping_loss_stub(*, device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)
