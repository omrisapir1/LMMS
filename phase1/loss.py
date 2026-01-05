"""
Phase 1 AnswerLoss
- Applies trailing-zero downsampling via per-digit Bernoulli masking.
- Loss is stochastic across batches; effective zero-rate matches keep_prob in expectation.
- Digit labels are assumed MSB-first; head index i corresponds to digit position i.
"""
import warnings
import torch
import torch.nn.functional as F
from typing import Optional, List


class AnswerLoss:
    def __init__(self, keep_prob: Optional[List[float]] = None):
        """
        keep_prob: Optional[List[float]] of length 5, values in [0,1].
        If None, defaults to [1.0] * 5 (no downsampling).
        """
        if keep_prob is None:
            keep_prob = [1.0] * 5
        # Ensure iterable and correct length
        try:
            n = len(keep_prob)
        except Exception as e:
            raise ValueError("keep_prob must be an iterable of length 5") from e
        if n != 5:
            raise ValueError("keep_prob must have length 5")
        # Validate values are within [0,1]
        for p in keep_prob:
            if not (0.0 <= p <= 1.0):
                raise ValueError("keep_prob values must be in [0,1]")
        self.keep_prob = keep_prob
        # Gate fallback warnings to avoid log spam
        self._warned_fallback = False

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 5, 10] float tensor (per-digit class logits)
        labels: [B, 5] long tensor with digits 0..9 (MSB-first)
        Returns: scalar loss tensor
        """
        if logits.ndim != 3 or logits.shape[1] != 5 or logits.shape[2] != 10:
            raise ValueError("logits must be [B, 5, 10]")
        if labels.ndim != 2 or labels.shape[1] != 5:
            raise ValueError("labels must be [B, 5]")
        # Initialize loss as scalar tensor on logits device/dtype to avoid promotion surprises
        total_loss = logits.new_zeros(())
        contributed = 0  # count how many digit positions contributed to the loss
        # Iterate digits i=0..4 (assumed MSB-first mapping to heads 0..4)
        for i in range(5):
            # Gather logits for digit i: [B, 10]
            li = logits[:, i, :]
            yi = labels[:, i]
            # Mask: include non-zero always; zeros with prob keep_prob[i]
            is_zero = (yi == 0)
            # Bernoulli draw per example for zero digits; construct probs tensor on the same device
            if self.keep_prob[i] >= 1.0:
                include_zero = torch.ones_like(is_zero, dtype=torch.bool)
            elif self.keep_prob[i] <= 0.0:
                # Explicit zero-drop: no stochastic sampling
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
            else:
                probs = torch.full(yi.shape, float(self.keep_prob[i]), device=yi.device, dtype=torch.float)
                # Sample only for zero positions to avoid unnecessary RNG work
                include_zero = torch.zeros_like(is_zero, dtype=torch.bool)
                if is_zero.any():
                    include_zero[is_zero] = torch.bernoulli(probs[is_zero]).bool()
            include_mask = (~is_zero) | (is_zero & include_zero)
            if include_mask.any():
                # Cross-entropy for included examples at this digit
                loss_i = F.cross_entropy(li[include_mask], yi[include_mask], reduction="mean")
                total_loss = total_loss + loss_i
                contributed += 1

        if contributed == 0:
            # Fallback: mean over digits of CE(logits[:, i], labels[:, i])
            if not self._warned_fallback:
                warnings.warn("AnswerLoss fallback activated: no digits included after masking", stacklevel=2)
                self._warned_fallback = True
            per_digit_losses = []
            for i in range(5):
                li = logits[:, i, :]
                yi = labels[:, i]
                per_digit_losses.append(F.cross_entropy(li, yi, reduction="mean"))
            fallback_loss = torch.stack(per_digit_losses).mean()
            return fallback_loss

        # Normalize by number of contributing digit positions to keep scale stable
        # Final loss = mean CE over contributing digit positions (not samples)
        return total_loss / float(contributed)
