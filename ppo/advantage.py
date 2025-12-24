from typing import Dict

import torch


def compute_advantages(
    batch: Dict[str, torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute simple Phase-1 advantages:
        A_t = r - V_t

    Args:
        batch: dict containing at least keys "rewards" and "value_old", both [N] float tensors
        normalize: whether to normalize advantages to zero-mean, unit-std
        eps: small constant for numerical stability

    Returns:
        Tensor [N] (float32) on the same device as batch["rewards"].

    Raises:
        KeyError, TypeError, RuntimeError for shape/type/numerical violations.
    """
    if "rewards" not in batch or "value_old" not in batch:
        missing = [k for k in ("rewards", "value_old") if k not in batch]
        raise KeyError(f"Batch missing required keys: {missing}")

    rewards = batch["rewards"]
    values = batch["value_old"]

    if not isinstance(rewards, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TypeError("Batch 'rewards' and 'value_old' must be torch.Tensor")

    if rewards.ndim != 1 or values.ndim != 1:
        raise RuntimeError(f"Expected 1D tensors. Got rewards.ndim={rewards.ndim}, values.ndim={values.ndim}")

    if rewards.shape != values.shape:
        raise RuntimeError(f"Shape mismatch: rewards{rewards.shape} vs value_old{values.shape}")

    device = rewards.device
    # Compute raw advantages on the rewards device, cast to float32
    adv = (rewards.to(device=device, dtype=torch.float32) - values.to(device=device, dtype=torch.float32))

    # Numerical sanity for raw advantages
    if not torch.isfinite(adv).all():
        raise RuntimeError("Non-finite values encountered in raw advantages (r - V).")

    if normalize:
        mean = adv.mean()
        std = adv.std()
        if not torch.isfinite(mean) or not torch.isfinite(std):
            raise RuntimeError("Non-finite mean/std encountered during advantage normalization.")
        if std <= 0:
            raise RuntimeError(f"Standard deviation is non-positive; cannot normalize advantages. std={std.item()}")
        adv = (adv - mean) / (std + eps)

        # Required checks after normalization
        if adv.shape != rewards.shape:
            raise RuntimeError("Advantages shape mismatch after normalization.")
        if not torch.isfinite(adv).all():
            raise RuntimeError("Non-finite values encountered in normalized advantages.")
        if abs(float(adv.mean().item())) >= 1e-3:

            raise RuntimeError(f"Normalized advantages mean is not ~0. anomalous mean={adv.mean().item()}")

    return adv
