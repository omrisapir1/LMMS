from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .dataset import TARGET_ANSWER, TARGET_DIGIT, TARGET_Z


@dataclass
class LossOutput:
    total: torch.Tensor
    l_z: torch.Tensor
    l_answer: torch.Tensor
    l_digits: torch.Tensor
    z_acc: float
    digit_exact_match: float
    clip_drop_count: float


@dataclass
class CounterfactualLossOutput:
    loss: torch.Tensor
    mean_sym_kl: float
    mean_entropy: float


def _debug_check_label_membership(
    labels: torch.Tensor,
    target_class: torch.Tensor,
    z_allowed_ids: Sequence[int],
    digit_allowed_ids: Sequence[int],
) -> None:
    z_allowed = set(int(x) for x in z_allowed_ids)
    d_allowed = set(int(x) for x in digit_allowed_ids)

    with torch.no_grad():
        z_mask = (target_class == TARGET_Z) & (labels >= 0)
        a_mask = (target_class == TARGET_ANSWER) & (labels >= 0)
        d_mask = (target_class == TARGET_DIGIT) & (labels >= 0)

        def first_bad(mask: torch.Tensor, allowed_set: set[int], name: str) -> bool:
            if not bool(mask.any()):
                return False
            idx = mask.nonzero(as_tuple=False)
            for k in range(min(50, idx.shape[0])):
                b, t = int(idx[k, 0]), int(idx[k, 1])
                y = int(labels[b, t])
                if y not in allowed_set:
                    print(f"[BAD {name}] b={b} t={t} label={y}")
                    return True
            return False

        if first_bad(z_mask | a_mask, z_allowed, "Z/ANSWER"):
            raise RuntimeError("Label not in z_allowed_ids")
        if first_bad(d_mask, d_allowed, "DIGIT"):
            raise RuntimeError("Label not in digit_allowed_ids")


def _build_inv_map(*, vocab_size: int, allowed_ids: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    allowed_t = torch.as_tensor(list(allowed_ids), dtype=torch.long, device=device)
    inv = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
    inv[allowed_t] = torch.arange(int(allowed_t.numel()), dtype=torch.long, device=device)
    return allowed_t, inv


def _restricted_ce_for_positions(
    *,
    logits: torch.Tensor,  # [B,L,V]
    labels: torch.Tensor,  # [B,L]
    idx: torch.Tensor,  # [N,2]
    allowed_t: torch.Tensor,  # [K]
    inv: torch.Tensor,  # [V] token-id -> [0..K-1] or -1
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(idx.shape[0]) == 0:
        zero = logits.new_zeros((), dtype=logits.dtype)
        empty = torch.empty((0,), dtype=torch.long, device=logits.device)
        return zero, empty, empty

    x_full = logits[idx[:, 0], idx[:, 1], :]  # [N,V]
    x = x_full[:, allowed_t]  # [N,K]
    y = labels[idx[:, 0], idx[:, 1]].long()  # [N]
    y_small = inv[y]
    if bool((y_small < 0).any()):
        bad = (y_small < 0).nonzero(as_tuple=False).view(-1)
        j = int(bad[0].item())
        b = int(idx[j, 0].item())
        t = int(idx[j, 1].item())
        yj = int(y[j].item())
        raise RuntimeError(
            f"Restricted CE label mapping failed at b={b} t={t}: label_id={yj} "
            f"not present in allowed ids of size {int(allowed_t.numel())}"
        )
    loss = F.cross_entropy(x, y_small, reduction="mean", label_smoothing=float(label_smoothing))
    preds_small = x.argmax(dim=-1)
    return loss, preds_small, y_small


def compute_weighted_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_class: torch.Tensor,
    z_allowed_ids: Sequence[int],
    digit_allowed_ids: Sequence[int],
    alpha_z: float,
    alpha_answer: float,
    alpha_digits: float,
    z_label_smoothing: float,
    keep_prob: Sequence[float],
) -> LossOutput:
    _debug_check_label_membership(
        labels=labels,
        target_class=target_class,
        z_allowed_ids=z_allowed_ids,
        digit_allowed_ids=digit_allowed_ids,
    )

    z_mask = (target_class == TARGET_Z) & (labels >= 0)
    answer_mask = (target_class == TARGET_ANSWER) & (labels >= 0)
    digit_mask = (target_class == TARGET_DIGIT) & (labels >= 0)
    vocab_size = int(logits.shape[-1])
    z_allowed_t, z_inv = _build_inv_map(vocab_size=vocab_size, allowed_ids=z_allowed_ids, device=logits.device)
    digit_allowed_t, digit_inv = _build_inv_map(
        vocab_size=vocab_size, allowed_ids=digit_allowed_ids, device=logits.device
    )

    z_idx = z_mask.nonzero(as_tuple=False)
    answer_idx = answer_mask.nonzero(as_tuple=False)
    digit_idx = digit_mask.nonzero(as_tuple=False)

    l_z, z_preds_small, z_y_small = _restricted_ce_for_positions(
        logits=logits,
        labels=labels,
        idx=z_idx,
        allowed_t=z_allowed_t,
        inv=z_inv,
        label_smoothing=float(z_label_smoothing),
    )
    l_answer, _, _ = _restricted_ce_for_positions(
        logits=logits,
        labels=labels,
        idx=answer_idx,
        allowed_t=z_allowed_t,
        inv=z_inv,
        label_smoothing=0.0,
    )
    if len(keep_prob) != 5:
        raise ValueError(f"keep_prob must have length 5, got {len(keep_prob)}")
    if len(digit_allowed_ids) != 10:
        raise ValueError("digit_allowed_ids must contain exactly 10 ids (digits 0-9)")
    zero_token_id = int(digit_allowed_ids[0])

    clip_drop_count = 0.0
    if int(digit_idx.shape[0]) == 0:
        l_digits = logits.new_zeros((), dtype=logits.dtype)
    else:
        x_full = logits[digit_idx[:, 0], digit_idx[:, 1], :]  # [N,V]
        x_digits = x_full[:, digit_allowed_t]  # [N,10]
        y_digits = labels[digit_idx[:, 0], digit_idx[:, 1]].long()  # [N]
        y_digits_small = digit_inv[y_digits]  # [N]
        if bool((y_digits_small < 0).any()):
            bad = (y_digits_small < 0).nonzero(as_tuple=False).view(-1)
            j = int(bad[0].item())
            b = int(digit_idx[j, 0].item())
            t = int(digit_idx[j, 1].item())
            yj = int(y_digits[j].item())
            raise RuntimeError(
                f"Digit CE label mapping failed at b={b} t={t}: label_id={yj} "
                f"not present in digit allowed ids of size {int(digit_allowed_t.numel())}"
            )

        ce_per_pos = F.cross_entropy(x_digits, y_digits_small, reduction="none")  # [N]
        keep_mask = torch.zeros_like(ce_per_pos, dtype=torch.bool)  # [N]
        keep_prob_t = torch.as_tensor(list(keep_prob), dtype=torch.float32, device=logits.device)

        # Build stochastic keep-mask per sample and per digit position.
        for b in range(int(labels.shape[0])):
            row_mask = digit_mask[b]
            n = int(row_mask.sum().item())
            if n <= 0:
                continue
            row_idx = (digit_idx[:, 0] == b).nonzero(as_tuple=False).view(-1)
            if int(row_idx.numel()) != n:
                raise RuntimeError(f"digit indexing mismatch for batch row {b}: expected {n}, got {int(row_idx.numel())}")
            if n > int(keep_prob_t.numel()):
                raise RuntimeError(f"digit count exceeds keep_prob length for row {b}: {n} > {int(keep_prob_t.numel())}")

            row_y = y_digits[row_idx]
            non_zero = row_y != zero_token_id
            row_rand = torch.rand((n,), dtype=torch.float32, device=logits.device)
            zero_keep = row_rand < keep_prob_t[:n]
            row_keep = non_zero | zero_keep
            keep_mask[row_idx] = row_keep

            # Debug assertions for mask behavior.
            if bool(non_zero.all()):
                assert bool(row_keep.all()), "all non-zero digits must always be kept"
            if bool((keep_prob_t[:n] == 1.0).any()):
                assert bool(row_keep[(row_y == zero_token_id) & (keep_prob_t[:n] == 1.0)].all()), (
                    "zero digit with keep_prob=1.0 must always be kept"
                )
            if bool((keep_prob_t[:n] == 0.0).any()):
                assert not bool(row_keep[(row_y == zero_token_id) & (keep_prob_t[:n] == 0.0)].any()), (
                    "zero digit with keep_prob=0.0 must always be dropped"
                )

        kept = keep_mask.to(dtype=ce_per_pos.dtype)
        clip_drop_count = float((~keep_mask).float().sum().item())
        kept_count = kept.sum()
        if float(kept_count.item()) <= 0.0:
            l_digits = logits.new_zeros((), dtype=logits.dtype)
        else:
            l_digits = (ce_per_pos * kept).sum() / kept_count

    total = float(alpha_z) * l_z + float(alpha_answer) * l_answer + float(alpha_digits) * l_digits

    with torch.no_grad():
        z_acc = 0.0
        if int(z_idx.shape[0]) > 0:
            z_acc = float((z_preds_small == z_y_small).float().mean().item())

        digit_exact_match = 0.0
        if digit_mask.any():
            # Requires exactly 5 digit targets per valid row to count as exact-match.
            bsz = labels.shape[0]
            correct = 0
            total_rows = 0
            for i in range(bsz):
                row_mask = digit_mask[i]
                n = int(row_mask.sum().item())
                if n != 5:
                    continue
                pos = row_mask.nonzero(as_tuple=False).view(-1)
                x_full = logits[i, pos, :]  # [5,V]
                x = x_full[:, digit_allowed_t]  # [5,10]
                pred_small = x.argmax(dim=-1)
                y = labels[i, pos].long()
                y_small = digit_inv[y]
                if bool((y_small < 0).any()):
                    raise RuntimeError(f"Digit label mapping failed for row={i}")
                total_rows += 1
                if bool(torch.equal(pred_small, y_small)):
                    correct += 1
            if total_rows > 0:
                digit_exact_match = float(correct / total_rows)

    return LossOutput(
        total=total,
        l_z=l_z,
        l_answer=l_answer,
        l_digits=l_digits,
        z_acc=z_acc,
        digit_exact_match=digit_exact_match,
        clip_drop_count=clip_drop_count,
    )


def extract_digit_logits(
    *,
    logits: torch.Tensor,
    target_class: torch.Tensor,
    digit_allowed_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(logits.shape[0])
    digit_id_tensor = torch.as_tensor(list(digit_allowed_ids), dtype=torch.long, device=logits.device)
    max_digits = int((target_class == TARGET_DIGIT).sum(dim=1).max().item()) if bsz > 0 else 0
    if max_digits <= 0:
        shape = (bsz, 0, int(digit_id_tensor.shape[0]))
        return logits.new_zeros(shape), torch.zeros((bsz, 0), dtype=torch.bool, device=logits.device)

    out = logits.new_zeros((bsz, max_digits, int(digit_id_tensor.shape[0])))
    valid = torch.zeros((bsz, max_digits), dtype=torch.bool, device=logits.device)
    for b in range(bsz):
        pos = (target_class[b] == TARGET_DIGIT).nonzero(as_tuple=False).view(-1)
        if int(pos.numel()) == 0:
            continue
        row = logits[b, pos, :][:, digit_id_tensor]
        n = int(row.shape[0])
        out[b, :n, :] = row
        valid[b, :n] = True
    return out, valid


def compute_counterfactual_regularizer(
    *,
    clean_digit_logits: torch.Tensor,
    cf_digit_logits: torch.Tensor,
    digit_valid_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    variant_name: str,
    kl_margin: float,
    eps: float,
) -> CounterfactualLossOutput:
    if clean_digit_logits.shape != cf_digit_logits.shape:
        raise ValueError(
            f"shape mismatch: clean={tuple(clean_digit_logits.shape)} cf={tuple(cf_digit_logits.shape)}"
        )
    if clean_digit_logits.ndim != 3:
        raise ValueError(f"expected [B,T,10] logits, got ndim={clean_digit_logits.ndim}")

    bsz = int(clean_digit_logits.shape[0])
    if bsz == 0:
        zero = clean_digit_logits.new_zeros(())
        return CounterfactualLossOutput(loss=zero, mean_sym_kl=0.0, mean_entropy=0.0)

    sample_mask = eligible_mask & digit_valid_mask.any(dim=1)
    if not bool(sample_mask.any()):
        zero = clean_digit_logits.new_zeros(())
        return CounterfactualLossOutput(loss=zero, mean_sym_kl=0.0, mean_entropy=0.0)

    p = F.softmax(clean_digit_logits.detach(), dim=-1).clamp(min=float(eps), max=1.0)
    q = F.softmax(cf_digit_logits, dim=-1).clamp(min=float(eps), max=1.0)
    log_p = torch.log(p)
    log_q = torch.log(q)

    kl_pq = (p * (log_p - log_q)).sum(dim=-1)  # [B,T]
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)  # [B,T]
    sym_kl = kl_pq + kl_qp  # [B,T]
    entropy = -(q * log_q).sum(dim=-1)  # [B,T]

    pos_mask = digit_valid_mask.to(dtype=clean_digit_logits.dtype)
    denom = pos_mask.sum(dim=1).clamp_min(1.0)
    sym_kl_per_sample = (sym_kl * pos_mask).sum(dim=1) / denom
    entropy_per_sample = (entropy * pos_mask).sum(dim=1) / denom

    sample_w = sample_mask.to(dtype=clean_digit_logits.dtype)
    sample_w_sum = sample_w.sum().clamp_min(1.0)

    variant = str(variant_name).lower().strip()
    if variant in ("truncate", "reverse"):
        per_sample_loss = torch.relu(float(kl_margin) - sym_kl_per_sample)
    elif variant == "random":
        per_sample_loss = -entropy_per_sample
    else:
        raise ValueError(f"unknown counterfactual variant: {variant_name}")

    loss = (per_sample_loss * sample_w).sum() / sample_w_sum
    mean_sym_kl = float(((sym_kl_per_sample * sample_w).sum() / sample_w_sum).detach().item())
    mean_entropy = float(((entropy_per_sample * sample_w).sum() / sample_w_sum).detach().item())

    return CounterfactualLossOutput(
        loss=loss,
        mean_sym_kl=mean_sym_kl,
        mean_entropy=mean_entropy,
    )
