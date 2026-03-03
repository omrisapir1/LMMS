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


def _masked_clone(logits: torch.Tensor, allowed_ids: Sequence[int]) -> torch.Tensor:
    # logits: [N, V]
    out = torch.full_like(logits, -1e4)
    allowed = torch.as_tensor(allowed_ids, device=logits.device, dtype=torch.long)
    out[:, allowed] = logits[:, allowed]
    return out


def apply_restricted_mask_inplace(
    *,
    logits: torch.Tensor,
    target_class: torch.Tensor,
    z_allowed_ids: Sequence[int],
    digit_allowed_ids: Sequence[int],
) -> torch.Tensor:
    # logits/target_class shapes: [B, L, V] and [B, L]
    z_or_answer = (target_class == TARGET_Z) | (target_class == TARGET_ANSWER)
    digit = target_class == TARGET_DIGIT

    if z_or_answer.any():
        idx = z_or_answer.nonzero(as_tuple=False)
        rows = logits[idx[:, 0], idx[:, 1], :]
        masked = _masked_clone(rows, z_allowed_ids)
        logits[idx[:, 0], idx[:, 1], :] = masked

    if digit.any():
        idx = digit.nonzero(as_tuple=False)
        rows = logits[idx[:, 0], idx[:, 1], :]
        masked = _masked_clone(rows, digit_allowed_ids)
        logits[idx[:, 0], idx[:, 1], :] = masked

    return logits


def _mean_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if not mask.any():
        return logits.new_zeros((), dtype=logits.dtype)
    idx = mask.nonzero(as_tuple=False)
    x = logits[idx[:, 0], idx[:, 1], :]
    y = labels[idx[:, 0], idx[:, 1]]
    with torch.no_grad():
        bad = torch.isneginf(x.gather(1, y.view(-1, 1))).view(-1)
        if bad.any():
            j = int(bad.nonzero(as_tuple=False)[0].item())
            b = int(idx[j, 0].item())
            t = int(idx[j, 1].item())
            yj = int(y[j].item())
            print("CE=inf debug:")
            print(f"  batch_idx={b} time_idx={t} label_id={yj}")
            # show top-10 logits ids (after masking)
            topv, topi = torch.topk(x[j], k=10)
            print("  top_ids:", [int(i) for i in topi.tolist()])
            print("  top_vals:", [float(v) for v in topv.tolist()])
            raise RuntimeError("Found masked-out true label -> CE would be inf")
    return F.cross_entropy(x, y, reduction="mean", label_smoothing=label_smoothing)


def compute_weighted_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_class: torch.Tensor,
    z_allowed_ids: Sequence[int],
    digit_allowed_ids: Sequence[int],
    w_z: float,
    w_answer: float,
    w_digits: float,
    z_label_smoothing: float,
) -> LossOutput:
    _debug_check_label_membership(
        labels=labels,
        target_class=target_class,
        z_allowed_ids=z_allowed_ids,
        digit_allowed_ids=digit_allowed_ids,
    )

    masked_logits = logits.clone()
    masked_logits = apply_restricted_mask_inplace(
        logits=masked_logits,
        target_class=target_class,
        z_allowed_ids=z_allowed_ids,
        digit_allowed_ids=digit_allowed_ids,
    )

    z_mask = (target_class == TARGET_Z) & (labels >= 0)
    answer_mask = (target_class == TARGET_ANSWER) & (labels >= 0)
    digit_mask = (target_class == TARGET_DIGIT) & (labels >= 0)

    l_z = _mean_ce(
        logits=masked_logits,
        labels=labels,
        mask=z_mask,
        label_smoothing=float(z_label_smoothing),
    )
    l_answer = _mean_ce(
        logits=masked_logits,
        labels=labels,
        mask=answer_mask,
        label_smoothing=0.0,
    )
    l_digits = _mean_ce(
        logits=masked_logits,
        labels=labels,
        mask=digit_mask,
        label_smoothing=0.0,
    )

    total = float(w_z) * l_z + float(w_answer) * l_answer + float(w_digits) * l_digits

    with torch.no_grad():
        preds = masked_logits.argmax(dim=-1)
        z_acc = 0.0
        if z_mask.any():
            z_acc = float((preds[z_mask] == labels[z_mask]).float().mean().item())

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
                total_rows += 1
                if bool(torch.equal(preds[i][row_mask], labels[i][row_mask])):
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
