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
