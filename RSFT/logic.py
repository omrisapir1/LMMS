from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


TARGET_IGNORE = 0
TARGET_Z = 1
TARGET_ANSWER = 2
TARGET_DIGIT = 3


@dataclass
class RolloutCandidate:
    prompt_idx: int
    rollout_idx: int
    z_token_ids: List[int]
    digit_token_ids: List[int]
    pred_digits: List[int]
    true_digits: List[int]


@dataclass
class AcceptedExample:
    prompt_idx: int
    rollout_idx: int
    z_token_ids: List[int]
    digit_token_ids: List[int]
    pred_digits: List[int]


def extract_z_before_answer(
    token_ids: Sequence[int],
    *,
    answer_token_id: int,
    stop_reason: object = None,
    finish_reason: object = None,
) -> Optional[List[int]]:
    ids = [int(x) for x in token_ids]
    ans = int(answer_token_id)

    # Case 1: answer token is present in returned token_ids.
    if ids:
        try:
            idx = ids.index(ans)
            return ids[:idx]
        except ValueError:
            pass

    # Case 2: backend indicates stop-on-answer, but omits the answer token from token_ids.
    stop_is_answer = False
    if stop_reason is not None:
        if isinstance(stop_reason, (int, float)) and int(stop_reason) == ans:
            stop_is_answer = True
        else:
            sr = str(stop_reason).strip().lower()
            if sr in {"stop", "eos_token", "eos"}:
                stop_is_answer = True
    if not stop_is_answer and finish_reason is not None:
        fr = str(finish_reason).strip().lower()
        if fr == "stop":
            stop_is_answer = True

    if stop_is_answer:
        return ids

    return None


def extract_z_before_answer_from_row(row: Dict[str, object], *, answer_token_id: int) -> Optional[List[int]]:
    return extract_z_before_answer(
        row.get("token_ids", []),
        answer_token_id=answer_token_id,
        stop_reason=row.get("stop_reason", None),
        finish_reason=row.get("finish_reason", None),
    )


def decode_digit_tokens(digit_token_ids: Sequence[int], digit_id_to_val: Dict[int, int]) -> Optional[List[int]]:
    if len(digit_token_ids) != 5:
        return None
    out: List[int] = []
    for tok in digit_token_ids:
        key = int(tok)
        if key not in digit_id_to_val:
            return None
        out.append(int(digit_id_to_val[key]))
    return out


def exact_digit_match(pred_digits: Sequence[int], true_digits: Sequence[int]) -> bool:
    if len(pred_digits) != 5 or len(true_digits) != 5:
        return False
    return all(int(a) == int(b) for a, b in zip(pred_digits, true_digits))


def select_shortest_valid(candidates: Sequence[RolloutCandidate]) -> Optional[AcceptedExample]:
    valid = [cand for cand in candidates if exact_digit_match(cand.pred_digits, cand.true_digits)]
    if not valid:
        return None
    valid_sorted = sorted(
        valid,
        key=lambda c: (len(c.z_token_ids), int(c.rollout_idx)),
    )
    mid = len(valid_sorted) // 2  # upper-middle for even counts
    best = valid_sorted[mid]
    return AcceptedExample(
        prompt_idx=int(best.prompt_idx),
        rollout_idx=int(best.rollout_idx),
        z_token_ids=list(best.z_token_ids),
        digit_token_ids=list(best.digit_token_ids),
        pred_digits=list(best.pred_digits),
    )


def build_training_example(
    *,
    prompt_ids: Sequence[int],
    z_token_ids: Sequence[int],
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    z_ids = [int(x) for x in z_token_ids]
    d_ids = [int(x) for x in digit_token_ids]
    if len(d_ids) != 5:
        return None

    suffix = z_ids + [int(answer_token_id)] + d_ids
    input_ids = [int(x) for x in prompt_ids] + suffix
    if len(input_ids) > int(max_length):
        return None

    attention_mask = [1] * len(input_ids)

    token_class = [TARGET_IGNORE] * len(prompt_ids)
    token_class += [TARGET_Z] * len(z_ids)
    token_class += [TARGET_ANSWER]
    token_class += [TARGET_DIGIT] * 5

    target_class = [TARGET_IGNORE] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        target_class[pos] = token_class[pos + 1]

    labels = [-100] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        tcls = target_class[pos]
        if tcls in (TARGET_Z, TARGET_ANSWER, TARGET_DIGIT):
            labels[pos] = int(input_ids[pos + 1])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_class": target_class,
        "z_len": len(z_ids),
    }


def collate_training_examples(examples: Sequence[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate empty examples")

    max_len = max(len(ex["input_ids"]) for ex in examples)

    def _pad(vals: List[int], pad_value: int) -> List[int]:
        if len(vals) >= max_len:
            return vals
        return vals + [pad_value] * (max_len - len(vals))

    input_ids = torch.tensor([_pad(list(ex["input_ids"]), int(pad_token_id)) for ex in examples], dtype=torch.long)
    attention_mask = torch.tensor([_pad(list(ex["attention_mask"]), 0) for ex in examples], dtype=torch.long)
    labels = torch.tensor([_pad(list(ex["labels"]), -100) for ex in examples], dtype=torch.long)
    target_class = torch.tensor([_pad(list(ex["target_class"]), TARGET_IGNORE) for ex in examples], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_class": target_class,
    }


def _restricted_masked_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    allowed_token_ids: Sequence[int],
) -> torch.Tensor:
    active = mask & (labels != -100)
    if not bool(active.any()):
        return logits.new_zeros(())

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_active = active.reshape(-1)
    sel_logits = flat_logits[flat_active]
    sel_labels = flat_labels[flat_active]

    allowed_t = torch.tensor([int(x) for x in allowed_token_ids], dtype=torch.long, device=logits.device)
    restricted_logits = sel_logits.index_select(1, allowed_t)

    vocab_size = int(logits.shape[-1])
    id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=logits.device)
    id_to_local[allowed_t] = torch.arange(allowed_t.numel(), dtype=torch.long, device=logits.device)
    local_labels = id_to_local[sel_labels]
    if bool((local_labels < 0).any()):
        raise RuntimeError("Restricted CE received labels outside allowed token set")

    return F.cross_entropy(restricted_logits, local_labels, reduction="mean")


def compute_rsft_losses(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_class: torch.Tensor,
    z_token_ids: Sequence[int],
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    w_z_ans: float,
    w_digits: float,
) -> Dict[str, torch.Tensor]:
    z_ans_mask = (target_class == TARGET_Z) | (target_class == TARGET_ANSWER)
    digits_mask = target_class == TARGET_DIGIT
    z_ans_allowed = [int(x) for x in z_token_ids] + [int(answer_token_id)]
    digits_allowed = [int(x) for x in digit_token_ids]

    l_z_ans = _restricted_masked_ce(
        logits=logits,
        labels=labels,
        mask=z_ans_mask,
        allowed_token_ids=z_ans_allowed,
    )
    l_digits = _restricted_masked_ce(
        logits=logits,
        labels=labels,
        mask=digits_mask,
        allowed_token_ids=digits_allowed,
    )
    total = float(w_z_ans) * l_z_ans + float(w_digits) * l_digits

    return {
        "l_z_ans": l_z_ans,
        "l_digits": l_digits,
        "loss": total,
    }


def mean_or_zero(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    if not rows:
        return 0.0
    return float(sum(rows) / len(rows))
