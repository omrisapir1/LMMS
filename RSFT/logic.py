from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


TARGET_IGNORE = 0
TARGET_Z = 1
TARGET_ANSWER = 2
TARGET_DIGIT = 3
TARGET_VERIFY = 4


@dataclass
class RoundTrace:
    z_token_ids: List[int]
    digit_token_ids: List[int]
    pred_digits: List[int]
    true_digits: List[int]
    verify_token_id: int
    is_correct: bool


@dataclass
class MultiRoundRollout:
    prompt_idx: int
    rollout_idx: int
    rounds: List[RoundTrace]
    success: bool


def extract_z_before_answer(
    token_ids: Sequence[int],
    *,
    answer_token_id: int,
    stop_reason: object = None,
    finish_reason: object = None,
) -> Optional[List[int]]:
    ids = [int(x) for x in token_ids]
    ans = int(answer_token_id)

    if ids:
        try:
            idx = ids.index(ans)
            return ids[:idx]
        except ValueError:
            pass

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


def _validate_accepted_rounds(
    *,
    rounds: Sequence[RoundTrace],
    finalize_token_id: int,
    retry_token_id: int,
) -> None:
    if len(rounds) == 0:
        raise RuntimeError("Accepted example must include at least one round")

    success_indices = [i for i, r in enumerate(rounds) if bool(r.is_correct)]
    if len(success_indices) > 1:
        raise RuntimeError("Accepted example cannot contain more than one successful round")
    if len(success_indices) == 1 and success_indices[0] != len(rounds) - 1:
        raise RuntimeError("Successful round must be terminal")

    for idx, rnd in enumerate(rounds):
        if len(rnd.digit_token_ids) != 5:
            raise RuntimeError(f"Digits phase must emit exactly 5 tokens per round, got {len(rnd.digit_token_ids)}")
        if len(rnd.pred_digits) != 5 or len(rnd.true_digits) != 5:
            raise RuntimeError("Each round must contain exactly 5 predicted digits and 5 true digits")
        if bool(rnd.is_correct):
            if int(rnd.verify_token_id) != int(finalize_token_id):
                raise RuntimeError("Successful round verify token must be <FINALIZE>")
            if not exact_digit_match(rnd.pred_digits, rnd.true_digits):
                raise RuntimeError("Successful round must have exact digit match")
        else:
            if int(rnd.verify_token_id) != int(retry_token_id):
                raise RuntimeError("Failed round verify token must be <RETRY>")
            if exact_digit_match(rnd.pred_digits, rnd.true_digits):
                raise RuntimeError("Failed round cannot have exact digit match")
            if len(success_indices) == 1 and idx == len(rounds) - 1:
                raise RuntimeError("When a successful round exists, final round must be successful")
    if len(success_indices) == 0:
        if any(bool(r.is_correct) for r in rounds):
            raise RuntimeError("Zero-success examples must have all rounds marked failed")
        if any(int(r.verify_token_id) != int(retry_token_id) for r in rounds):
            raise RuntimeError("Zero-success examples must use <RETRY> verify token on all rounds")


def build_training_example(
    *,
    prompt_ids: Sequence[int],
    rounds: Sequence[RoundTrace],
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    max_length: int,
) -> Optional[Dict[str, object]]:
    _validate_accepted_rounds(
        rounds=rounds,
        finalize_token_id=int(finalize_token_id),
        retry_token_id=int(retry_token_id),
    )

    success_indices = [i for i, r in enumerate(rounds) if bool(r.is_correct)]
    has_success = len(success_indices) == 1
    success_idx = int(success_indices[0]) if has_success else -1

    suffix: List[int] = []
    token_class_suffix: List[int] = []
    round_z_lens: List[int] = []
    failed_rounds_before_success = 0
    per_round_supervision: List[Dict[str, int]] = []

    for idx, rnd in enumerate(rounds):
        is_supervised_final = bool(has_success and idx == success_idx)

        z_ids = [int(x) for x in rnd.z_token_ids]
        d_ids = [int(x) for x in rnd.digit_token_ids]
        if len(d_ids) != 5:
            raise RuntimeError("Digits phase must emit exactly 5 tokens per round")

        suffix.extend(z_ids)
        suffix.append(int(answer_token_id))
        suffix.extend(d_ids)
        suffix.append(int(rnd.verify_token_id))

        round_z_lens.append(len(z_ids))

        if is_supervised_final:
            token_class_suffix.extend([TARGET_Z] * len(z_ids))
            token_class_suffix.append(TARGET_ANSWER)
            token_class_suffix.extend([TARGET_DIGIT] * 5)
            token_class_suffix.append(TARGET_VERIFY)
            per_round_supervision.append(
                {
                    "z_ans": int(len(z_ids) + 1),
                    "digits": 5,
                    "verify": 1,
                }
            )
        else:
            token_class_suffix.extend([TARGET_IGNORE] * len(z_ids))
            token_class_suffix.append(TARGET_IGNORE)
            token_class_suffix.extend([TARGET_IGNORE] * 5)
            token_class_suffix.append(TARGET_VERIFY)
            failed_rounds_before_success += 1
            per_round_supervision.append(
                {
                    "z_ans": 0,
                    "digits": 0,
                    "verify": 1,
                }
            )

    input_ids = [int(x) for x in prompt_ids] + suffix
    if len(input_ids) > int(max_length):
        return None

    attention_mask = [1] * len(input_ids)

    token_class = [TARGET_IGNORE] * len(prompt_ids)
    token_class.extend(token_class_suffix)

    target_class = [TARGET_IGNORE] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        target_class[pos] = token_class[pos + 1]

    labels = [-100] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        tcls = target_class[pos]
        if tcls in (TARGET_Z, TARGET_ANSWER, TARGET_DIGIT, TARGET_VERIFY):
            labels[pos] = int(input_ids[pos + 1])

    # Safety contract checks for accepted examples.
    expected_z_ans = len(rounds[-1].z_token_ids) + 1 if has_success else 0
    supervised_z_ans = sum(1 for tc in token_class if tc in (TARGET_Z, TARGET_ANSWER))
    supervised_digits = sum(1 for tc in token_class if tc == TARGET_DIGIT)
    supervised_verify = sum(1 for tc in token_class if tc == TARGET_VERIFY)
    if supervised_verify != len(rounds):
        raise RuntimeError("Each round must contribute exactly one supervised verify token")
    if has_success:
        if supervised_z_ans != expected_z_ans or supervised_digits != 5:
            raise RuntimeError("Final successful round must provide full supervision for Z/<ANSWER>/digits")
    else:
        if supervised_z_ans != 0 or supervised_digits != 0:
            raise RuntimeError("Zero-success sequence must disable Z/<ANSWER>/digit supervision")
    # Corrected masking rule:
    # 1) verify supervision is always active for every round
    # 2) z/answer/digit supervision is active only on the final correct round
    for idx, stats in enumerate(per_round_supervision):
        if int(stats["verify"]) != 1:
            raise RuntimeError("Verification-token loss must be active for every round")
        if not (has_success and idx == success_idx):
            if int(stats["z_ans"]) != 0 or int(stats["digits"]) != 0:
                raise RuntimeError("Non-final rounds must not have Z/<ANSWER>/digit supervision")
        else:
            if int(stats["z_ans"]) != expected_z_ans or int(stats["digits"]) != 5:
                raise RuntimeError("Final correct round must have full Z/<ANSWER>/digit supervision")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_class": target_class,
        "round_count": int(len(rounds)),
        "failed_rounds": int(failed_rounds_before_success),
        "round_z_lens": [int(x) for x in round_z_lens],
    }


def collate_training_examples(examples: Sequence[Dict[str, object]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate empty examples")

    max_len = max(len(ex["input_ids"]) for ex in examples)  # type: ignore[index]

    def _pad(vals: List[int], pad_value: int) -> List[int]:
        if len(vals) >= max_len:
            return vals
        return vals + [pad_value] * (max_len - len(vals))

    input_ids = torch.tensor(
        [_pad(list(ex["input_ids"]), int(pad_token_id)) for ex in examples],  # type: ignore[index]
        dtype=torch.long,
    )
    attention_mask = torch.tensor([_pad(list(ex["attention_mask"]), 0) for ex in examples], dtype=torch.long)  # type: ignore[index]
    labels = torch.tensor([_pad(list(ex["labels"]), -100) for ex in examples], dtype=torch.long)  # type: ignore[index]
    target_class = torch.tensor(
        [_pad(list(ex["target_class"]), TARGET_IGNORE) for ex in examples],  # type: ignore[index]
        dtype=torch.long,
    )

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
    verify_token_ids: Sequence[int],
    w_z_ans: float,
    w_digits: float,
    w_verify: float,
) -> Dict[str, torch.Tensor]:
    z_ans_mask = (target_class == TARGET_Z) | (target_class == TARGET_ANSWER)
    digits_mask = target_class == TARGET_DIGIT
    verify_mask = target_class == TARGET_VERIFY

    z_ans_allowed = [int(x) for x in z_token_ids] + [int(answer_token_id)]
    digits_allowed = [int(x) for x in digit_token_ids]
    verify_allowed = [int(x) for x in verify_token_ids]

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
    l_verify = _restricted_masked_ce(
        logits=logits,
        labels=labels,
        mask=verify_mask,
        allowed_token_ids=verify_allowed,
    )

    total = float(w_z_ans) * l_z_ans + float(w_digits) * l_digits + float(w_verify) * l_verify

    return {
        "l_z_ans": l_z_ans,
        "l_digits": l_digits,
        "l_verify": l_verify,
        "loss": total,
    }


def mean_or_zero(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    if not rows:
        return 0.0
    return float(sum(rows) / len(rows))
