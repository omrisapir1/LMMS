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
    executed_verify_token_id: int
    verify_target_token_id: int
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

    success_indices = [
        i
        for i, r in enumerate(rounds)
        if bool(r.is_correct) and int(r.executed_verify_token_id) == int(finalize_token_id)
    ]
    if len(success_indices) > 1:
        raise RuntimeError("Accepted example cannot contain more than one successful round")
    if len(success_indices) == 1 and success_indices[0] != len(rounds) - 1:
        raise RuntimeError("Successful round must be terminal")

    for idx, rnd in enumerate(rounds):
        if len(rnd.digit_token_ids) != 5:
            raise RuntimeError(f"Digits phase must emit exactly 5 tokens per round, got {len(rnd.digit_token_ids)}")
        if len(rnd.pred_digits) != 5 or len(rnd.true_digits) != 5:
            raise RuntimeError("Each round must contain exactly 5 predicted digits and 5 true digits")
        if int(rnd.executed_verify_token_id) not in (int(finalize_token_id), int(retry_token_id)):
            raise RuntimeError("Executed verify token must be exactly one of <FINALIZE>/<RETRY>")
        if int(rnd.verify_target_token_id) not in (int(finalize_token_id), int(retry_token_id)):
            raise RuntimeError("Verify target token must be exactly one of <FINALIZE>/<RETRY>")
        if bool(rnd.is_correct):
            if not exact_digit_match(rnd.pred_digits, rnd.true_digits):
                raise RuntimeError("Successful round must have exact digit match")
            if int(rnd.verify_target_token_id) != int(finalize_token_id):
                raise RuntimeError("Correct round verify target must be <FINALIZE>")
            if int(rnd.executed_verify_token_id) == int(retry_token_id):
                if int(rnd.verify_target_token_id) != int(finalize_token_id):
                    raise RuntimeError("Correct round with executed <RETRY> must still target <FINALIZE>")
        else:
            if exact_digit_match(rnd.pred_digits, rnd.true_digits):
                raise RuntimeError("Failed round cannot have exact digit match")
            if int(rnd.executed_verify_token_id) != int(retry_token_id):
                raise RuntimeError("Wrong round executed verify token must be <RETRY>")
            if int(rnd.verify_target_token_id) != int(retry_token_id):
                raise RuntimeError("Wrong round verify target must be <RETRY>")
            if len(success_indices) == 1 and idx == len(rounds) - 1:
                raise RuntimeError("When a successful round exists, final round must be successful")
    # Zero-success trajectories are allowed; they may include correct-but-retry rounds.


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

    success_indices = [
        i
        for i, r in enumerate(rounds)
        if bool(r.is_correct) and int(r.executed_verify_token_id) == int(finalize_token_id)
    ]
    has_success = len(success_indices) == 1
    success_idx = int(success_indices[0]) if has_success else -1

    suffix: List[int] = []
    token_class_suffix: List[int] = []
    verify_target_suffix: List[int] = []
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
        suffix.append(int(rnd.executed_verify_token_id))
        verify_target_suffix.extend(
            [TARGET_IGNORE] * len(z_ids) + [TARGET_IGNORE] + [TARGET_IGNORE] * 5 + [int(rnd.verify_target_token_id)]
        )

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
    verify_target_by_token_pos = [TARGET_IGNORE] * len(prompt_ids)
    verify_target_by_token_pos.extend(verify_target_suffix)

    target_class = [TARGET_IGNORE] * len(input_ids)
    verify_target_by_label_pos = [TARGET_IGNORE] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        target_class[pos] = token_class[pos + 1]
        verify_target_by_label_pos[pos] = int(verify_target_by_token_pos[pos + 1])

    labels = [-100] * len(input_ids)
    for pos in range(len(input_ids) - 1):
        tcls = target_class[pos]
        if tcls in (TARGET_Z, TARGET_ANSWER, TARGET_DIGIT):
            labels[pos] = int(input_ids[pos + 1])
        elif tcls == TARGET_VERIFY:
            tgt = int(verify_target_by_label_pos[pos])
            if tgt not in (int(finalize_token_id), int(retry_token_id)):
                raise RuntimeError("Verify position missing valid verify target label")
            labels[pos] = int(tgt)

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
    # If executed verify is <FINALIZE> the sequence must terminate at that round.
    for idx, rnd in enumerate(rounds):
        if int(rnd.executed_verify_token_id) == int(finalize_token_id) and idx != (len(rounds) - 1):
            raise RuntimeError("Sequence must terminate immediately after executed <FINALIZE>")

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


def _full_vocab_masked_ce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask & (labels != -100)
    if not bool(active.any()):
        return logits.new_zeros(())

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_active = active.reshape(-1)
    sel_logits = flat_logits[flat_active]
    sel_labels = flat_labels[flat_active]
    return F.cross_entropy(sel_logits, sel_labels, reduction="mean")


def compute_rsft_losses(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_class: torch.Tensor,
    z_token_ids: Sequence[int],
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    verify_token_ids: Sequence[int],
    w_z: float,
    w_answer: float,
    w_digits: float,
    w_verify: float,
) -> Dict[str, torch.Tensor]:
    # Kept for API/config compatibility; not used by full-vocab Z/<ANSWER> losses.
    _ = z_token_ids
    _ = answer_token_id

    z_mask = target_class == TARGET_Z
    answer_mask = target_class == TARGET_ANSWER
    digits_mask = target_class == TARGET_DIGIT
    verify_mask = target_class == TARGET_VERIFY

    digits_allowed = [int(x) for x in digit_token_ids]
    verify_allowed = [int(x) for x in verify_token_ids]

    # Z and <ANSWER> are trained with full-vocab CE on disjoint masks.
    l_z = _full_vocab_masked_ce(
        logits=logits,
        labels=labels,
        mask=z_mask,
    )
    l_answer = _full_vocab_masked_ce(
        logits=logits,
        labels=labels,
        mask=answer_mask,
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

    total = (
        float(w_z) * l_z
        + float(w_answer) * l_answer
        + float(w_digits) * l_digits
        + float(w_verify) * l_verify
    )

    return {
        "l_z": l_z,
        "l_answer": l_answer,
        "l_digits": l_digits,
        "l_verify": l_verify,
        "loss": total,
    }


def mean_or_zero(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    if not rows:
        return 0.0
    return float(sum(rows) / len(rows))
