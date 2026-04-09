import json
import os
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from RSFT.config import Config
from RSFT.dataset import PromptExample
from RSFT.logic import decode_digit_tokens, exact_digit_match, extract_z_before_answer_from_row, mean_or_zero


def _stable_example_key(ex: PromptExample) -> Tuple[str, int]:
    # Stable key across runs/resume even if upstream dataset iteration order changes.
    payload = f"{ex.question}\n{','.join(str(int(x)) for x in ex.true_digits)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, int(len(ex.prompt_ids))


def _select_eval_examples(examples: Sequence[PromptExample], max_q: int) -> List[PromptExample]:
    if max_q <= 0:
        return []
    if len(examples) <= max_q:
        return list(examples)
    rows = [( _stable_example_key(ex), int(i), ex) for i, ex in enumerate(examples)]
    rows.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    return [x[2] for x in rows[: int(max_q)]]


def _write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_multi_round_policy(
    *,
    engine,
    prompt_rows: Sequence[List[int]],
    true_digits_rows: Sequence[Sequence[int]],
    cfg: Config,
    answer_token_id: int,
    digit_id_to_val: Dict[int, int],
    greedy: bool,
    eval_mode: str,
    max_rounds_override: Optional[int] = None,
    verify_logit_bias: Optional[Dict[int, float]] = None,
    sequence_meta: Optional[Sequence[Dict[str, object]]] = None,
    collect_sequence_logs: bool = False,
) -> Tuple[List[bool], List[float], int, int, List[Dict[str, object]]]:
    if eval_mode not in {"standard", "retry_bias", "oracle_auto_retry"}:
        raise RuntimeError(f"Unsupported eval_mode={eval_mode!r}")
    if eval_mode != "retry_bias" and verify_logit_bias is not None:
        raise RuntimeError("Verify logit bias is only allowed in retry_bias eval mode")

    max_rounds = int(max_rounds_override) if max_rounds_override is not None else int(cfg.rollout.max_rounds)
    n_seq = len(prompt_rows)
    current_prompts = [list(x) for x in prompt_rows]
    status = ["active"] * n_seq
    success = [False] * n_seq
    failure_reason: List[Optional[str]] = [None] * n_seq

    answered_z_lengths: List[float] = []
    no_answer_count = 0
    total_attempts = 0

    verify_allowed = [int(x) for x in list(getattr(engine, "verify_allowed_token_ids", []) or [])]
    finalize_token_id = int(getattr(engine, "finalize_token_id", -1))
    retry_token_id = int(getattr(engine, "retry_token_id", -1))
    if len(verify_allowed) != 2:
        raise RuntimeError("Evaluation requires exactly two verify allowed tokens")
    if sorted(set(verify_allowed)) != sorted({finalize_token_id, retry_token_id}):
        raise RuntimeError("Evaluation verify set must be exactly <FINALIZE>/<RETRY>")

    generated_rollout_token_ids_by_seq: List[List[int]] = [[] for _ in range(n_seq)]
    sequence_rows: List[Dict[str, object]] = []
    if collect_sequence_logs:
        if sequence_meta is None or len(sequence_meta) != n_seq:
            raise RuntimeError("collect_sequence_logs=True requires sequence_meta with one row per sequence")
        for i in range(n_seq):
            meta = dict(sequence_meta[i])
            meta.setdefault("prompt_idx", int(i))
            meta.setdefault("rollout_idx", 0)
            meta.setdefault("question", "")
            meta.setdefault("decode_policy", "greedy" if bool(greedy) else "sampled")
            meta["accepted"] = False
            meta["terminal_status"] = "active"
            meta["failure_reason"] = None
            meta["round_count_observed"] = 0
            meta["full_rollout_token_ids"] = []
            meta["full_sequence_token_ids"] = []
            meta["rounds"] = []
            sequence_rows.append(meta)

    for round_idx in range(1, max_rounds + 1):
        active = [i for i, st in enumerate(status) if st == "active"]
        if not active:
            break

        z_rows = engine.generate_z(
            prompt_token_ids=[current_prompts[i] for i in active],
            num_samples_per_prompt=1,
            max_new_tokens=int(cfg.eval.k_max),
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            greedy=bool(greedy),
        )
        if len(z_rows) != len(active):
            raise RuntimeError("Evaluation Z row count mismatch")

        valid_for_digits: List[int] = []
        z_ids_rows: List[List[int]] = []
        digit_prompts: List[List[int]] = []
        for j, idx in enumerate(active):
            total_attempts += 1
            z_ids = extract_z_before_answer_from_row(z_rows[j], answer_token_id=answer_token_id)
            if z_ids is None:
                no_answer_count += 1
                status[idx] = "failed"
                failure_reason[idx] = "no_answer_before_max_tokens"
                if collect_sequence_logs:
                    true_digits = "".join(str(int(x)) for x in list(true_digits_rows[idx]))
                    sequence_rows[idx]["rounds"].append(  # type: ignore[index]
                        {
                            "round_idx": int(round_idx),
                            "z_len": 0,
                            "z_token_ids": [],
                            "digit_token_ids": [],
                            "pred_digits": "",
                            "true_digits": true_digits,
                            "is_correct": False,
                            "verify_token_id": -1,
                            "verify_action": "NONE",
                            "round_generated_token_ids": [],
                            "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[idx]),
                            "round_event": "no_answer_before_max_tokens",
                        }
                    )
                continue
            answered_z_lengths.append(float(len(z_ids)))
            valid_for_digits.append(idx)
            z_ids_rows.append([int(x) for x in z_ids])
            digit_prompts.append(list(current_prompts[idx]) + list(z_ids) + [int(answer_token_id)])

        if not digit_prompts:
            continue

        digit_rows = engine.generate_digits(
            prompt_token_ids=digit_prompts,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            greedy=bool(greedy),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
        )
        if len(digit_rows) != len(valid_for_digits):
            raise RuntimeError("Evaluation digit row count mismatch")

        verify_prompts: List[List[int]] = []
        verify_owner: List[int] = []
        verify_ctx: List[Tuple[List[int], List[int], bool]] = []
        forced_finalize_ctx: List[Tuple[int, List[int], List[int], bool]] = []
        for j, idx in enumerate(valid_for_digits):
            dig_tokens = [int(x) for x in digit_rows[j]]
            if len(dig_tokens) != 5:
                raise RuntimeError(f"Digits phase must emit exactly 5 tokens per round, got {len(dig_tokens)}")
            pred = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
            if pred is None:
                raise RuntimeError("Digit decode failed during evaluation despite restricted digit set")
            is_correct = bool(exact_digit_match(pred, true_digits_rows[idx]))
            z_ids = z_ids_rows[j]

            if round_idx >= max_rounds:
                forced_finalize_ctx.append((idx, z_ids, dig_tokens, is_correct))
                continue

            if eval_mode == "oracle_auto_retry" and (not is_correct):
                round_generated = list(z_ids) + [int(answer_token_id)] + list(dig_tokens) + [int(retry_token_id)]
                current_prompts[idx].extend(round_generated)
                generated_rollout_token_ids_by_seq[idx].extend(round_generated)
                if collect_sequence_logs:
                    sequence_rows[idx]["rounds"].append(  # type: ignore[index]
                        {
                            "round_idx": int(round_idx),
                            "z_len": int(len(z_ids)),
                            "z_token_ids": list(z_ids),
                            "digit_token_ids": list(dig_tokens),
                            "pred_digits": "".join(str(int(x)) for x in pred),
                            "true_digits": "".join(str(int(x)) for x in list(true_digits_rows[idx])),
                            "is_correct": bool(is_correct),
                            "verify_token_id": int(retry_token_id),
                            "verify_action": "RETRY",
                            "round_generated_token_ids": list(round_generated),
                            "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[idx]),
                        }
                    )
                continue

            verify_prompt = list(current_prompts[idx]) + list(z_ids) + [int(answer_token_id)] + list(dig_tokens)
            verify_prompts.append(verify_prompt)
            verify_owner.append(idx)
            verify_ctx.append((z_ids, dig_tokens, is_correct))

        for idx, z_ids, dig_tokens, is_correct in forced_finalize_ctx:
            round_generated = list(z_ids) + [int(answer_token_id)] + list(dig_tokens) + [int(finalize_token_id)]
            current_prompts[idx].extend(round_generated)
            generated_rollout_token_ids_by_seq[idx].extend(round_generated)
            if collect_sequence_logs:
                pred = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
                if pred is None:
                    raise RuntimeError("Digit decode failed while building forced-finalize trace")
                sequence_rows[idx]["rounds"].append(  # type: ignore[index]
                    {
                        "round_idx": int(round_idx),
                        "z_len": int(len(z_ids)),
                        "z_token_ids": list(z_ids),
                        "digit_token_ids": list(dig_tokens),
                        "pred_digits": "".join(str(int(x)) for x in pred),
                        "true_digits": "".join(str(int(x)) for x in list(true_digits_rows[idx])),
                        "is_correct": bool(is_correct),
                        "verify_token_id": int(finalize_token_id),
                        "verify_action": "FINALIZE",
                        "round_generated_token_ids": list(round_generated),
                        "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[idx]),
                    }
                )
            if is_correct:
                success[idx] = True
                status[idx] = "success"
            else:
                status[idx] = "failed"
                failure_reason[idx] = "max_rounds_reached_without_success"

        if not verify_prompts:
            continue

        verify_rows = engine.generate_verify(
            prompt_token_ids=verify_prompts,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            greedy=bool(greedy),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            logit_bias=(verify_logit_bias if eval_mode == "retry_bias" else None),
        )
        if len(verify_rows) != len(verify_owner):
            raise RuntimeError("Evaluation verify row count mismatch")

        for j, idx in enumerate(verify_owner):
            row = [int(x) for x in verify_rows[j]]
            if len(row) != 1:
                raise RuntimeError(f"Verify phase must emit exactly 1 token, got {len(row)}")
            verify_token_id = int(row[0])
            if verify_token_id not in verify_allowed:
                raise RuntimeError("Invalid verify token observed; verify-phase masking is broken")

            z_ids, dig_tokens, is_correct = verify_ctx[j]
            pred = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
            if pred is None:
                raise RuntimeError("Digit decode failed while building verify trace")
            round_generated = list(z_ids) + [int(answer_token_id)] + list(dig_tokens) + [int(verify_token_id)]
            current_prompts[idx].extend(round_generated)
            generated_rollout_token_ids_by_seq[idx].extend(round_generated)
            if collect_sequence_logs:
                sequence_rows[idx]["rounds"].append(  # type: ignore[index]
                    {
                        "round_idx": int(round_idx),
                        "z_len": int(len(z_ids)),
                        "z_token_ids": list(z_ids),
                        "digit_token_ids": list(dig_tokens),
                        "pred_digits": "".join(str(int(x)) for x in pred),
                        "true_digits": "".join(str(int(x)) for x in list(true_digits_rows[idx])),
                        "is_correct": bool(is_correct),
                        "verify_token_id": int(verify_token_id),
                        "verify_action": ("FINALIZE" if int(verify_token_id) == int(finalize_token_id) else "RETRY"),
                        "round_generated_token_ids": list(round_generated),
                        "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[idx]),
                    }
                )

            if verify_token_id == finalize_token_id:
                if is_correct:
                    success[idx] = True
                    status[idx] = "success"
                else:
                    status[idx] = "failed"
                    failure_reason[idx] = "finalized_wrong_answer"
            elif verify_token_id == retry_token_id:
                if round_idx >= max_rounds:
                    status[idx] = "failed"
                    failure_reason[idx] = "max_rounds_reached_without_success"
            else:
                raise RuntimeError("Invalid verify token observed; verify-phase masking is broken")

    if collect_sequence_logs:
        for i in range(n_seq):
            terminal_status = "success" if bool(success[i]) else "failed"
            reason = failure_reason[i]
            if terminal_status != "success" and reason is None:
                reason = "unknown_failure"
            sequence_rows[i]["accepted"] = bool(success[i])
            sequence_rows[i]["terminal_status"] = str(terminal_status)
            sequence_rows[i]["failure_reason"] = (None if terminal_status == "success" else str(reason))
            sequence_rows[i]["round_count_observed"] = int(len(sequence_rows[i]["rounds"]))  # type: ignore[arg-type]
            sequence_rows[i]["full_rollout_token_ids"] = list(generated_rollout_token_ids_by_seq[i])
            sequence_rows[i]["full_sequence_token_ids"] = list(prompt_rows[i]) + list(generated_rollout_token_ids_by_seq[i])

    return success, answered_z_lengths, int(no_answer_count), int(total_attempts), sequence_rows


def _evaluate_mode(
    *,
    mode_name: str,
    policy_mode: str,
    engine,
    examples: Sequence[PromptExample],
    cfg: Config,
    answer_token_id: int,
    digit_id_to_val: Dict[int, int],
    max_q: int,
    eval_batch_size: int,
    n: int,
    verify_logit_bias: Optional[Dict[int, float]] = None,
    max_rounds_override: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    greedy_success = 0
    passn_success = 0
    no_answer_count = 0
    total_attempts = 0
    answered_z_lengths: List[float] = []
    mode_sequence_rows: List[Dict[str, object]] = []

    for chunk_start in range(0, max_q, eval_batch_size):
        chunk = examples[chunk_start : min(max_q, chunk_start + eval_batch_size)]

        prompt_rows = [list(ex.prompt_ids) for ex in chunk]
        true_rows = [list(ex.true_digits) for ex in chunk]
        greedy_meta: List[Dict[str, object]] = []
        for local_idx, ex in enumerate(chunk):
            greedy_meta.append(
                {
                    "prompt_idx": int(chunk_start + local_idx),
                    "rollout_idx": 0,
                    "question": str(ex.question),
                    "decode_policy": "greedy",
                    "eval_mode": str(mode_name),
                }
            )

        greedy_ok, z_lens_g, no_ans_g, attempts_g, greedy_rows = _run_multi_round_policy(
            engine=engine,
            prompt_rows=prompt_rows,
            true_digits_rows=true_rows,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            greedy=True,
            eval_mode=policy_mode,
            max_rounds_override=max_rounds_override,
            verify_logit_bias=verify_logit_bias,
            sequence_meta=greedy_meta,
            collect_sequence_logs=True,
        )
        greedy_success += sum(1 for ok in greedy_ok if ok)
        answered_z_lengths.extend(z_lens_g)
        no_answer_count += int(no_ans_g)
        total_attempts += int(attempts_g)
        mode_sequence_rows.extend(greedy_rows)

        expanded_prompts: List[List[int]] = []
        expanded_true: List[List[int]] = []
        sampled_meta: List[Dict[str, object]] = []
        owners: List[int] = []
        for idx, ex in enumerate(chunk):
            for rollout_idx in range(n):
                expanded_prompts.append(list(ex.prompt_ids))
                expanded_true.append(list(ex.true_digits))
                owners.append(idx)
                sampled_meta.append(
                    {
                        "prompt_idx": int(chunk_start + idx),
                        "rollout_idx": int(rollout_idx),
                        "question": str(ex.question),
                        "decode_policy": "pass_at_n_sample",
                        "eval_mode": str(mode_name),
                    }
                )

        sampled_ok, z_lens_s, no_ans_s, attempts_s, sampled_rows = _run_multi_round_policy(
            engine=engine,
            prompt_rows=expanded_prompts,
            true_digits_rows=expanded_true,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            greedy=False,
            eval_mode=policy_mode,
            max_rounds_override=max_rounds_override,
            verify_logit_bias=verify_logit_bias,
            sequence_meta=sampled_meta,
            collect_sequence_logs=True,
        )
        answered_z_lengths.extend(z_lens_s)
        no_answer_count += int(no_ans_s)
        total_attempts += int(attempts_s)
        mode_sequence_rows.extend(sampled_rows)

        pass_ok = [False] * len(chunk)
        for j, ok in enumerate(sampled_ok):
            if ok:
                pass_ok[owners[j]] = True
        passn_success += sum(1 for ok in pass_ok if ok)

    attempts = max(total_attempts, 1)
    return (
        {
            f"greedy_exact_{mode_name}": float(greedy_success) / float(max_q),
            f"pass_at_n_{mode_name}": float(passn_success) / float(max_q),
            f"mean_z_length_{mode_name}": mean_or_zero(answered_z_lengths),
            f"no_answer_before_kmax_rate_{mode_name}": float(no_answer_count) / float(attempts),
        },
        mode_sequence_rows,
    )


def evaluate_with_rollout_engine(
    *,
    engine,
    examples: Sequence[PromptExample],
    cfg: Config,
    answer_token_id: int,
    digit_id_to_val: Dict[int, int],
    eval_log_dir: Optional[str] = None,
    eval_step: Optional[int] = None,
) -> Dict[str, object]:
    mode_file_keys = ("standard", "retry_bias", "oracle_retry")
    max_q = min(len(examples), int(cfg.eval.max_eval_questions))
    selected_examples = _select_eval_examples(examples, int(max_q))
    max_q = int(len(selected_examples))
    if max_q <= 0:
        out = {
            "evaluated_questions": 0.0,
            "greedy_exact": 0.0,
            "pass_at_n": 0.0,
            "mean_z_length": 0.0,
            "no_answer_before_kmax_rate": 0.0,
            "greedy_exact_standard": 0.0,
            "pass_at_n_standard": 0.0,
            "mean_z_length_standard": 0.0,
            "no_answer_before_kmax_rate_standard": 0.0,
            "eval_modes_ran": "standard",
        }
        if eval_log_dir:
            step_dir = os.path.join(str(eval_log_dir), f"step_{int(eval_step) if eval_step is not None else -1:06d}")
            for mode_name in mode_file_keys:
                _write_jsonl(os.path.join(step_dir, f"{mode_name}.jsonl"), [])
            out["eval_rollout_log_dir"] = str(step_dir)
        return out

    eval_batch_size = max(1, int(getattr(cfg.eval, "vllm_batch_size", 1)))
    n = max(1, int(cfg.eval.pass_at_n))

    out: Dict[str, object] = {"evaluated_questions": float(max_q)}
    modes_ran: List[str] = ["standard"]
    sequence_logs_by_mode: Dict[str, List[Dict[str, object]]] = {
        "standard": [],
        "retry_bias": [],
        "oracle_retry": [],
    }

    standard, standard_rows = _evaluate_mode(
        mode_name="standard",
        policy_mode="standard",
        engine=engine,
        examples=selected_examples,
        cfg=cfg,
        answer_token_id=answer_token_id,
        digit_id_to_val=digit_id_to_val,
        max_q=max_q,
        eval_batch_size=eval_batch_size,
        n=n,
    )
    out.update(standard)
    sequence_logs_by_mode["standard"] = standard_rows

    out["greedy_exact"] = float(standard["greedy_exact_standard"])
    out["pass_at_n"] = float(standard["pass_at_n_standard"])
    out["mean_z_length"] = float(standard["mean_z_length_standard"])
    out["no_answer_before_kmax_rate"] = float(standard["no_answer_before_kmax_rate_standard"])

    if bool(cfg.eval.eval_retry_bias_enabled):
        retry_token_id = int(getattr(engine, "retry_token_id", -1))
        finalize_token_id = int(getattr(engine, "finalize_token_id", -1))
        retry_bias = float(cfg.eval.verify_retry_logit_bias)
        verify_bias = {
            int(retry_token_id): float(retry_bias),
            int(finalize_token_id): 0.0,
        }
        retry_metrics, retry_rows = _evaluate_mode(
            mode_name="retry_bias",
            policy_mode="retry_bias",
            engine=engine,
            examples=selected_examples,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            max_q=max_q,
            eval_batch_size=eval_batch_size,
            n=n,
            verify_logit_bias=verify_bias,
        )
        out.update(retry_metrics)
        out["verify_retry_logit_bias"] = float(retry_bias)
        modes_ran.append("retry_bias")
        sequence_logs_by_mode["retry_bias"] = retry_rows

    if bool(cfg.eval.eval_oracle_auto_retry_enabled):
        oracle_rounds = int(cfg.eval.oracle_auto_retry_max_rounds)
        oracle_metrics, oracle_rows = _evaluate_mode(
            mode_name="oracle_retry",
            policy_mode="oracle_auto_retry",
            engine=engine,
            examples=selected_examples,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            max_q=max_q,
            eval_batch_size=eval_batch_size,
            n=n,
            max_rounds_override=oracle_rounds,
        )
        out.update(oracle_metrics)
        out["oracle_auto_retry_max_rounds"] = float(oracle_rounds)
        modes_ran.append("oracle_auto_retry")
        sequence_logs_by_mode["oracle_retry"] = oracle_rows

    out["eval_modes_ran"] = ",".join(modes_ran)

    if eval_log_dir:
        step_dir = os.path.join(str(eval_log_dir), f"step_{int(eval_step) if eval_step is not None else -1:06d}")
        for mode_name in mode_file_keys:
            _write_jsonl(os.path.join(step_dir, f"{mode_name}.jsonl"), sequence_logs_by_mode.get(mode_name, []))
        out["eval_rollout_log_dir"] = str(step_dir)

    return out
