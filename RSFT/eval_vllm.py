from typing import Dict, List, Sequence, Tuple

from RSFT.config import Config
from RSFT.dataset import PromptExample
from RSFT.logic import decode_digit_tokens, exact_digit_match, extract_z_before_answer_from_row, mean_or_zero


def _chunks(rows: Sequence[PromptExample], batch_size: int) -> List[Sequence[PromptExample]]:
    k = max(1, int(batch_size))
    return [rows[i : i + k] for i in range(0, len(rows), k)]


def _run_multi_round_policy(
    *,
    engine,
    prompt_rows: Sequence[List[int]],
    true_digits_rows: Sequence[Sequence[int]],
    cfg: Config,
    answer_token_id: int,
    digit_id_to_val: Dict[int, int],
    greedy: bool,
) -> Tuple[List[bool], List[float], int, int]:
    max_rounds = int(cfg.rollout.max_rounds)
    n_seq = len(prompt_rows)
    current_prompts = [list(x) for x in prompt_rows]
    status = ["active"] * n_seq
    success = [False] * n_seq

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
        for j, idx in enumerate(valid_for_digits):
            dig_tokens = [int(x) for x in digit_rows[j]]
            if len(dig_tokens) != 5:
                raise RuntimeError(f"Digits phase must emit exactly 5 tokens per round, got {len(dig_tokens)}")
            pred = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
            if pred is None:
                raise RuntimeError("Digit decode failed during evaluation despite restricted digit set")
            is_correct = bool(exact_digit_match(pred, true_digits_rows[idx]))

            z_ids = z_ids_rows[j]
            verify_prompt = list(current_prompts[idx]) + list(z_ids) + [int(answer_token_id)] + list(dig_tokens)
            verify_prompts.append(verify_prompt)
            verify_owner.append(idx)
            verify_ctx.append((z_ids, dig_tokens, is_correct))

        verify_rows = engine.generate_verify(
            prompt_token_ids=verify_prompts,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            greedy=bool(greedy),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
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
            current_prompts[idx].extend(list(z_ids))
            current_prompts[idx].append(int(answer_token_id))
            current_prompts[idx].extend(list(dig_tokens))
            current_prompts[idx].append(int(verify_token_id))

            if verify_token_id == finalize_token_id:
                if is_correct:
                    success[idx] = True
                status[idx] = "done"
            elif verify_token_id == retry_token_id:
                if round_idx >= max_rounds:
                    status[idx] = "failed"
            else:
                raise RuntimeError("Invalid verify token observed; verify-phase masking is broken")

    return success, answered_z_lengths, int(no_answer_count), int(total_attempts)


def evaluate_with_rollout_engine(
    *,
    engine,
    examples: Sequence[PromptExample],
    cfg: Config,
    answer_token_id: int,
    digit_id_to_val: Dict[int, int],
) -> Dict[str, float]:
    max_q = min(len(examples), int(cfg.eval.max_eval_questions))
    if max_q <= 0:
        return {
            "evaluated_questions": 0.0,
            "greedy_exact": 0.0,
            "pass_at_n": 0.0,
            "mean_z_length": 0.0,
            "no_answer_before_kmax_rate": 0.0,
        }

    eval_batch_size = max(1, int(getattr(cfg.eval, "vllm_batch_size", 1)))
    n = max(1, int(cfg.eval.pass_at_n))

    greedy_success = 0
    passn_success = 0
    no_answer_count = 0
    total_attempts = 0
    answered_z_lengths: List[float] = []

    for chunk in _chunks(examples[:max_q], eval_batch_size):
        prompt_rows = [list(ex.prompt_ids) for ex in chunk]
        true_rows = [list(ex.true_digits) for ex in chunk]

        greedy_ok, z_lens_g, no_ans_g, attempts_g = _run_multi_round_policy(
            engine=engine,
            prompt_rows=prompt_rows,
            true_digits_rows=true_rows,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            greedy=True,
        )
        greedy_success += sum(1 for ok in greedy_ok if ok)
        answered_z_lengths.extend(z_lens_g)
        no_answer_count += int(no_ans_g)
        total_attempts += int(attempts_g)

        expanded_prompts: List[List[int]] = []
        expanded_true: List[List[int]] = []
        owners: List[int] = []
        for idx, ex in enumerate(chunk):
            for _ in range(n):
                expanded_prompts.append(list(ex.prompt_ids))
                expanded_true.append(list(ex.true_digits))
                owners.append(idx)

        sampled_ok, z_lens_s, no_ans_s, attempts_s = _run_multi_round_policy(
            engine=engine,
            prompt_rows=expanded_prompts,
            true_digits_rows=expanded_true,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
            greedy=False,
        )
        answered_z_lengths.extend(z_lens_s)
        no_answer_count += int(no_ans_s)
        total_attempts += int(attempts_s)

        pass_ok = [False] * len(chunk)
        for j, ok in enumerate(sampled_ok):
            if ok:
                pass_ok[owners[j]] = True
        passn_success += sum(1 for ok in pass_ok if ok)

    attempts = max(total_attempts, 1)
    return {
        "evaluated_questions": float(max_q),
        "greedy_exact": float(greedy_success) / float(max_q),
        "pass_at_n": float(passn_success) / float(max_q),
        "mean_z_length": mean_or_zero(answered_z_lengths),
        "no_answer_before_kmax_rate": float(no_answer_count) / float(attempts),
    }
