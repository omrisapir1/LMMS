from typing import Dict, List, Sequence

from RSFT.config import Config
from RSFT.dataset import PromptExample
from RSFT.logic import decode_digit_tokens, exact_digit_match, extract_z_before_answer_from_row, mean_or_zero


def _chunks(rows: Sequence[PromptExample], batch_size: int) -> List[Sequence[PromptExample]]:
    k = max(1, int(batch_size))
    return [rows[i : i + k] for i in range(0, len(rows), k)]


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

    greedy_success = 0
    passn_success = 0
    no_answer_count = 0
    total_attempts = 0
    answered_z_lengths: List[float] = []
    eval_batch_size = max(1, int(getattr(cfg.eval, "vllm_batch_size", 1)))
    n = max(1, int(cfg.eval.pass_at_n))

    for chunk in _chunks(examples[:max_q], eval_batch_size):
        prompt_rows = [list(ex.prompt_ids) for ex in chunk]

        # Greedy exact metric: true greedy decoding in Z and digit phases, batched.
        greedy_z_rows = engine.generate_z(
            prompt_token_ids=prompt_rows,
            num_samples_per_prompt=1,
            max_new_tokens=int(cfg.eval.k_max),
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            greedy=True,
        )
        greedy_digit_prompts: List[List[int]] = []
        greedy_idx_map: List[int] = []
        for ex_idx, ex in enumerate(chunk):
            greedy_z = extract_z_before_answer_from_row(greedy_z_rows[ex_idx], answer_token_id=answer_token_id)
            if greedy_z is None:
                continue
            greedy_digit_prompts.append(list(ex.prompt_ids) + list(greedy_z) + [int(answer_token_id)])
            greedy_idx_map.append(ex_idx)
        if greedy_digit_prompts:
            greedy_digit_rows = engine.generate_digits(
                prompt_token_ids=greedy_digit_prompts,
                temperature=float(cfg.rollout.temperature),
                top_p=float(cfg.rollout.top_p),
                greedy=True,
                min_p=float(cfg.rollout.min_p),
                repetition_penalty=float(cfg.rollout.repetition_penalty),
            )
            for j, dig in enumerate(greedy_digit_rows):
                ex_idx = greedy_idx_map[j]
                pred = decode_digit_tokens(dig, digit_id_to_val=digit_id_to_val)
                if pred is not None and exact_digit_match(pred, chunk[ex_idx].true_digits):
                    greedy_success += 1

        # pass@N and no-answer metrics, batched by prompts.
        z_rows = engine.generate_z(
            prompt_token_ids=prompt_rows,
            num_samples_per_prompt=n,
            max_new_tokens=int(cfg.eval.k_max),
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            greedy=False,
        )
        candidate_prompt_ids: List[List[int]] = []
        candidate_owner: List[int] = []
        passn_ok: List[bool] = [False] * len(chunk)
        for flat_idx, row in enumerate(z_rows):
            ex_idx = flat_idx // int(n)
            total_attempts += 1
            z_ids = extract_z_before_answer_from_row(row, answer_token_id=answer_token_id)
            if z_ids is None:
                no_answer_count += 1
                continue
            answered_z_lengths.append(float(len(z_ids)))
            candidate_prompt_ids.append(list(chunk[ex_idx].prompt_ids) + list(z_ids) + [int(answer_token_id)])
            candidate_owner.append(ex_idx)

        if candidate_prompt_ids:
            digit_rows = engine.generate_digits(
                prompt_token_ids=candidate_prompt_ids,
                temperature=float(cfg.rollout.temperature),
                top_p=float(cfg.rollout.top_p),
                greedy=bool(cfg.rollout.digit_greedy),
                min_p=float(cfg.rollout.min_p),
                repetition_penalty=float(cfg.rollout.repetition_penalty),
            )
            for j, dig in enumerate(digit_rows):
                ex_idx = candidate_owner[j]
                pred = decode_digit_tokens(dig, digit_id_to_val=digit_id_to_val)
                if pred is not None and exact_digit_match(pred, chunk[ex_idx].true_digits):
                    passn_ok[ex_idx] = True
        passn_success += sum(1 for ok in passn_ok if ok)

    attempts = max(total_attempts, 1)
    return {
        "evaluated_questions": float(max_q),
        "greedy_exact": float(greedy_success) / float(max_q),
        "pass_at_n": float(passn_success) / float(max_q),
        "mean_z_length": mean_or_zero(answered_z_lengths),
        "no_answer_before_kmax_rate": float(no_answer_count) / float(attempts),
    }
