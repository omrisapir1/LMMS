

from typing import Dict, List, Sequence

from RSFT.config import Config
from RSFT.dataset import PromptExample
from RSFT.logic import decode_digit_tokens, exact_digit_match, extract_z_before_answer_from_row, mean_or_zero


from typing import Dict, List, Sequence

from RSFT.config import Config
from RSFT.dataset import PromptExample
from RSFT.logic import decode_digit_tokens, exact_digit_match, extract_z_before_answer_from_row, mean_or_zero


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

    for ex in examples[:max_q]:
        prompt_row = [list(ex.prompt_ids)]

        # Greedy exact metric: true greedy decoding in Z and digit phases.
        greedy_z_rows = engine.generate_z(
            prompt_token_ids=prompt_row,
            max_new_tokens=int(cfg.eval.k_max),
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            greedy=True,
        )
        greedy_z = extract_z_before_answer_from_row(greedy_z_rows[0], answer_token_id=answer_token_id)
        if greedy_z is not None:
            greedy_digits = engine.generate_digits(
                prompt_token_ids=[list(ex.prompt_ids) + list(greedy_z) + [int(answer_token_id)]],
                temperature=float(cfg.rollout.temperature),
                top_p=float(cfg.rollout.top_p),
                greedy=True,
                min_p=float(cfg.rollout.min_p),
                repetition_penalty=float(cfg.rollout.repetition_penalty),
            )[0]
            greedy_pred = decode_digit_tokens(greedy_digits, digit_id_to_val=digit_id_to_val)
            if greedy_pred is not None and exact_digit_match(greedy_pred, ex.true_digits):
                greedy_success += 1

        # pass@N and no-answer metrics
        n = int(cfg.eval.pass_at_n)
        prompt_batch = [list(ex.prompt_ids) for _ in range(n)]
        z_rows = engine.generate_z(
            prompt_token_ids=prompt_batch,
            max_new_tokens=int(cfg.eval.k_max),
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
            greedy=False,
        )

        candidate_prompt_ids: List[List[int]] = []
        for row in z_rows:
            total_attempts += 1
            z_ids = extract_z_before_answer_from_row(row, answer_token_id=answer_token_id)
            if z_ids is None:
                no_answer_count += 1
                continue
            answered_z_lengths.append(float(len(z_ids)))
            candidate_prompt_ids.append(list(ex.prompt_ids) + list(z_ids) + [int(answer_token_id)])

        if not candidate_prompt_ids:
            continue

        digit_rows = engine.generate_digits(
            prompt_token_ids=candidate_prompt_ids,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            greedy=bool(cfg.rollout.digit_greedy),
            min_p=float(cfg.rollout.min_p),
            repetition_penalty=float(cfg.rollout.repetition_penalty),
        )

        ok = False
        for dig in digit_rows:
            pred = decode_digit_tokens(dig, digit_id_to_val=digit_id_to_val)
            if pred is not None and exact_digit_match(pred, ex.true_digits):
                ok = True
                break
        if ok:
            passn_success += 1

    attempts = max(total_attempts, 1)
    return {
        "evaluated_questions": float(max_q),
        "greedy_exact": float(greedy_success) / float(max_q),
        "pass_at_n": float(passn_success) / float(max_q),
        "mean_z_length": mean_or_zero(answered_z_lengths),
        "no_answer_before_kmax_rate": float(no_answer_count) / float(attempts),
    }
