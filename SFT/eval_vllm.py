from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import AutoTokenizer


@dataclass
class EvalMetrics:
    pass_at_n: float
    greedy_exact_match: float
    greedy_keep_first_z_exact_match: float
    greedy_keep_last_z_exact_match: float
    greedy_reverse_z_exact_match: float
    greedy_random_z_exact_match: float
    mean_z_len: float
    no_answer_before_kmax_rate: float


def find_first_answer_pos(token_ids: List[int], answer_token_id: int) -> int:
    for i, tid in enumerate(token_ids):
        if int(tid) == int(answer_token_id):
            return i
    return -1


def truncate_phaseA_to_answer(token_ids: List[int], answer_token_id: int) -> tuple[List[int], bool]:
    pos = find_first_answer_pos(token_ids, answer_token_id)
    if pos < 0:
        return list(token_ids), False
    return list(token_ids[: pos + 1]), True


def _build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _phase_b_digits(token_ids: List[int], digit_id_to_val: Dict[int, int]) -> List[int]:
    out: List[int] = []
    for tid in token_ids:
        if int(tid) in digit_id_to_val:
            out.append(int(digit_id_to_val[int(tid)]))
    return out


def _parse_target_digits(row: Dict) -> List[int] | None:
    digits = row.get("answer_digits")
    if digits is not None:
        try:
            d = [int(x) for x in digits]
            if len(d) == 5 and not any(x < 0 or x > 9 for x in d):
                return d
        except Exception:
            pass

    final_answer = row.get("final_answer")
    if final_answer is None:
        return None
    try:
        text = str(final_answer).strip()
        if text.startswith("+"):
            text = text[1:]
        if not text.isdigit():
            return None
        if len(text) < 1 or len(text) > 5:
            return None
        padded = text.zfill(5)
        return [int(ch) for ch in padded]
    except Exception:
        return None


def _sample_random_z_ids(z_token_ids: List[int], length: int, seed: int) -> List[int]:
    if length <= 0:
        return []
    rng = random.Random(int(seed))
    if length <= len(z_token_ids):
        return [int(x) for x in rng.sample(z_token_ids, k=length)]
    return [int(rng.choice(z_token_ids)) for _ in range(length)]


def evaluate_with_vllm(
    *,
    model_path: str,
    records: Iterable[Dict],
    pass_at_n: int,
    k_max: int,
    temperature: float,
    top_p: float,
    vocab_size: int,
    vllm_cuda_visible_devices: str | None = None,
    output_jsonl_path: str | None = None,
) -> EvalMetrics:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("vLLM is required for SFT evaluation and is not available.") from exc

    tokenizer = AutoTokenizer.from_pretrained('/root/.cache/huggingface/hub/models--omrisap--Qwen-9B-512Z/snapshots/540bddc5affd6f3bfe4f977131a762d001cc112b', use_fast=True)

    z_tokens = [f"<z_{i}>" for i in range(int(vocab_size))]
    z_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in z_tokens]
    if any(i < 0 for i in z_token_ids):
        raise RuntimeError("Tokenizer missing some Z tokens for evaluation.")

    closing_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(closing_think_ids) != 1:
        raise RuntimeError(f"Tokenizer closing think token must be single-token, got {closing_think_ids}")
    answer_token_id = int(closing_think_ids[0])

    digit_token_ids: List[int] = []
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Digit tokenization check failed in eval for '{d}' -> {ids}")
        digit_token_ids.append(int(ids[0]))

    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vllm_cuda_visible_devices is not None and str(vllm_cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_cuda_visible_devices)
    try:
        llm = LLM(model="/root/.cache/huggingface/hub/models--omrisap--Qwen-9B-512Z/snapshots/540bddc5affd6f3bfe4f977131a762d001cc112b_vllm", tokenizer='/root/.cache/huggingface/hub/models--omrisap--Qwen-9B-512Z/snapshots/540bddc5affd6f3bfe4f977131a762d001cc112b', trust_remote_code=True, gpu_memory_utilization=0.95)
    finally:
        if old_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices
    z_token_id_set = set(z_token_ids)
    digit_id_to_val = {tid: i for i, tid in enumerate(digit_token_ids)}

    writer = None
    if output_jsonl_path:
        Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        writer = open(output_jsonl_path, "a", encoding="utf-8")

    items: List[Dict] = []
    labeled_items = 0
    for row in records:
        q = row.get("problem")
        if q is None:
            q = row.get("question")
        if q is None:
            continue
        target_digits = _parse_target_digits(row)
        if target_digits is not None:
            labeled_items += 1
        items.append({"problem": str(q), "answer_digits": target_digits})

    if not items:
        try:
            if writer is not None:
                writer.write(json.dumps({"warning": "no eval items with a problem field"}) + "\n")
        finally:
            if writer is not None:
                writer.close()
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    prompts = [_build_prompt(tokenizer, x["problem"]) for x in items]

    phase_a_sampled_params = SamplingParams(
        n=int(pass_at_n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(k_max) + 1,
        allowed_token_ids=sorted(list(z_token_ids) + [answer_token_id]),
        stop_token_ids=[answer_token_id],
    )
    phase_a_greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(k_max) + 1,
        allowed_token_ids=sorted(list(z_token_ids) + [answer_token_id]),
        stop_token_ids=[answer_token_id],
    )
    phase_b_sampled_params = SamplingParams(
        n=1,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=5,
        allowed_token_ids=sorted(digit_token_ids),
    )
    phase_b_greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=5,
        allowed_token_ids=sorted(digit_token_ids),
    )

    sampled_phase_a_outputs = llm.generate(prompts, phase_a_sampled_params)
    if len(sampled_phase_a_outputs) != len(prompts):
        raise RuntimeError(
            f"sampled_phase_a_outputs length mismatch: "
            f"{len(sampled_phase_a_outputs)} != {len(prompts)}"
        )
    greedy_phase_a_outputs = llm.generate(prompts, phase_a_greedy_params)
    if len(greedy_phase_a_outputs) != len(prompts):
        raise RuntimeError(
            f"greedy_phase_a_outputs length mismatch: "
            f"{len(greedy_phase_a_outputs)} != {len(prompts)}"
        )

    sampled_phase_b_prompts: List[str] = []
    sampled_phase_b_owner: List[tuple[int, int]] = []
    sampled_phase_a_ids: List[List[List[int]]] = []
    sampled_phase_a_texts: List[List[str]] = []
    sampled_phase_a_has_answer: List[List[bool]] = []
    sampled_phase_a_z_lens: List[List[int]] = []

    for item_idx, (prompt, sampled_out) in enumerate(zip(prompts, sampled_phase_a_outputs)):
        row_ids: List[List[int]] = []
        row_texts: List[str] = []
        row_has_answer: List[bool] = []
        row_z_lens: List[int] = []
        for cand_idx, candidate in enumerate(sampled_out.outputs):
            raw_ids = [int(x) for x in candidate.token_ids]
            trunc_ids, has_answer = truncate_phaseA_to_answer(raw_ids, answer_token_id)
            row_ids.append(trunc_ids)
            row_texts.append(tokenizer.decode(trunc_ids, skip_special_tokens=False))
            row_has_answer.append(bool(has_answer))
            if has_answer:
                ans_pos = find_first_answer_pos(trunc_ids, answer_token_id)
                z_len = sum(1 for t in trunc_ids[:ans_pos] if int(t) in z_token_id_set)
                row_z_lens.append(int(z_len))
                sampled_phase_b_prompts.append(prompt + row_texts[-1])
                sampled_phase_b_owner.append((item_idx, cand_idx))
            else:
                row_z_lens.append(0)
        sampled_phase_a_ids.append(row_ids)
        sampled_phase_a_texts.append(row_texts)
        sampled_phase_a_has_answer.append(row_has_answer)
        sampled_phase_a_z_lens.append(row_z_lens)

    sampled_phase_b_ids_by_owner: Dict[tuple[int, int], List[int]] = {}
    sampled_phase_b_text_by_owner: Dict[tuple[int, int], str] = {}
    if sampled_phase_b_prompts:
        sampled_phase_b_outputs = llm.generate(sampled_phase_b_prompts, phase_b_sampled_params)
        if len(sampled_phase_b_outputs) != len(sampled_phase_b_prompts):
            raise RuntimeError(
                f"sampled_phase_b_outputs length mismatch: "
                f"{len(sampled_phase_b_outputs)} != {len(sampled_phase_b_prompts)}"
            )
        for owner, out in zip(sampled_phase_b_owner, sampled_phase_b_outputs):
            phase_b_ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            sampled_phase_b_ids_by_owner[owner] = phase_b_ids
            sampled_phase_b_text_by_owner[owner] = tokenizer.decode(phase_b_ids, skip_special_tokens=False)

    greedy_phase_b_prompts: List[str] = []
    greedy_phase_b_owner: List[int] = []
    greedy_phase_a_ids: List[List[int]] = []
    greedy_phase_a_texts: List[str] = []
    greedy_phase_a_has_answer: List[bool] = []
    greedy_phase_a_z_lens: List[int] = []
    greedy_phase_a_z_ids: List[List[int]] = []

    corruption_variants = ("keep_first_z", "keep_last_z", "reverse_z", "random_z")
    corrupted_phase_b_prompts: List[str] = []
    corrupted_phase_b_owner: List[tuple[int, str]] = []
    corrupted_phase_a_ids_by_owner: Dict[tuple[int, str], List[int]] = {}
    corrupted_phase_a_text_by_owner: Dict[tuple[int, str], str] = {}

    for item_idx, (prompt, greedy_out) in enumerate(zip(prompts, greedy_phase_a_outputs)):
        raw_ids = [int(x) for x in greedy_out.outputs[0].token_ids] if greedy_out.outputs else []
        trunc_ids, has_answer = truncate_phaseA_to_answer(raw_ids, answer_token_id)
        phase_a_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)
        greedy_phase_a_ids.append(trunc_ids)
        greedy_phase_a_texts.append(phase_a_text)
        greedy_phase_a_has_answer.append(bool(has_answer))
        if has_answer:
            ans_pos = find_first_answer_pos(trunc_ids, answer_token_id)
            z_ids = [int(tid) for tid in trunc_ids[:ans_pos] if int(tid) in z_token_id_set]
            z_len = len(z_ids)
            greedy_phase_a_z_lens.append(int(z_len))
            greedy_phase_a_z_ids.append(z_ids)
            greedy_phase_b_prompts.append(prompt + phase_a_text)
            greedy_phase_b_owner.append(item_idx)

            z_first = z_ids[:1]
            z_last = z_ids[-1:]
            z_rev = list(reversed(z_ids))
            z_rand = _sample_random_z_ids(z_token_ids, len(z_ids), seed=1337 + item_idx)
            corrupted_map = {
                "keep_first_z": z_first,
                "keep_last_z": z_last,
                "reverse_z": z_rev,
                "random_z": z_rand,
            }
            for variant_name in corruption_variants:
                z_ids_corrupt = corrupted_map[variant_name]
                phase_a_ids_corrupt = list(z_ids_corrupt) + [answer_token_id]
                phase_a_text_corrupt = tokenizer.decode(phase_a_ids_corrupt, skip_special_tokens=False)
                owner = (item_idx, variant_name)
                corrupted_phase_a_ids_by_owner[owner] = phase_a_ids_corrupt
                corrupted_phase_a_text_by_owner[owner] = phase_a_text_corrupt
                corrupted_phase_b_owner.append(owner)
                corrupted_phase_b_prompts.append(prompt + phase_a_text_corrupt)
        else:
            greedy_phase_a_z_lens.append(0)
            greedy_phase_a_z_ids.append([])

    greedy_phase_b_ids_by_owner: Dict[int, List[int]] = {}
    greedy_phase_b_text_by_owner: Dict[int, str] = {}
    if greedy_phase_b_prompts:
        greedy_phase_b_outputs = llm.generate(greedy_phase_b_prompts, phase_b_greedy_params)
        if len(greedy_phase_b_outputs) != len(greedy_phase_b_prompts):
            raise RuntimeError(
                f"greedy_phase_b_outputs length mismatch: "
                f"{len(greedy_phase_b_outputs)} != {len(greedy_phase_b_prompts)}"
            )
        for owner, out in zip(greedy_phase_b_owner, greedy_phase_b_outputs):
            phase_b_ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            greedy_phase_b_ids_by_owner[owner] = phase_b_ids
            greedy_phase_b_text_by_owner[owner] = tokenizer.decode(phase_b_ids, skip_special_tokens=False)

    corrupted_phase_b_ids_by_owner: Dict[tuple[int, str], List[int]] = {}
    corrupted_phase_b_text_by_owner: Dict[tuple[int, str], str] = {}
    if corrupted_phase_b_prompts:
        corrupted_phase_b_outputs = llm.generate(corrupted_phase_b_prompts, phase_b_greedy_params)
        if len(corrupted_phase_b_outputs) != len(corrupted_phase_b_prompts):
            raise RuntimeError(
                f"corrupted_phase_b_outputs length mismatch: "
                f"{len(corrupted_phase_b_outputs)} != {len(corrupted_phase_b_prompts)}"
            )
        for owner, out in zip(corrupted_phase_b_owner, corrupted_phase_b_outputs):
            phase_b_ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            corrupted_phase_b_ids_by_owner[owner] = phase_b_ids
            corrupted_phase_b_text_by_owner[owner] = tokenizer.decode(phase_b_ids, skip_special_tokens=False)

    pass_hits = 0
    greedy_hits = 0
    greedy_keep_first_z_hits = 0
    greedy_keep_last_z_hits = 0
    greedy_reverse_z_hits = 0
    greedy_random_z_hits = 0
    z_lens: List[int] = []
    no_answer = 0

    try:
        for i, (row, prompt) in enumerate(zip(items, prompts)):
            target = row["answer_digits"]
            has_label = target is not None

            sample_correct = False
            row_has_answer = any(sampled_phase_a_has_answer[i]) if i < len(sampled_phase_a_has_answer) else False
            sampled_candidates: List[Dict] = []

            for cand_idx, phase_a_ids in enumerate(sampled_phase_a_ids[i]):
                phase_a_text = sampled_phase_a_texts[i][cand_idx]
                phase_a_has_answer = sampled_phase_a_has_answer[i][cand_idx]
                phase_a_z_len = sampled_phase_a_z_lens[i][cand_idx]
                phase_b_ids = sampled_phase_b_ids_by_owner.get((i, cand_idx), [])
                phase_b_text = sampled_phase_b_text_by_owner.get((i, cand_idx), "")
                full_sequence = prompt + phase_a_text + phase_b_text

                if phase_a_has_answer:
                    z_lens.append(int(phase_a_z_len))
                    if has_label:
                        digits_pred = _phase_b_digits(phase_b_ids, digit_id_to_val)
                        if len(digits_pred) == 5 and digits_pred == target:
                            sample_correct = True

                sampled_candidates.append(
                    {
                        "phaseA_token_ids": phase_a_ids,
                        "phaseA_text": phase_a_text,
                        "phaseA_has_answer": bool(phase_a_has_answer),
                        "phaseA_z_len": int(phase_a_z_len),
                        "phaseB_token_ids": phase_b_ids if phase_a_has_answer else None,
                        "phaseB_text": phase_b_text if phase_a_has_answer else "",
                        "full_sequence": full_sequence,
                    }
                )

            greedy_correct = False
            greedy_phase_a_ids_row = greedy_phase_a_ids[i]
            greedy_phase_a_text_row = greedy_phase_a_texts[i]
            greedy_has_answer = greedy_phase_a_has_answer[i]
            greedy_phase_a_z_len_row = greedy_phase_a_z_lens[i]
            greedy_phase_a_z_ids_row = greedy_phase_a_z_ids[i]
            greedy_phase_b_ids_row = greedy_phase_b_ids_by_owner.get(i, [])
            greedy_phase_b_text_row = greedy_phase_b_text_by_owner.get(i, "")

            if has_label and greedy_has_answer:
                greedy_digits = _phase_b_digits(greedy_phase_b_ids_row, digit_id_to_val)
                greedy_correct = len(greedy_digits) == 5 and greedy_digits == target

            keep_first_correct = False
            keep_last_correct = False
            reverse_correct = False
            random_correct = False
            greedy_corruptions: Dict[str, Dict] = {}
            for variant_name in corruption_variants:
                owner = (i, variant_name)
                phase_a_ids_corrupt = corrupted_phase_a_ids_by_owner.get(owner)
                phase_a_text_corrupt = corrupted_phase_a_text_by_owner.get(owner, "")
                phase_b_ids_corrupt = corrupted_phase_b_ids_by_owner.get(owner, [])
                phase_b_text_corrupt = corrupted_phase_b_text_by_owner.get(owner, "")
                exact_match = False
                digits_pred: List[int] = []
                if has_label and greedy_has_answer:
                    digits_pred = _phase_b_digits(phase_b_ids_corrupt, digit_id_to_val)
                    exact_match = len(digits_pred) == 5 and digits_pred == target
                greedy_corruptions[variant_name] = {
                    "phaseA_text_corrupt": phase_a_text_corrupt if greedy_has_answer else "",
                    "phaseA_token_ids_corrupt": phase_a_ids_corrupt if greedy_has_answer else None,
                    "phaseB_token_ids": phase_b_ids_corrupt if greedy_has_answer else None,
                    "phaseB_text": phase_b_text_corrupt if greedy_has_answer else "",
                    "digits_pred": digits_pred if greedy_has_answer else [],
                    "exact_match": bool(exact_match),
                }
                if variant_name == "keep_first_z":
                    keep_first_correct = bool(exact_match)
                elif variant_name == "keep_last_z":
                    keep_last_correct = bool(exact_match)
                elif variant_name == "reverse_z":
                    reverse_correct = bool(exact_match)
                elif variant_name == "random_z":
                    random_correct = bool(exact_match)

            if has_label:
                if sample_correct:
                    pass_hits += 1
                if greedy_correct:
                    greedy_hits += 1
                if keep_first_correct:
                    greedy_keep_first_z_hits += 1
                if keep_last_correct:
                    greedy_keep_last_z_hits += 1
                if reverse_correct:
                    greedy_reverse_z_hits += 1
                if random_correct:
                    greedy_random_z_hits += 1
                if not row_has_answer:
                    no_answer += 1

            if writer is not None:
                writer.write(
                    json.dumps(
                        {
                            "problem": row["problem"],
                            "prompt": prompt,
                            "target_digits": target,
                            "has_label": bool(has_label),
                            "pass_hit": bool(sample_correct),
                            "greedy_hit": bool(greedy_correct),
                            "greedy_keep_first_z_hit": bool(keep_first_correct),
                            "greedy_keep_last_z_hit": bool(keep_last_correct),
                            "greedy_reverse_z_hit": bool(reverse_correct),
                            "greedy_random_z_hit": bool(random_correct),
                            "has_answer_before_kmax": bool(row_has_answer),
                            "sampled_candidates": sampled_candidates,
                            "greedy": {
                                "phaseA_token_ids": greedy_phase_a_ids_row,
                                "phaseA_text": greedy_phase_a_text_row,
                                "phaseA_has_answer": bool(greedy_has_answer),
                                "phaseA_z_len": int(greedy_phase_a_z_len_row),
                                "z_ids": greedy_phase_a_z_ids_row,
                                "phaseB_token_ids": greedy_phase_b_ids_row if greedy_has_answer else None,
                                "phaseB_text": greedy_phase_b_text_row if greedy_has_answer else "",
                                "full_sequence": prompt + greedy_phase_a_text_row + greedy_phase_b_text_row,
                                "corruptions": greedy_corruptions,
                            },
                        }
                    )
                    + "\n"
                )
    finally:
        if writer is not None:
            writer.close()

    total = int(labeled_items)
    if total <= 0:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    mean_z = float(sum(z_lens) / len(z_lens)) if z_lens else 0.0
    return EvalMetrics(
        pass_at_n=float(pass_hits / total),
        greedy_exact_match=float(greedy_hits / total),
        greedy_keep_first_z_exact_match=float(greedy_keep_first_z_hits / total),
        greedy_keep_last_z_exact_match=float(greedy_keep_last_z_hits / total),
        greedy_reverse_z_exact_match=float(greedy_reverse_z_hits / total),
        greedy_random_z_exact_match=float(greedy_random_z_hits / total),
        mean_z_len=mean_z,
        no_answer_before_kmax_rate=float(no_answer / total),
    )
