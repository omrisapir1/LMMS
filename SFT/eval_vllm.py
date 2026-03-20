from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import AutoTokenizer

from .dataset import HarmonyTemplateBuilder, resolve_digit_token_ids


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


def _resolve_eval_paths(model_path: str) -> tuple[str, str]:
    base = Path(model_path)
    if (base / "full_model").is_dir():
        resolved_model = base / "full_model"
        resolved_tokenizer = (base / "tokenizer")
        return str(resolved_model), str(resolved_tokenizer)

    if base.name == "full_model":
        parent = base.parent
        sibling_tokenizer = parent / "tokenizer"
        if sibling_tokenizer.is_dir():
            return str(base), str(sibling_tokenizer)
        return str(base), str(base)

    return str(base), str(base)


def find_first_token_pos(token_ids: List[int], token_id: int) -> int:
    for i, tid in enumerate(token_ids):
        if int(tid) == int(token_id):
            return i
    return -1


def truncate_phase_a_to_analysis_end(token_ids: List[int], analysis_end_token_id: int) -> tuple[List[int], bool]:
    pos = find_first_token_pos(token_ids, analysis_end_token_id)
    if pos < 0:
        return list(token_ids), False
    return list(token_ids[: pos + 1]), True


def _extract_phase_a_ids_and_has_end(candidate, analysis_end_token_id: int) -> tuple[List[int], bool]:
    raw_ids = [int(x) for x in getattr(candidate, "token_ids", [])]
    trunc_ids, has_end = truncate_phase_a_to_analysis_end(raw_ids, analysis_end_token_id)
    if has_end:
        return trunc_ids, True

    stop_reason = getattr(candidate, "stop_reason", None)
    finish_reason = getattr(candidate, "finish_reason", None)
    stop_matches_end = False
    if stop_reason is not None:
        try:
            stop_matches_end = int(stop_reason) == int(analysis_end_token_id)
        except Exception:
            stop_matches_end = str(stop_reason).strip() == str(int(analysis_end_token_id))

    # vLLM can stop on stop_token_ids without returning that token in token_ids.
    if stop_matches_end or str(finish_reason).strip().lower() == "stop":
        return list(raw_ids) + [int(analysis_end_token_id)], True

    return trunc_ids, False


def _parse_target_digits(row: Dict) -> List[int] | None:
    digits = row.get("answer_digits")
    if digits is not None:
        try:
            d = [int(x) for x in digits]
            if len(d) == 5 and not any(x < 0 or x > 9 for x in d):
                return d
        except Exception:
            pass

    answer_int = row.get("answer_int")
    if answer_int is not None:
        try:
            value = int(answer_int)
            if 0 <= value <= 99999:
                return [int(ch) for ch in f"{value:05d}"]
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


def _digits_from_generated_ids(token_ids: List[int], digit_id_to_val: Dict[int, int]) -> List[int]:
    out: List[int] = []
    for tid in token_ids:
        if int(tid) in digit_id_to_val:
            out.append(int(digit_id_to_val[int(tid)]))
    return out


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

    resolved_model_path, resolved_tokenizer_path = _resolve_eval_paths(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path, use_fast=True)
    debug_print_examples = 2  # Set to 0 to disable.

    z_tokens = [f"<z_{i}>" for i in range(int(vocab_size))]
    z_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in z_tokens]
    if any(i < 0 for i in z_token_ids):
        raise RuntimeError("Tokenizer missing some Z tokens for evaluation.")

    digit_token_ids = resolve_digit_token_ids(tokenizer)
    digit_id_to_val = {tid: i for i, tid in enumerate(digit_token_ids)}
    z_token_id_set = set(z_token_ids)

    harmony = HarmonyTemplateBuilder(tokenizer=tokenizer)
    analysis_end_token_id = int(harmony.analysis_end_token_id)

    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vllm_cuda_visible_devices is not None and str(vllm_cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_cuda_visible_devices)
    try:
        llm = LLM(
            model=resolved_model_path,
            tokenizer=resolved_tokenizer_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )
    finally:
        if old_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices

    writer = None
    if output_jsonl_path:
        Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        writer = open(output_jsonl_path, "a", encoding="utf-8")

    items: List[Dict] = []
    for row in records:
        q = row.get("problem")
        if q is None:
            q = row.get("question")
        if q is None:
            continue
        target_digits = _parse_target_digits(row)
        scaffold = harmony.build_scaffold(str(q))
        analysis_prompt_ids = scaffold.analysis_prompt_ids
        final_header_ids = scaffold.final_header_ids
        items.append(
            {
                "problem": str(q),
                "answer_digits": target_digits,
                "analysis_prompt_ids": list(analysis_prompt_ids),
                "analysis_prompt_text": tokenizer.decode(analysis_prompt_ids, skip_special_tokens=False),
                "final_header_ids": list(final_header_ids),
                "final_header_text": tokenizer.decode(final_header_ids, skip_special_tokens=False),
            }
        )

    if not items:
        try:
            if writer is not None:
                writer.write(json.dumps({"warning": "no eval items with a problem field"}) + "\n")
        finally:
            if writer is not None:
                writer.close()
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    analysis_prompts = [x["analysis_prompt_text"] for x in items]
    final_headers = [x["final_header_text"] for x in items]

    phase_a_sampled_params = SamplingParams(
        n=int(pass_at_n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(k_max) + 1,
        allowed_token_ids=sorted(list(z_token_ids) + [analysis_end_token_id]),
        stop_token_ids=[analysis_end_token_id],
    )
    phase_a_greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(k_max) + 1,
        allowed_token_ids=sorted(list(z_token_ids) + [analysis_end_token_id]),
        stop_token_ids=[analysis_end_token_id],
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

    sampled_phase_a_outputs = llm.generate(analysis_prompts, phase_a_sampled_params)
    greedy_phase_a_outputs = llm.generate(analysis_prompts, phase_a_greedy_params)

    if len(sampled_phase_a_outputs) != len(items) or len(greedy_phase_a_outputs) != len(items):
        raise RuntimeError("phase-A outputs length mismatch")

    sampled_phase_b_prompts: List[str] = []
    sampled_phase_b_owner: List[tuple[int, int]] = []
    sampled_phase_a_ids: List[List[List[int]]] = []
    sampled_phase_a_has_end: List[List[bool]] = []
    sampled_phase_a_z_lens: List[List[int]] = []

    for item_idx, sampled_out in enumerate(sampled_phase_a_outputs):
        row_ids: List[List[int]] = []
        row_has_end: List[bool] = []
        row_z_lens: List[int] = []

        for cand_idx, candidate in enumerate(sampled_out.outputs):
            trunc_ids, has_end = _extract_phase_a_ids_and_has_end(candidate, analysis_end_token_id)
            row_ids.append(trunc_ids)
            row_has_end.append(bool(has_end))
            if has_end:
                end_pos = find_first_token_pos(trunc_ids, analysis_end_token_id)
                z_ids = [int(tid) for tid in trunc_ids[:end_pos] if int(tid) in z_token_id_set]
                row_z_lens.append(int(len(z_ids)))
                sampled_phase_b_owner.append((item_idx, cand_idx))
                sampled_phase_b_prompts.append(
                    analysis_prompts[item_idx]
                    + tokenizer.decode(trunc_ids, skip_special_tokens=False)
                    + final_headers[item_idx]
                )
            else:
                row_z_lens.append(0)

        sampled_phase_a_ids.append(row_ids)
        sampled_phase_a_has_end.append(row_has_end)
        sampled_phase_a_z_lens.append(row_z_lens)

    sampled_phase_b_digits_by_owner: Dict[tuple[int, int], List[int]] = {}
    if sampled_phase_b_prompts:
        sampled_phase_b_outputs = llm.generate(sampled_phase_b_prompts, phase_b_sampled_params)
        if len(sampled_phase_b_outputs) != len(sampled_phase_b_prompts):
            raise RuntimeError("phase-B sampled outputs length mismatch")
        for owner, out in zip(sampled_phase_b_owner, sampled_phase_b_outputs):
            ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            sampled_phase_b_digits_by_owner[owner] = _digits_from_generated_ids(ids, digit_id_to_val)

    greedy_phase_b_prompts: List[str] = []
    greedy_phase_b_owner: List[int] = []

    greedy_phase_a_has_end: List[bool] = []
    greedy_phase_a_z_lens: List[int] = []
    greedy_phase_a_z_ids: List[List[int]] = []

    corruption_variants = ("keep_first_z", "keep_last_z", "reverse_z", "random_z")
    corrupted_phase_b_prompts: List[str] = []
    corrupted_phase_b_owner: List[tuple[int, str]] = []

    for item_idx, greedy_out in enumerate(greedy_phase_a_outputs):
        if greedy_out.outputs:
            raw_ids = [int(x) for x in greedy_out.outputs[0].token_ids]
            trunc_ids, has_end = _extract_phase_a_ids_and_has_end(greedy_out.outputs[0], analysis_end_token_id)
        else:
            raw_ids, trunc_ids, has_end = [], [], False
        greedy_phase_a_has_end.append(bool(has_end))

        if item_idx < int(debug_print_examples):
            print(f"===== EVAL DEBUG EXAMPLE {item_idx + 1} =====")
            print("PHASE A PROMPT")
            print(items[item_idx]["analysis_prompt_text"])
            print("phase_a_prompt_tokens:", tokenizer.convert_ids_to_tokens(items[item_idx]["analysis_prompt_ids"]))
            print("phase_a_prompt_ids:", items[item_idx]["analysis_prompt_ids"])
            print("PHASE A GENERATED")
            print("phase_a_generated_raw_ids:", raw_ids)
            print("phase_a_generated_truncated_ids:", trunc_ids)
            print("phase_a_generated_text:", tokenizer.decode(trunc_ids, skip_special_tokens=False))
            print("phase_a_has_analysis_end:", bool(has_end))

        if has_end:
            end_pos = find_first_token_pos(trunc_ids, analysis_end_token_id)
            z_ids = [int(tid) for tid in trunc_ids[:end_pos] if int(tid) in z_token_id_set]
            greedy_phase_a_z_ids.append(z_ids)
            greedy_phase_a_z_lens.append(int(len(z_ids)))

            greedy_phase_b_owner.append(item_idx)
            phase_b_prompt = (
                analysis_prompts[item_idx]
                + tokenizer.decode(trunc_ids, skip_special_tokens=False)
                + final_headers[item_idx]
            )
            greedy_phase_b_prompts.append(phase_b_prompt)

            if item_idx < int(debug_print_examples):
                phase_b_prompt_ids = tokenizer.encode(phase_b_prompt, add_special_tokens=False)
                print("PHASE B PROMPT")
                print(phase_b_prompt)
                print("phase_b_prompt_tokens:", tokenizer.convert_ids_to_tokens(phase_b_prompt_ids))
                print("phase_b_prompt_ids:", phase_b_prompt_ids)
                print("final_header_text:", items[item_idx]["final_header_text"])
                print("final_header_tokens:", tokenizer.convert_ids_to_tokens(items[item_idx]["final_header_ids"]))
                print("final_header_ids:", items[item_idx]["final_header_ids"])

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
                corr_ids = list(corrupted_map[variant_name]) + [analysis_end_token_id]
                corrupted_phase_b_owner.append((item_idx, variant_name))
                corrupted_phase_b_prompts.append(
                    analysis_prompts[item_idx]
                    + tokenizer.decode(corr_ids, skip_special_tokens=False)
                    + final_headers[item_idx]
                )
        else:
            greedy_phase_a_z_ids.append([])
            greedy_phase_a_z_lens.append(0)
            if item_idx < int(debug_print_examples):
                print("PHASE B PROMPT")
                print("skipped: analysis end not detected in phase A")

    greedy_phase_b_digits_by_owner: Dict[int, List[int]] = {}
    if greedy_phase_b_prompts:
        greedy_phase_b_outputs = llm.generate(greedy_phase_b_prompts, phase_b_greedy_params)
        if len(greedy_phase_b_outputs) != len(greedy_phase_b_prompts):
            raise RuntimeError("phase-B greedy outputs length mismatch")
        for owner, out in zip(greedy_phase_b_owner, greedy_phase_b_outputs):
            ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            greedy_phase_b_digits_by_owner[owner] = _digits_from_generated_ids(ids, digit_id_to_val)
            if owner < int(debug_print_examples):
                print("phase_b_pred_token_ids:", ids)
                print("phase_b_pred_text:", tokenizer.decode(ids, skip_special_tokens=False))
                print("phase_b_pred_digits:", greedy_phase_b_digits_by_owner[owner])

    corrupted_phase_b_digits_by_owner: Dict[tuple[int, str], List[int]] = {}
    if corrupted_phase_b_prompts:
        corrupted_phase_b_outputs = llm.generate(corrupted_phase_b_prompts, phase_b_greedy_params)
        if len(corrupted_phase_b_outputs) != len(corrupted_phase_b_prompts):
            raise RuntimeError("phase-B corrupted outputs length mismatch")
        for owner, out in zip(corrupted_phase_b_owner, corrupted_phase_b_outputs):
            ids = [int(x) for x in out.outputs[0].token_ids] if out.outputs else []
            corrupted_phase_b_digits_by_owner[owner] = _digits_from_generated_ids(ids, digit_id_to_val)

    pass_hits = 0
    greedy_hits = 0
    greedy_keep_first_z_hits = 0
    greedy_keep_last_z_hits = 0
    greedy_reverse_z_hits = 0
    greedy_random_z_hits = 0
    z_lens: List[int] = []
    no_answer = 0

    try:
        for i, row in enumerate(items):
            target = row["answer_digits"]
            has_label = target is not None

            sample_correct = False
            for cand_idx, phase_a_ids in enumerate(sampled_phase_a_ids[i]):
                has_end = sampled_phase_a_has_end[i][cand_idx]
                if not has_end:
                    continue
                pred_digits = sampled_phase_b_digits_by_owner.get((i, cand_idx), [])
                is_correct = bool(has_label and len(pred_digits) == 5 and pred_digits == target)
                if is_correct:
                    sample_correct = True

            if has_label and sample_correct:
                pass_hits += 1

            greedy_has_end = bool(greedy_phase_a_has_end[i])
            if greedy_has_end:
                z_lens.append(int(greedy_phase_a_z_lens[i]))
            else:
                no_answer += 1

            greedy_digits = greedy_phase_b_digits_by_owner.get(i, []) if greedy_has_end else []
            greedy_correct = bool(has_label and len(greedy_digits) == 5 and greedy_digits == target)
            if greedy_correct:
                greedy_hits += 1

            for variant_name in corruption_variants:
                corr_digits = corrupted_phase_b_digits_by_owner.get((i, variant_name), [])
                corr_correct = bool(has_label and len(corr_digits) == 5 and corr_digits == target)
                if variant_name == "keep_first_z" and corr_correct:
                    greedy_keep_first_z_hits += 1
                if variant_name == "keep_last_z" and corr_correct:
                    greedy_keep_last_z_hits += 1
                if variant_name == "reverse_z" and corr_correct:
                    greedy_reverse_z_hits += 1
                if variant_name == "random_z" and corr_correct:
                    greedy_random_z_hits += 1

            if writer is not None:
                writer.write(
                    json.dumps(
                        {
                            "problem": row["problem"],
                            "target_digits": target,
                            "greedy_has_analysis_end": greedy_has_end,
                            "greedy_z_len": int(greedy_phase_a_z_lens[i]),
                            "greedy_pred_digits": greedy_digits,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if writer is not None:
            writer.close()

    labeled = sum(1 for x in items if x["answer_digits"] is not None)
    denom = max(1, labeled)

    return EvalMetrics(
        pass_at_n=float(pass_hits / denom),
        greedy_exact_match=float(greedy_hits / denom),
        greedy_keep_first_z_exact_match=float(greedy_keep_first_z_hits / denom),
        greedy_keep_last_z_exact_match=float(greedy_keep_last_z_hits / denom),
        greedy_reverse_z_exact_match=float(greedy_reverse_z_hits / denom),
        greedy_random_z_exact_match=float(greedy_random_z_hits / denom),
        mean_z_len=float(sum(z_lens) / max(1, len(z_lens))),
        no_answer_before_kmax_rate=float(no_answer / max(1, len(items))),
    )
