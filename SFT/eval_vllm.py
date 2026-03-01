from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from transformers import AutoTokenizer

from phase1.dataset import SYSTEM_PROMPT


@dataclass
class EvalMetrics:
    pass_at_n: float
    greedy_exact_match: float
    mean_z_len: float
    no_answer_before_kmax_rate: float


def _extract_answer_stats(
    *,
    token_ids: List[int],
    answer_token_id: int,
    z_token_id_set: set[int],
    digit_id_to_val: Dict[int, int],
    target_digits: List[int] | None,
) -> tuple[bool, bool, int]:
    ans_pos = -1
    for i, tid in enumerate(token_ids):
        if int(tid) == answer_token_id:
            ans_pos = i
            break

    if ans_pos < 0:
        return False, False, 0

    z_len = 0
    for t in token_ids[:ans_pos]:
        if t in z_token_id_set:
            z_len += 1

    digits = []
    for t in token_ids[ans_pos + 1 :]:
        if t in digit_id_to_val:
            digits.append(digit_id_to_val[t])
        if len(digits) == 5:
            break

    ok = len(digits) == 5 and (target_digits is not None and digits == target_digits)
    return True, ok, z_len


def _build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class GrammarLogitsProcessor:
    def __init__(
        self,
        *,
        answer_token_id: int,
        z_token_ids: Sequence[int],
        digit_token_ids: Sequence[int],
        k_max: int,
        eos_token_id: int | None,
    ) -> None:
        self.answer_token_id = int(answer_token_id)
        self.z_token_ids = set(int(x) for x in z_token_ids)
        self.digit_token_ids = set(int(x) for x in digit_token_ids)
        self.k_max = int(k_max)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)

    def __call__(self, token_ids: List[int], logits):
        answer_pos = -1
        for i, tid in enumerate(token_ids):
            if int(tid) == self.answer_token_id:
                answer_pos = i

        allowed: set[int]
        if answer_pos < 0:
            z_count = sum(1 for t in token_ids if int(t) in self.z_token_ids)
            if z_count >= self.k_max:
                allowed = {self.answer_token_id}
            else:
                allowed = set(self.z_token_ids)
                allowed.add(self.answer_token_id)
        else:
            digits_after_answer = sum(1 for t in token_ids[answer_pos + 1 :] if int(t) in self.digit_token_ids)
            if digits_after_answer >= 5:
                if self.eos_token_id is not None and self.eos_token_id >= 0:
                    allowed = {self.eos_token_id}
                else:
                    return logits
            else:
                allowed = self.digit_token_ids

        original = logits.clone() if hasattr(logits, "clone") else logits.copy()
        logits[:] = float("-inf")
        for idx in allowed:
            logits[idx] = original[idx]
        return logits


def evaluate_with_vllm(
    *,
    model_path: str,
    records: Iterable[Dict],
    pass_at_n: int,
    k_max: int,
    temperature: float,
    top_p: float,
    vocab_size: int,
    output_jsonl_path: str | None = None,
) -> EvalMetrics:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("vLLM is required for SFT evaluation and is not available.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    z_tokens = [f"<z_{i}>" for i in range(int(vocab_size))]
    z_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in z_tokens]
    if any(i < 0 for i in z_token_ids):
        raise RuntimeError("Tokenizer missing some Z tokens for evaluation.")

    answer_token_id = int(tokenizer.convert_tokens_to_ids("<ANSWER>"))
    if answer_token_id < 0:
        raise RuntimeError("Tokenizer missing <ANSWER> token for evaluation.")

    digit_token_ids = []
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Digit tokenization check failed in eval for '{d}' -> {ids}")
        digit_token_ids.append(int(ids[0]))

    llm = LLM(model=model_path, tokenizer=model_path, trust_remote_code=True,gpu_memory_utilization=0.25)
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
            continue
        digits = row.get("answer_digits")
        target_digits = None
        if digits is not None:
            try:
                d = [int(x) for x in digits]
                if len(d) == 5 and not any(x < 0 or x > 9 for x in d):
                    target_digits = d
                    labeled_items += 1
            except Exception:
                target_digits = None
        items.append({"problem": str(q), "answer_digits": target_digits})

    if not items:
        try:
            if writer is not None:
                writer.write(json.dumps({"warning": "no eval items with a problem field"}) + "\n")
        finally:
            if writer is not None:
                writer.close()
        return EvalMetrics(0.0, 0.0, 0.0, 1.0)

    prompts = [_build_prompt(tokenizer, x["problem"]) for x in items]
    processor = GrammarLogitsProcessor(
        answer_token_id=answer_token_id,
        z_token_ids=z_token_ids,
        digit_token_ids=digit_token_ids,
        k_max=int(k_max),
        eos_token_id=tokenizer.eos_token_id,
    )

    max_new_tokens = int(k_max) + 1 + 5
    sampled_params = SamplingParams(
        n=int(pass_at_n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=max_new_tokens,
        logits_processors=[processor],
    )
    greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        logits_processors=[processor],
    )

    sampled_outputs = llm.generate(prompts, sampled_params)
    greedy_outputs = llm.generate(prompts, greedy_params)

    pass_hits = 0
    greedy_hits = 0
    z_lens: List[int] = []
    no_answer = 0

    try:
        for row, prompt, sampled_out, greedy_out in zip(items, prompts, sampled_outputs, greedy_outputs):
            target = row["answer_digits"]
            has_label = target is not None
            sample_correct = False
            row_z_lens: List[int] = []
            row_has_answer = False
            sampled_generated_texts: List[str] = []
            sampled_full_sequences: List[str] = []
            sampled_token_ids: List[List[int]] = []

            for candidate in sampled_out.outputs:
                toks = [int(x) for x in candidate.token_ids]
                gen_text = tokenizer.decode(toks, skip_special_tokens=False)
                sampled_generated_texts.append(gen_text)
                sampled_full_sequences.append(prompt + gen_text)
                sampled_token_ids.append(toks)
                has_answer, ok, z_len = _extract_answer_stats(
                    token_ids=toks,
                    answer_token_id=answer_token_id,
                    z_token_id_set=z_token_id_set,
                    digit_id_to_val=digit_id_to_val,
                    target_digits=target,
                )
                if has_answer:
                    row_has_answer = True
                    row_z_lens.append(z_len)
                if ok:
                    sample_correct = True

            greedy_correct = False
            greedy_generated_text = ""
            greedy_full_sequence = ""
            greedy_token_ids: List[int] = []
            if greedy_out.outputs:
                greedy_toks = [int(x) for x in greedy_out.outputs[0].token_ids]
                greedy_token_ids = greedy_toks
                greedy_generated_text = tokenizer.decode(greedy_toks, skip_special_tokens=False)
                greedy_full_sequence = prompt + greedy_generated_text
                _, greedy_correct, _ = _extract_answer_stats(
                    token_ids=greedy_toks,
                    answer_token_id=answer_token_id,
                    z_token_id_set=z_token_id_set,
                    digit_id_to_val=digit_id_to_val,
                    target_digits=target,
                )

            if has_label:
                if sample_correct:
                    pass_hits += 1
                if greedy_correct:
                    greedy_hits += 1
                if row_z_lens:
                    z_lens.extend(row_z_lens)
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
                            "z_lens": row_z_lens,
                            "has_answer": bool(row_has_answer),
                            "sampled_generated_texts": sampled_generated_texts,
                            "sampled_full_sequences": sampled_full_sequences,
                            "sampled_token_ids": sampled_token_ids,
                            "greedy_generated_text": greedy_generated_text,
                            "greedy_full_sequence": greedy_full_sequence,
                            "greedy_token_ids": greedy_token_ids,
                        }
                    )
                    + "\n"
                )
    finally:
        if writer is not None:
            writer.close()

    total = int(labeled_items)
    if total <= 0:
        return EvalMetrics(0.0, 0.0, 0.0, 1.0)
    mean_z = float(sum(z_lens) / len(z_lens)) if z_lens else 0.0
    return EvalMetrics(
        pass_at_n=float(pass_hits / total),
        greedy_exact_match=float(greedy_hits / total),
        mean_z_len=mean_z,
        no_answer_before_kmax_rate=float(no_answer / total),
    )
