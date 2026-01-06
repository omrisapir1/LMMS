"""
Compute trailing-zero downsampling keep_prob per digit from a Hugging Face Phase 1 dataset.

Dataset schema (HF):
- question: str
- answer: str or number
- generated_answer: str

Digits are derived from `answer`:
- extract integer (prefer \\boxed{...})
- zero-pad to 5 digits (MSB-first)

Outputs JSON with fields:
- p0: observed zero rate per digit
- keep_prob: Bernoulli keep probability per digit
- p0_effective: expected zero rate after masking
- target_p0
- total_examples
- counts

Usage:
  python -m phase1.compute_keep_prob \
    --dataset YOUR_DATASET_NAME \
    --split train \
    --output data/keep_prob.json \
    --target_p0 0.30
"""

import argparse
import json
import re
from typing import List
from datasets import load_dataset


def extract_integer_from_answer(ans) -> int | None:
    """
    Extract integer from answer.
    Priority:
    1) \\boxed{...}
    2) first integer substring
    """
    if ans is None:
        return None
    s = str(ans)

    m = re.search(r"\\boxed\{\s*([+-]?\d+)\s*\}", s)
    if m:
        return int(m.group(1))

    m2 = re.search(r"([+-]?\d+)", s)
    if m2:
        return int(m2.group(1))

    return None


def compute_stats(dataset_name: str, split: str, target_p0: float):
    ds = load_dataset(dataset_name, split=split)

    total_examples = 0
    counts = {
        "per_digit_total": [0] * 5,
        "per_digit_zero": [0] * 5,
    }

    for rec in ds:
        answer = rec.get("answer")
        answer_int = extract_integer_from_answer(answer)

        if answer_int is None:
            continue
        if not (0 <= answer_int <= 99999):
            continue

        digits = [int(c) for c in f"{answer_int:05d}"]

        total_examples += 1
        for i, d in enumerate(digits):
            counts["per_digit_total"][i] += 1
            if d == 0:
                counts["per_digit_zero"][i] += 1

    # Compute p0 and keep_prob
    p0: List[float] = []
    keep_prob: List[float] = []

    for i in range(5):
        tot = counts["per_digit_total"][i]
        p = counts["per_digit_zero"][i] / tot if tot > 0 else 0.0
        p0.append(p)

        if p == 0.0:
            kp = 1.0
        else:
            kp = min(1.0, target_p0 / p)

        keep_prob.append(kp)

        if 0 < p < 1e-4:
            print(f"Warning: extremely low zero-rate for digit {i}: p0={p}")

    p0_effective = [p0[i] * keep_prob[i] for i in range(5)]

    # MSB sanity check
    if total_examples > 1000:
        if counts["per_digit_zero"][0] < counts["per_digit_zero"][4]:
            print("Warning: zero-rate pattern suggests digits may not be MSB-first")

    return {
        "p0": p0,
        "keep_prob": keep_prob,
        "p0_effective": p0_effective,
        "target_p0": target_p0,
        "total_examples": total_examples,
        "counts": counts,
        "dataset": dataset_name,
        "split": split,
        "script": "compute_keep_prob.py",
        "note": "Digits derived from `answer` field (HF-native)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="omrisap/GSM8k-Aug_qwen_62K_CoTsplitted", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--output", default='dataset', help="Path to output JSON with keep_prob")
    parser.add_argument("--target_p0", type=float, default=0.30, help="Target zero fraction per digit")
    args = parser.parse_args()

    result = compute_stats(
        dataset_name=args.dataset,
        split=args.split,
        target_p0=args.target_p0,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote keep_prob JSON to {args.output}")
    print(f"Processed {result['total_examples']} examples from {args.dataset}:{args.split}")


if __name__ == "__main__":
    main()
