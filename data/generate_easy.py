#!/usr/bin/env python3
"""
Generate a synthetic "very easy" math word-problem dataset where:
- Final answer is ALWAYS an integer in {1..9}
- Most numbers appearing in the question are < 100
- Some questions include larger numbers up to 1,000,000

Outputs JSONL with fields:
{"id": "...", "question": "...", "answer": <int>, "template": "...", "meta": {...}}

Example:
python gen_easy_1to9_dataset.py --n 50000 --out dataset.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple


MAX_BIG_DEFAULT = 1_000_000


@dataclass(frozen=True)
class Config:
    n: int
    out: str
    seed: int
    p_small: float
    small_max: int
    big_max: int
    ensure_unique: bool
    max_tries_per_row: int


def sample_k(rng: random.Random, cfg: Config) -> Tuple[int, str]:
    """
    Sample a "noise" integer k.
    Most of the time it's < small_max, sometimes up to big_max.
    Returns: (k, size_bucket)
    """
    if rng.random() < cfg.p_small:
        k = rng.randint(1, cfg.small_max)
        return k, "small"
    k = rng.randint(cfg.small_max + 1, cfg.big_max)
    return k, "big"


def render_template(
    rng: random.Random,
    y: int,
    k1: int,
    k2: int,
    k3: int,
) -> Tuple[str, str, Dict]:
    """
    Return (question_text, template_name, meta)
    """
    templates = [
        "add_sub_direct",
        "sub_from_sum",
        "mul_div_direct",
        "div_mul_direct",
        "two_step_cancel",
        "counting_add",
        "counting_remove",
    ]
    t = rng.choice(templates)

    if t == "add_sub_direct":
        # (y + k1) - k1 = y
        a = y + k1
        q = f"What is {a} minus {k1}?"
        meta = {"a": a, "b": k1}
        return q, t, meta

    if t == "sub_from_sum":
        # (k1 + y) - k1 = y, phrased as difference
        a = k1 + y
        q = f"What is the difference between {a} and {k1}?"
        meta = {"a": a, "b": k1}
        return q, t, meta

    if t == "mul_div_direct":
        # (y * k1) / k1 = y
        a = y * k1
        q = f"What is {a} divided by {k1}?"
        meta = {"a": a, "b": k1}
        return q, t, meta

    if t == "div_mul_direct":
        # (y * k1) / y = k1  -> not allowed (answer not 1..9)
        # Instead: (y * k1) / k1 = y but with 'quotient' wording
        a = y * k1
        q = f"What is the quotient of {a} and {k1}?"
        meta = {"a": a, "b": k1}
        return q, t, meta

    if t == "two_step_cancel":
        # ((y + k1 + k2) - k1) - k2 = y
        a = y + k1 + k2
        q = (
            f"Start with {a}. Subtract {k1}, then subtract {k2}. "
            f"What number do you get?"
        )
        meta = {"start": a, "sub1": k1, "sub2": k2}
        return q, t, meta

    if t == "counting_add":
        # "You have y candies. You get k1, then give back k1. How many now?" -> y
        # Keep it ultra simple but still includes numbers.
        q = (
            f"You have {y} marbles. You get {k1} more marbles, "
            f"and then you give away {k1} marbles. How many marbles do you have now?"
        )
        meta = {"start": y, "add": k1, "remove": k1}
        return q, t, meta

    if t == "counting_remove":
        # "You have (y + k1) stickers and give away k1." -> y
        a = y + k1
        q = f"You have {a} stickers and give away {k1} stickers. How many stickers are left?"
        meta = {"start": a, "remove": k1}
        return q, t, meta

    # Fallback (shouldn't happen)
    a = y + k1
    q = f"What is {a} minus {k1}?"
    meta = {"a": a, "b": k1}
    return q, "fallback", meta


def generate_one(rng: random.Random, cfg: Config) -> Dict:
    y = rng.randint(1, 9)

    k1, bucket1 = sample_k(rng, cfg)
    k2, bucket2 = sample_k(rng, cfg)
    k3, bucket3 = sample_k(rng, cfg)

    question, template, meta = render_template(rng, y, k1, k2, k3)

    # Extra metadata to help you analyze "mostly < 100" adherence
    meta2 = {
        "answer": y,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k1_bucket": bucket1,
        "k2_bucket": bucket2,
        "k3_bucket": bucket3,
    }
    meta2.update(meta)

    row = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": y,
        "template": template,
        "meta": meta2,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000, help="Number of examples to generate.")
    ap.add_argument("--out", type=str, default="easy_1to9.jsonl", help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--p-small",
        type=float,
        default=0.95,
        help="Probability that sampled 'k' values are < small-max (so most numbers are < 100).",
    )
    ap.add_argument(
        "--small-max",
        type=int,
        default=99,
        help="Upper bound for 'small' sampled numbers (inclusive).",
    )
    ap.add_argument(
        "--big-max",
        type=int,
        default=MAX_BIG_DEFAULT,
        help="Upper bound for 'big' sampled numbers (inclusive).",
    )
    ap.add_argument(
        "--ensure-unique",
        action="store_true",
        help="Try to avoid duplicate questions (slower, but cleaner).",
    )
    ap.add_argument(
        "--max-tries-per-row",
        type=int,
        default=50,
        help="When --ensure-unique is set, how many retries to find a new question.",
    )
    args = ap.parse_args()

    cfg = Config(
        n=args.n,
        out=args.out,
        seed=args.seed,
        p_small=args.p_small,
        small_max=args.small_max,
        big_max=args.big_max,
        ensure_unique=args.ensure_unique,
        max_tries_per_row=args.max_tries_per_row,
    )

    if not (0.0 <= cfg.p_small <= 1.0):
        raise ValueError("--p-small must be between 0 and 1.")
    if cfg.small_max < 1:
        raise ValueError("--small-max must be >= 1.")
    if cfg.big_max < cfg.small_max:
        raise ValueError("--big-max must be >= --small-max.")
    if cfg.n < 1:
        raise ValueError("--n must be >= 1.")

    rng = random.Random(cfg.seed)

    seen_questions = set()

    with open(cfg.out, "w", encoding="utf-8") as f:
        for _ in range(cfg.n):
            if not cfg.ensure_unique:
                row = generate_one(rng, cfg)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # Ensure unique question text (best-effort)
            row = None
            for _try in range(cfg.max_tries_per_row):
                candidate = generate_one(rng, cfg)
                q = candidate["question"]
                if q not in seen_questions:
                    seen_questions.add(q)
                    row = candidate
                    break

            # If we fail to find a unique question after retries, just write the last candidate
            if row is None:
                row = candidate
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Small, friendly summary
    print(f"Wrote {cfg.n} rows to: {cfg.out}")
    print(
        f"Sampling: p_small={cfg.p_small} (k<={cfg.small_max}), otherwise k in [{cfg.small_max+1}..{cfg.big_max}]"
    )
    if cfg.ensure_unique:
        print("Uniqueness: enabled (best-effort)")


if __name__ == "__main__":
    main()
