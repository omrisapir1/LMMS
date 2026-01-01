#!/usr/bin/env python3
"""
Balanced 0..99999 dataset with 5 digit-head labels (d4..d0), ~10% per digit per head.

Guarantees:
- answer y is NEVER shown as a standalone number token in the question (word-boundary filter).
- templates are mathematically valid (answer is always correct).
- each question requires non-trivial computation (>=2 numbers and meaningful structure).
- uniqueness enforced (question text unique).
- digit balancing: for each head, digits 0..9 ~10% (commit-on-accept so retries don't skew).

Output JSONL fields:
{
  "id": "...",
  "question": "...",
  "answer": 12345,
  "digit_labels": {"d4":1,"d3":2,"d2":3,"d1":4,"d0":5},
  "template": "family_name",
  "meta": {...}
}

Example:
python gen_balanced_math_0to99999.py --n 250000 --out balanced.jsonl --seed 42 --ensure-unique
"""

from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
# Add strict integer sqrt checks
import math


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class Config:
    n: int
    out: str
    seed: int
    ensure_unique: bool
    max_tries_per_row: int
    # number sampling for parameters/noise
    p_small: float
    small_max: int
    big_max: int
    # progress print
    log_every: int


# -------------------------
# Utilities
# -------------------------

NAMES = ["Noa", "Liam", "Maya", "Eitan", "Ava", "Omer", "Dana", "Yoni", "Sara", "Roni", "Tal", "Ben"]
ITEMS = ["marbles", "stickers", "coins", "books", "cards", "balls", "tokens", "candies", "notebooks", "pencils"]
PLACES = ["a store", "a park", "a library", "a school", "a market", "a warehouse", "a stadium", "a workshop"]
UNITS = ["minutes", "hours", "days", "weeks"]
VEHICLES = ["bus", "train", "car", "bike", "tram"]
SPORTS = ["points", "goals", "stars", "tickets"]


def digits5(y: int) -> Dict[str, int]:
    if not (0 <= y <= 99999):
        raise ValueError(f"answer out of range: {y}")
    return {
        "d4": (y // 10000) % 10,
        "d3": (y // 1000) % 10,
        "d2": (y // 100) % 10,
        "d1": (y // 10) % 10,
        "d0": y % 10,
    }


def sample_k(rng: random.Random, cfg: Config, lo: int = 1) -> int:
    """Most of the time small, sometimes big; always >= lo."""
    if rng.random() < cfg.p_small:
        return rng.randint(max(lo, 1), max(lo, cfg.small_max))
    return rng.randint(max(lo, cfg.small_max + 1), max(lo, cfg.big_max))


def contains_answer_as_token(q: str, y: int) -> bool:
    """Reject if y appears as a standalone number token."""
    return re.search(rf"\b{re.escape(str(y))}\b", q) is not None


def count_numbers(q: str) -> int:
    return len(re.findall(r"\d+", q))


def maybe_noise_clause(rng: random.Random, cfg: Config) -> str:
    """Add harmless noise with extra numbers that don't affect the math."""
    if rng.random() < 0.85:
        return ""
    a = sample_k(rng, cfg)
    b = sample_k(rng, cfg)
    place = rng.choice(PLACES)
    # Ensure noise doesn't accidentally equal answer token; we only filter answer anyway.
    return f" (A note: at {place}, {a} and {b} are mentioned, but they are not needed.)"


def choose_phrase(rng: random.Random, options: List[str]) -> str:
    return rng.choice(options)


# Helper to extract the sqrt operand from a question string
def extract_sqrt_operand(q: str) -> Optional[int]:
    """
    Extract the integer A from patterns like:
    - "What is the square root of {A}?"
    - "Compute √{A}."
    Returns None if not found.
    """
    m = re.search(r"square root of\s*(\d+)", q, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m2 = re.search(r"√\s*(\d+)", q)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None
    return None


# -------------------------
# Balanced digit sampler (commit only after acceptance)
# -------------------------

class BalancedDigitSampler:
    """
    Samples 5 digits aiming for ~uniform 0..9 per position.
    IMPORTANT: call commit(digits) only after row is accepted.
    """

    def __init__(self, rng: random.Random, n_total: int):
        self.rng = rng
        self.n_total = n_total
        self.target = n_total / 10.0
        # head index 0..4 corresponds to d4..d0
        self.counts = {h: {d: 0 for d in range(10)} for h in range(5)}

    def _sample_digit(self, head: int) -> int:
        weights = []
        for d in range(10):
            deficit = self.target - self.counts[head][d]
            # keep positive; encourage underfilled digits
            w = max(0.5, deficit)
            weights.append(w)
        return self.rng.choices(range(10), weights=weights, k=1)[0]

    def propose(self) -> Tuple[int, List[int]]:
        d4 = self._sample_digit(0)
        d3 = self._sample_digit(1)
        d2 = self._sample_digit(2)
        d1 = self._sample_digit(3)
        d0 = self._sample_digit(4)
        y = d4 * 10000 + d3 * 1000 + d2 * 100 + d1 * 10 + d0
        return y, [d4, d3, d2, d1, d0]

    def commit(self, digits: List[int]) -> None:
        if len(digits) != 5:
            raise ValueError("digits must be length 5")
        for head, d in enumerate(digits):
            self.counts[head][d] += 1


# -------------------------
# 25 VALID template families (answer is hidden)
# Each template returns (question, template_name, meta) or None if can't safely construct.
# -------------------------

TemplateFn = Callable[[random.Random, Config, int], Optional[Tuple[str, str, Dict]]]


def build_templates() -> List[TemplateFn]:
    templates: List[TemplateFn] = []

    # 1) A - B = y
    def t_sub_direct(rng, cfg, y):
        k = sample_k(rng, cfg)
        a = y + k
        q = choose_phrase(rng, [
            f"What is {a} minus {k}?",
            f"Compute: {a} - {k}.",
            f"If you subtract {k} from {a}, what do you get?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "sub_direct", {"a": a, "b": k}

    templates.append(t_sub_direct)

    # 2) difference between (y+k) and k
    def t_diff_between(rng, cfg, y):
        k = sample_k(rng, cfg)
        a = y + k
        q = choose_phrase(rng, [
            f"What is the difference between {a} and {k}?",
            f"Find the difference: {a} and {k}.",
            f"How much larger is {a} than {k}?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "diff_between", {"a": a, "b": k}

    templates.append(t_diff_between)

    # 3) A + B = S, find A (A = y)
    def t_add_reverse(rng, cfg, y):
        k = sample_k(rng, cfg)
        s = y + k
        q = choose_phrase(rng, [
            f"A number plus {k} equals {s}. What is the number?",
            f"If x + {k} = {s}, find x.",
            f"The sum of a number and {k} is {s}. What is the number?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "add_reverse", {"k": k, "sum": s}

    templates.append(t_add_reverse)

    # 4) A - B = D, find A (A = y)
    def t_sub_reverse(rng, cfg, y):
        k = sample_k(rng, cfg)
        d = y - k
        if d < 0:
            return None
        q = choose_phrase(rng, [
            f"A number minus {k} equals {d}. What is the number?",
            f"If x - {k} = {d}, find x.",
            f"After subtracting {k} from a number you get {d}. What was the number?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "sub_reverse", {"k": k, "diff": d}

    templates.append(t_sub_reverse)

    # 5) (y*k)/k
    def t_div_direct(rng, cfg, y):
        k = sample_k(rng, cfg, lo=2)
        a = y * k
        q = choose_phrase(rng, [
            f"What is {a} divided by {k}?",
            f"Compute: {a} / {k}.",
            f"Find the quotient of {a} and {k}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "div_direct", {"a": a, "k": k}

    templates.append(t_div_direct)

    # 6) x*k = a, find x (x=y)
    def t_mul_reverse(rng, cfg, y):
        k = sample_k(rng, cfg, lo=2)
        a = y * k
        q = choose_phrase(rng, [
            f"A number multiplied by {k} equals {a}. What is the number?",
            f"If x × {k} = {a}, find x.",
            f"The product of a number and {k} is {a}. Find the number.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "mul_reverse", {"k": k, "prod": a}

    templates.append(t_mul_reverse)

    # 7) two-step subtraction forward: start - k1 - k2
    def t_two_sub_forward(rng, cfg, y):
        k1 = sample_k(rng, cfg)
        k2 = sample_k(rng, cfg)
        start = y + k1 + k2
        q = choose_phrase(rng, [
            f"Start with {start}. Subtract {k1}, then subtract {k2}. What number do you get?",
            f"You have {start}. You take away {k1} and then {k2}. What is left?",
            f"Compute {start} - {k1} - {k2}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "two_sub_forward", {"start": start, "k1": k1, "k2": k2}

    templates.append(t_two_sub_forward)

    # 8) two-step add/sub story
    def t_story_add_sub(rng, cfg, y):
        name = rng.choice(NAMES)
        item = rng.choice(ITEMS)
        add = sample_k(rng, cfg)
        sub = sample_k(rng, cfg)
        start = y - add + sub
        if start < 0:
            return None
        q = choose_phrase(rng, [
            f"{name} has {start} {item}. They get {add} more and then give away {sub}. How many {item} do they have now?",
            f"{name} starts with {start} {item}. After gaining {add} and losing {sub}, how many remain?",
            f"In total, {name} had {start} {item}, then +{add}, then -{sub}. What is the final count?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "story_add_sub", {"start": start, "add": add, "sub": sub, "item": item}

    templates.append(t_story_add_sub)

    # 9) group counting: g * p = y (always possible by choosing p=1, g=y, but we hide y)
    def t_group_count(rng, cfg, y):
        item = rng.choice(ITEMS)
        # choose p to make g reasonable and avoid revealing y directly (y must not appear; g or p can be y but then y appears -> reject by filter)
        # We'll try to find a factorization where neither g nor p equals y (to reduce rejections).
        for _ in range(50):
            p = rng.randint(2, 500) if y != 0 else rng.randint(2, 500)
            g = y // p if p != 0 else 0
            if p != 0 and y % p == 0 and g >= 1:
                if g != y and p != y:
                    break
        else:
            # fallback: g = y + k, p = 1? but p=1 makes it trivial; better skip
            return None

        q = choose_phrase(rng, [
            f"There are {g} boxes with {p} {item} in each box. How many {item} are there in total?",
            f"A stack has {p} {item}. There are {g} identical stacks. What is the total number of {item}?",
            f"{g} groups each contain {p} {item}. Find the total.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "group_count", {"groups": g, "per_group": p, "item": item}

    templates.append(t_group_count)

    # 10) equal split: total / n = y
    def t_equal_split(rng, cfg, y):
        n = rng.randint(2, 50)
        total = y * n
        q = choose_phrase(rng, [
            f"{total} items are split equally among {n} people. How many items does each person get?",
            f"If {total} is divided into {n} equal parts, what is one part?",
            f"Share {total} equally between {n} friends. How many each?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "equal_split", {"total": total, "n": n}

    templates.append(t_equal_split)

    # 11) rate * time = y (construct as y = r * t)
    def t_rate_time(rng, cfg, y):
        t = rng.randint(2, 60)
        r = y // t
        if t == 0 or r == 0 or y % t != 0:
            return None
        unit = rng.choice(UNITS)
        q = choose_phrase(rng, [
            f"A machine makes {r} items per {unit[:-1]}. How many items does it make in {t} {unit}?",
            f"At a rate of {r} per {unit[:-1]}, what is the total after {t} {unit}?",
            f"A printer produces {r} pages each {unit[:-1]}. How many pages in {t} {unit}?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "rate_time", {"rate": r, "time": t, "unit": unit}

    templates.append(t_rate_time)

    # 12) reverse mixed: (x + a) * b - c = target, solve x (x=y)
    def t_reverse_mixed1(rng, cfg, y):
        a = rng.randint(1, 300)
        b = rng.randint(2, 25)
        c = rng.randint(0, 500)
        target = (y + a) * b - c
        q = choose_phrase(rng, [
            f"Take a number, add {a}, multiply by {b}, then subtract {c}. The result is {target}. What was the original number?",
            f"After doing +{a}, ×{b}, and -{c}, you get {target}. Find the starting number.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "reverse_mixed1", {"a": a, "b": b, "c": c, "target": target}

    templates.append(t_reverse_mixed1)

    # 13) reverse mixed: (x*b + a) / c = target2, solve x
    def t_reverse_mixed2(rng, cfg, y):
        b = rng.randint(2, 30)
        c = rng.randint(2, 20)
        a = rng.randint(0, 500)
        # ensure divisibility: (y*b + a) divisible by c for clean integer target
        # We'll set target = (y*b + a) // c and require exact.
        val = y * b + a
        if val % c != 0:
            return None
        target = val // c
        q = choose_phrase(rng, [
            f"A number is multiplied by {b}, then {a} is added, and then the result is divided by {c} to get {target}. What was the original number?",
            f"Do ×{b}, then +{a}, then ÷{c}. The output is {target}. Find the input.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "reverse_mixed2", {"b": b, "a": a, "c": c, "target": target}

    templates.append(t_reverse_mixed2)

    # 14) money total: a*b + c*d = y
    def t_two_purchases(rng, cfg, y):
        # choose b,d small; solve for one of prices
        b = rng.randint(1, 30)
        d = rng.randint(1, 30)
        p1 = rng.randint(1, 500)
        part1 = p1 * b
        rest = y - part1
        if rest < 0:
            return None
        # find p2 integer with c=d? We'll set c=d and p2 = rest/d
        if d == 0 or rest % d != 0:
            return None
        p2 = rest // d
        if p2 < 0 or p2 > 5000:
            return None
        item1 = rng.choice(ITEMS)
        item2 = rng.choice(ITEMS)
        q = choose_phrase(rng, [
            f"You buy {b} {item1} for {p1} each and {d} {item2} for {p2} each. What is the total cost?",
            f"Total cost question: {b} items at {p1} each plus {d} items at {p2} each. What is the total?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "two_purchases", {"b": b, "p1": p1, "d": d, "p2": p2}

    templates.append(t_two_purchases)

    # 15) change: pay - cost = y
    def t_change(rng, cfg, y):
        cost = sample_k(rng, cfg)
        pay = cost + y
        q = choose_phrase(rng, [
            f"You pay {pay} and the cost is {cost}. How much change do you get?",
            f"An item costs {cost}. You give {pay}. What is the change?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "change", {"pay": pay, "cost": cost}

    templates.append(t_change)

    # 16) percent-of: y = p% of A (use p=25 or 50 or 20 etc), ask for p% of A
    def t_percent_of(rng, cfg, y):
        p = rng.choice([10, 20, 25, 50])
        # A = y * 100 / p
        if (y * 100) % p != 0:
            return None
        A = (y * 100) // p
        q = choose_phrase(rng, [
            f"What is {p}% of {A}?",
            f"Find {p} percent of {A}.",
            f"Compute {p}% of {A}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "percent_of", {"p": p, "A": A}

    templates.append(t_percent_of)

    # 17) remainder: A mod b = y (choose b > y)
    def t_remainder(rng, cfg, y):
        b = rng.randint(y + 1, min(y + 5000, cfg.big_max))
        q0 = rng.randint(1, 5000)
        A = q0 * b + y
        q = choose_phrase(rng, [
            f"When {A} is divided by {b}, what is the remainder?",
            f"Find the remainder when {A} ÷ {b}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "remainder", {"A": A, "b": b}

    templates.append(t_remainder)

    # 18) average: average of n numbers = y (construct numbers summing to n*y)
    def t_average(rng, cfg, y):
        n = rng.randint(2, 8)
        total = n * y
        # pick (n-1) random parts, last makes sum exact; ensure non-negative
        parts = []
        remaining = total
        for _ in range(n - 1):
            v = rng.randint(0, max(0, remaining))
            parts.append(v)
            remaining -= v
        parts.append(remaining)
        rng.shuffle(parts)
        q = choose_phrase(rng, [
            f"The average of these {n} numbers is what? {', '.join(map(str, parts))}",
            f"Find the mean of the following {n} values: {', '.join(map(str, parts))}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "average", {"n": n, "parts": parts}

    templates.append(t_average)

    # 19) arithmetic sequence nth term: a1 + (n-1)d = y
    def t_arith_nth(rng, cfg, y):
        n = rng.randint(2, 25)
        d = rng.randint(1, 500)
        a1 = y - (n - 1) * d
        if a1 < 0:
            return None
        q = choose_phrase(rng, [
            f"An arithmetic sequence starts at {a1} and increases by {d} each step. What is the {n}th term?",
            f"Sequence: first term {a1}, common difference {d}. Find term number {n}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "arith_nth", {"a1": a1, "d": d, "n": n}

    templates.append(t_arith_nth)

    # 20) square root: sqrt(A) = y  (A=y^2)
    def t_sqrt(rng, cfg, y):
        A = y * y
        # Defensive: assert perfect square invariant
        assert math.isqrt(A) ** 2 == A, "A must be a perfect square"
        # IMPORTANT: no noise for sqrt to avoid extra numbers
        q = choose_phrase(rng, [
            f"What is the square root of {A}?",
            f"Compute √{A}.",
        ])
        return q, "sqrt", {"A": A}

    templates.append(t_sqrt)

    # 21) perimeter of rectangle: 2(L+W)=y
    def t_perimeter(rng, cfg, y):
        # choose W, solve L = y/2 - W
        if y % 2 != 0:
            return None
        half = y // 2
        W = rng.randint(0, min(half, 50000))
        L = half - W
        q = choose_phrase(rng, [
            f"A rectangle has length {L} and width {W}. What is its perimeter?",
            f"Find the perimeter of a rectangle with sides {L} and {W}.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "perimeter_rect", {"L": L, "W": W}

    templates.append(t_perimeter)

    # 22) area of rectangle: L*W=y (needs factorization)
    def t_area(rng, cfg, y):
        if y == 0:
            L = rng.randint(0, 1000)
            W = 0
        else:
            # try factor pairs
            for _ in range(80):
                W = rng.randint(1, 999)
                if y % W == 0:
                    L = y // W
                    if 1 <= L <= 200000 and L != y and W != y:
                        break
            else:
                return None
        q = choose_phrase(rng, [
            f"A rectangle has length {L} and width {W}. What is its area?",
            f"Compute the area of a {L} by {W} rectangle.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "area_rect", {"L": L, "W": W}

    templates.append(t_area)

    # 23) temperature-like but pure arithmetic: score updates with three steps (forward)
    def t_score_updates(rng, cfg, y):
        a = sample_k(rng, cfg)
        b = sample_k(rng, cfg)
        c = sample_k(rng, cfg)
        start = y - a + b - c
        if start < 0:
            return None
        unit = rng.choice(SPORTS)
        q = choose_phrase(rng, [
            f"A team starts with {start} {unit}. They gain {a}, lose {b}, then gain {c}. What is their final {unit}?",
            f"Starting at {start} {unit}: +{a}, -{b}, +{c}. Final {unit}?",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "score_updates", {"start": start, "a": a, "b": b, "c": c}

    templates.append(t_score_updates)

    # 24) time conversion: total minutes from hours/minutes = y
    def t_time_convert(rng, cfg, y):
        # represent y as 60*h + m
        h = y // 60
        m = y % 60
        q = choose_phrase(rng, [
            f"A trip lasts {h} hours and {m} minutes. How many minutes is that in total?",
            f"Convert {h} hours and {m} minutes to minutes.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "time_convert", {"h": h, "m": m}

    templates.append(t_time_convert)

    # 25) distance round trip: go + return = y (choose go, return)
    def t_round_trip(rng, cfg, y):
        go = rng.randint(0, y)
        back = y - go
        veh = rng.choice(VEHICLES)
        q = choose_phrase(rng, [
            f"A {veh} travels {go} km going and {back} km returning. What is the total distance traveled?",
            f"Total distance: {go} km out and {back} km back. Find the round trip distance.",
        ]) + maybe_noise_clause(rng, cfg)
        return q, "round_trip", {"go": go, "back": back, "vehicle": veh}

    templates.append(t_round_trip)

    assert len(templates) == 25, f"expected 25 templates, got {len(templates)}"
    return templates


# -------------------------
# Generation loop
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250_000)
    ap.add_argument("--out", type=str, default="balanced_math.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ensure-unique", action="store_true")
    ap.add_argument("--max-tries-per-row", type=int, default=400)

    ap.add_argument("--p-small", type=float, default=0.90)
    ap.add_argument("--small-max", type=int, default=99)
    ap.add_argument("--big-max", type=int, default=999_999)

    ap.add_argument("--log-every", type=int, default=10_000)
    args = ap.parse_args()

    cfg = Config(
        n=args.n,
        out=args.out,
        seed=args.seed,
        ensure_unique=args.ensure_unique,
        max_tries_per_row=args.max_tries_per_row,
        p_small=args.p_small,
        small_max=args.small_max,
        big_max=args.big_max,
        log_every=args.log_every,
    )

    if not (0.0 <= cfg.p_small <= 1.0):
        raise ValueError("--p-small must be between 0 and 1")
    if cfg.small_max < 1:
        raise ValueError("--small-max must be >= 1")
    if cfg.big_max < cfg.small_max:
        raise ValueError("--big-max must be >= --small-max")
    if cfg.n < 1:
        raise ValueError("--n must be >= 1")

    rng = random.Random(cfg.seed)
    sampler = BalancedDigitSampler(rng=rng, n_total=cfg.n)
    templates = build_templates()

    seen_questions = set()

    def accept_question(q: str, y: int, tname: Optional[str] = None, meta: Optional[Dict] = None) -> bool:
        """
        Template-aware acceptance:
        - Enforce that y does not appear as a standalone token
        - Default: at least 2 numbers in the question
        - For sqrt: extract operand A and enforce A == y*y and perfect-square check; allow single-number question
        - Enforce uniqueness if requested
        """
        if contains_answer_as_token(q, y):
            return False
        # Special handling for sqrt
        if tname == "sqrt":
            A_extracted = extract_sqrt_operand(q)
            if A_extracted is None:
                return False
            if A_extracted != y * y:
                return False
            if math.isqrt(A_extracted) ** 2 != A_extracted:
                return False
            # allow single-number sqrt prompts; no need for >=2 numbers
        else:
            if count_numbers(q) < 2:
                return False
        if cfg.ensure_unique and q in seen_questions:
            return False
        return True

    wrote = 0
    with open(cfg.out, "w", encoding="utf-8") as f:
        while wrote < cfg.n:
            # propose y but do NOT commit until a row is accepted
            y, digits = sampler.propose()

            row = None
            last_candidate = None

            for _try in range(cfg.max_tries_per_row):
                tmpl = rng.choice(templates)
                out = tmpl(rng, cfg, y)
                if out is None:
                    continue
                q, tname, meta = out
                last_candidate = (q, tname, meta)

                if not accept_question(q, y, tname=tname, meta=meta):
                    continue

                # Global invariant for sqrt: ensure question operand equals answer^2
                if tname == "sqrt":
                    A_extracted = extract_sqrt_operand(q)
                    assert A_extracted is not None, "sqrt operand must be extractable"
                    assert y * y == A_extracted, "sqrt operand must equal answer^2"
                    # Also ensure meta consistency
                    assert "A" in meta and meta["A"] == A_extracted, "meta.A must equal operand"

                if cfg.ensure_unique:
                    seen_questions.add(q)

                # commit digits only now (keeps global digit-balance stable)
                sampler.commit(digits)

                row = {
                    "id": str(uuid.uuid4()),
                    "question": q,
                    "answer": y,
                    "digit_labels": digits5(y),
                    "template": tname,
                    "meta": meta,
                }

                # Final invariant check for sqrt rows before write
                if tname == "sqrt":
                    A_extracted2 = extract_sqrt_operand(q)
                    assert A_extracted2 is not None
                    assert row["answer"] * row["answer"] == A_extracted2

                break

            if row is None:
                # failed to create a safe unique question for this y; resample y (no commit happened)
                continue

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

            if cfg.log_every and wrote % cfg.log_every == 0:
                print(f"[{wrote}/{cfg.n}] unique={len(seen_questions)}")

    # Print digit balance summary
    print(f"Wrote {wrote} rows to: {cfg.out}")
    for h in range(5):
        # h=0 corresponds to d4, etc.
        print(f"Head d{4-h} counts: {sampler.counts[h]}")


if __name__ == "__main__":
    main()
