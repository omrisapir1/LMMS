from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

# 0=ignore(prompt/structure), 1=analysis(z), 2=analysis-end(<|end|>), 3=digit
TARGET_IGNORE = 0
TARGET_ANALYSIS = 1
TARGET_ANALYSIS_END = 2
TARGET_DIGIT = 3

HARMONY_END_TOKEN = "<|end|>"
HARMONY_RETURN_TOKEN = "<|return|>"


@dataclass
class SFTSample:
    question: str
    z_ids: List[int]
    answer_digits: List[int]


@dataclass
class HarmonyScaffold:
    # Prefix up to analysis content start (includes analysis header).
    analysis_prefix_ids: List[int]
    # Segment between analysis content end marker and final content start.
    # This includes exactly one supervised analysis-closing <|end|> and then final header structure.
    between_ids: List[int]
    # Tail after final content (e.g., <|return|> and any structural tail).
    final_suffix_ids: List[int]
    # Relative index of the analysis-closing <|end|> token inside between_ids.
    analysis_end_rel_in_between: int
    # Structural final header ids only (used by eval generation).
    final_header_ids: List[int]

    @property
    def analysis_prompt_ids(self) -> List[int]:
        return list(self.analysis_prefix_ids)


def _digits_from_answer_int(answer_int: object) -> Optional[List[int]]:
    if answer_int is None:
        return None
    try:
        value = int(answer_int)
    except Exception:
        return None
    if value < 0 or value > 99999:
        return None
    return [int(ch) for ch in f"{value:05d}"]


def _validate_digits(digits: List[int]) -> List[int]:
    if len(digits) != 5:
        raise ValueError(f"answer_digits must have length 5, got {len(digits)}")
    out: List[int] = []
    for d in digits:
        v = int(d)
        if not (0 <= v <= 9):
            raise ValueError(f"answer digit out of range [0,9]: {v}")
        out.append(v)
    return out


def _find_subsequence_once(haystack: Sequence[int], needle: Sequence[int], name: str) -> int:
    if len(needle) == 0:
        raise RuntimeError(f"empty marker subsequence for {name}")
    hits: List[int] = []
    n = len(needle)
    lim = len(haystack) - n + 1
    for i in range(max(0, lim)):
        if list(haystack[i : i + n]) == list(needle):
            hits.append(i)
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one {name} marker occurrence, got {len(hits)}"
        )
    return int(hits[0])


def resolve_digit_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    for d in "0123456789":
        tok = tokenizer.encode(d, add_special_tokens=False)
        if len(tok) != 1:
            raise RuntimeError(
                f"Digit tokenization check failed for '{d}': expected 1 token, got {tok}"
            )
        ids.append(int(tok[0]))
    return ids


def resolve_harmony_special_ids(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, tok in (
        ("analysis_end_token_id", HARMONY_END_TOKEN),
        ("return_token_id", HARMONY_RETURN_TOKEN),
    ):
        tid = int(tokenizer.convert_tokens_to_ids(tok))
        if tid < 0:
            raise RuntimeError(f"Tokenizer missing required Harmony token: {tok}")
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if enc != [tid]:
            raise RuntimeError(
                f"Harmony tokenization contract violated for {tok}: got {enc}, expected [{tid}]"
            )
        out[key] = tid
    return out


class HarmonyTemplateBuilder:
    _AN_SENT = "[[[ANALYSIS_SENTINEL_314159]]]"
    _FI_SENT = "[[[FINAL_SENTINEL_271828]]]"

    def __init__(self, *, tokenizer, system_prompt: Optional[str] = None) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = None if system_prompt is None else str(system_prompt)
        ids = resolve_harmony_special_ids(self.tokenizer)
        self.analysis_end_token_id = int(ids["analysis_end_token_id"])
        self.return_token_id = int(ids["return_token_id"])
        self._validate_contract()

    def _tokenize_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if not isinstance(ids, list) or (len(ids) > 0 and not isinstance(ids[0], int)):
            raise RuntimeError("chat template tokenization returned unexpected shape")
        return [int(x) for x in ids]

    def _render_with_sentinels(self, question: str) -> tuple[List[int], int, int, int]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": str(question)})
        messages.append(
            {
                "role": "assistant",
                "thinking": self._AN_SENT,
                "content": self._FI_SENT,
            }
        )
        ids = self._tokenize_chat(messages)
        an_ids = self.tokenizer.encode(self._AN_SENT, add_special_tokens=False)
        fi_ids = self.tokenizer.encode(self._FI_SENT, add_special_tokens=False)
        an_start = _find_subsequence_once(ids, an_ids, "analysis")
        fi_start = _find_subsequence_once(ids, fi_ids, "final")
        an_end = int(an_start + len(an_ids))
        fi_end = int(fi_start + len(fi_ids))
        if not (0 <= an_start < an_end <= fi_start < fi_end <= len(ids)):
            raise RuntimeError("invalid Harmony sentinel boundaries")
        return ids, an_start, an_end, fi_start

    def build_scaffold(self, question: str) -> HarmonyScaffold:
        ids, an_start, an_end, fi_start = self._render_with_sentinels(question)
        analysis_prefix_ids = list(ids[:an_start])
        between_ids = list(ids[an_end:fi_start])
        fi_ids = self.tokenizer.encode(self._FI_SENT, add_special_tokens=False)
        fi_end = int(fi_start + len(fi_ids))
        final_suffix_ids = list(ids[fi_end:])

        if len(analysis_prefix_ids) == 0:
            raise RuntimeError("empty analysis prefix from chat template")
        if len(between_ids) == 0:
            raise RuntimeError("empty between segment from chat template")

        end_rel = [i for i, tid in enumerate(between_ids) if int(tid) == self.analysis_end_token_id]
        if len(end_rel) != 1:
            raise RuntimeError(
                "expected exactly one analysis-closing <|end|> token between analysis and final content, "
                f"got {len(end_rel)}"
            )
        analysis_end_rel = int(end_rel[0])
        final_header_ids = list(between_ids[analysis_end_rel + 1 :])
        if len(final_header_ids) == 0:
            raise RuntimeError("empty final header segment after analysis-closing <|end|>")

        return HarmonyScaffold(
            analysis_prefix_ids=analysis_prefix_ids,
            between_ids=between_ids,
            final_suffix_ids=final_suffix_ids,
            analysis_end_rel_in_between=analysis_end_rel,
            final_header_ids=final_header_ids,
        )

    def _validate_contract(self) -> None:
        scaffold = self.build_scaffold("2 + 2 = ?")
        if scaffold.analysis_end_rel_in_between < 0:
            raise RuntimeError("Harmony scaffold validation failed: missing analysis end index")
        if scaffold.analysis_end_rel_in_between >= len(scaffold.between_ids):
            raise RuntimeError("Harmony scaffold validation failed: invalid analysis end index")
        # Prefer, but do not require, explicit <|return|> in suffix because some templates may vary.
        # If absent, training still masks all trailing structure.


class SFTDataset(Dataset):
    def __init__(
        self,
        *,
        records: Iterable[Dict],
        tokenizer,
        vocab_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.harmony = HarmonyTemplateBuilder(tokenizer=self.tokenizer)
        self.analysis_end_token_id = int(self.harmony.analysis_end_token_id)

        self.z_token_texts = [f"<z_{i}>" for i in range(self.vocab_size)]
        self.z_token_ids = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in self.z_token_texts]
        if any(tid < 0 for tid in self.z_token_ids):
            raise RuntimeError("Some Z tokens were not added to tokenizer.")

        self.digit_token_ids = resolve_digit_token_ids(self.tokenizer)
        for i in range(self.vocab_size):
            z_tok = f"<z_{i}>"
            z_enc = self.tokenizer.encode(z_tok, add_special_tokens=False)
            if z_enc != [self.z_token_ids[i]]:
                raise RuntimeError(
                    f"Tokenization contract violated for {z_tok}: got {z_enc}, "
                    f"expected [{self.z_token_ids[i]}]"
                )
        self.samples: List[SFTSample] = []

        dropped = 0
        for row in records:
            q = row.get("question")
            z = row.get("z_ids")
            d = row.get("answer_digits")
            digits: Optional[List[int]] = None
            if d is not None:
                try:
                    digits = _validate_digits([int(x) for x in d])
                except Exception:
                    digits = None
            if digits is None:
                from_int = _digits_from_answer_int(row.get("answer_int"))
                if from_int is not None:
                    digits = _validate_digits(from_int)

            if q is None or z is None or digits is None:
                dropped += 1
                continue
            z_ids = [int(x) for x in z]
            if any(x < 0 or x >= self.vocab_size for x in z_ids):
                dropped += 1
                continue
            self.samples.append(SFTSample(question=str(q), z_ids=z_ids, answer_digits=digits))

        self.stats = {"dropped": dropped, "kept": len(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        scaffold = self.harmony.build_scaffold(sample.question)
        z_token_ids = [self.z_token_ids[z] for z in sample.z_ids]
        digit_ids = [self.digit_token_ids[d] for d in sample.answer_digits]

        input_ids = (
            list(scaffold.analysis_prefix_ids)
            + list(z_token_ids)
            + list(scaffold.between_ids)
            + list(digit_ids)
            + list(scaffold.final_suffix_ids)
        )
        attention_mask = [1] * len(input_ids)

        token_class = [TARGET_IGNORE] * len(input_ids)
        analysis_start = len(scaffold.analysis_prefix_ids)
        analysis_end = analysis_start + len(z_token_ids)
        for pos in range(analysis_start, analysis_end):
            token_class[pos] = TARGET_ANALYSIS

        between_start = analysis_end
        end_abs = between_start + int(scaffold.analysis_end_rel_in_between)
        if not (0 <= end_abs < len(token_class)):
            raise RuntimeError("analysis-end absolute token index out of range")
        if int(input_ids[end_abs]) != int(self.analysis_end_token_id):
            raise RuntimeError(
                "analysis-end supervision target is not <|end|> token; template contract drifted"
            )
        token_class[end_abs] = TARGET_ANALYSIS_END

        digit_start = len(scaffold.analysis_prefix_ids) + len(z_token_ids) + len(scaffold.between_ids)
        for i in range(5):
            pos = digit_start + i
            if pos >= len(token_class):
                raise RuntimeError("digit supervision position out of range")
            token_class[pos] = TARGET_DIGIT

        target_class = [TARGET_IGNORE] * len(input_ids)
        for pos in range(len(input_ids) - 1):
            target_class[pos] = token_class[pos + 1]

        labels = [-100] * len(input_ids)
        for pos in range(len(input_ids) - 1):
            tcls = target_class[pos]
            if tcls in (TARGET_ANALYSIS, TARGET_ANALYSIS_END, TARGET_DIGIT):
                labels[pos] = int(input_ids[pos + 1])

        analysis_target_positions = [i for i, t in enumerate(target_class) if t == TARGET_ANALYSIS]
        analysis_end_target_positions = [i for i, t in enumerate(target_class) if t == TARGET_ANALYSIS_END]
        digit_target_positions = [i for i, t in enumerate(target_class) if t == TARGET_DIGIT]
        if len(analysis_end_target_positions) != 1:
            raise RuntimeError(
                f"expected exactly one supervised analysis-end position, got {len(analysis_end_target_positions)}"
            )
        if len(digit_target_positions) != 5:
            raise RuntimeError(
                f"expected exactly 5 supervised digit positions, got {len(digit_target_positions)}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_class": target_class,
            "z_len": len(z_token_ids),
            "analysis_target_positions": analysis_target_positions,
            "analysis_end_target_positions": analysis_end_target_positions,
            "digit_target_positions": digit_target_positions,
        }


class SFTCollator:
    def __init__(self, *, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.total_seen = 0
        self.total_kept = 0
        self.total_dropped_invalid_after_clip = 0
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer requires pad_token_id or eos_token_id")
            self.pad_token_id = int(self.tokenizer.eos_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        clipped: List[Dict] = []
        batch_dropped_invalid_after_clip = 0
        batch_seen = len(batch)
        for ex in batch:
            keep = min(len(ex["input_ids"]), self.max_length)
            ex2 = {
                "input_ids": ex["input_ids"][:keep],
                "attention_mask": ex["attention_mask"][:keep],
                "labels": ex["labels"][:keep],
                "target_class": ex["target_class"][:keep],
                "z_len": ex["z_len"],
                "analysis_target_positions": [p for p in ex["analysis_target_positions"] if p < keep],
                "analysis_end_target_positions": [p for p in ex["analysis_end_target_positions"] if p < keep],
                "digit_target_positions": [p for p in ex["digit_target_positions"] if p < keep],
            }
            has_one_analysis_end = len(ex2["analysis_end_target_positions"]) == 1
            has_all_digits = len(ex2["digit_target_positions"]) == 5
            if not (has_one_analysis_end and has_all_digits):
                batch_dropped_invalid_after_clip += 1
                continue
            clipped.append(ex2)

        self.total_seen += int(batch_seen)
        self.total_kept += int(len(clipped))
        self.total_dropped_invalid_after_clip += int(batch_dropped_invalid_after_clip)

        if len(clipped) == 0:
            raise RuntimeError(
                "All examples in batch were dropped after max_length clipping because supervision became invalid "
                "(required exactly one analysis-end target and exactly five digit targets). "
                f"batch_seen={batch_seen} batch_dropped_invalid_after_clip={batch_dropped_invalid_after_clip} "
                f"max_length={self.max_length} totals_seen={self.total_seen} totals_dropped_invalid_after_clip={self.total_dropped_invalid_after_clip}"
            )

        max_len = max(len(x["input_ids"]) for x in clipped)

        def pad(vals: List[int], pad_value: int) -> List[int]:
            if len(vals) < max_len:
                return vals + [pad_value] * (max_len - len(vals))
            return vals

        input_ids = torch.tensor([pad(x["input_ids"], self.pad_token_id) for x in clipped], dtype=torch.long)
        attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in clipped], dtype=torch.long)
        labels = torch.tensor([pad(x["labels"], -100) for x in clipped], dtype=torch.long)
        target_class = torch.tensor([pad(x["target_class"], TARGET_IGNORE) for x in clipped], dtype=torch.long)
        z_lens = torch.tensor([int(x["z_len"]) for x in clipped], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_class": target_class,
            "z_lens": z_lens,
            "batch_seen": int(batch_seen),
            "batch_kept": int(len(clipped)),
            "batch_dropped_invalid_after_clip": int(batch_dropped_invalid_after_clip),
            "total_seen": int(self.total_seen),
            "total_kept": int(self.total_kept),
            "total_dropped_invalid_after_clip": int(self.total_dropped_invalid_after_clip),
        }
