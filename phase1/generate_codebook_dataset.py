from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .dataset import ANSWER_TOKEN, LATENT_TOKEN, build_prompt
    from .model import Phase1CoconutModel
except ImportError:
    from dataset import ANSWER_TOKEN, LATENT_TOKEN, build_prompt  # type: ignore
    from model import Phase1CoconutModel  # type: ignore


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("qid", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answer_int", pa.int32()),
        pa.field("answer_digits", pa.list_(pa.int32())),
        pa.field("K_star", pa.int32()),
        pa.field("k_max", pa.int32()),
        pa.field("latent_vectors", pa.list_(pa.list_(pa.float32()))),
    ]
)


@dataclass
class Example:
    qid: str
    question: str
    answer_int: int
    answer_digits: List[int]
    digit_token_ids: List[int]
    prefix_input_ids: List[int]
    prefix_attention_mask: List[int]


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _build_digit_id_map(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Digit '{d}' is not a single token for this tokenizer: got {ids}"
            )
        out[d] = int(ids[0])
    return out


def _parse_answer_int(answer) -> Optional[int]:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text.isdigit():
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    if not (0 <= value <= 99999):
        return None
    return value


def _build_base_case_for_k_legacy(
    *,
    ex: Example,
    k: int,
    tokenizer,
    answer_token_id: int,
    max_positions: Optional[int],
) -> Optional[Dict]:
    answer_text = ''.join([LATENT_TOKEN] * int(k) + [ANSWER_TOKEN])
    enc = build_prompt(ex.question, answer_text, tokenizer)
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])

    # Need room for 5 autoregressively generated digits.
    if max_positions is not None and (len(input_ids) + 5) > max_positions:
        return None

    answer_positions = [i for i, t in enumerate(input_ids) if int(t) == int(answer_token_id)]
    if len(answer_positions) != 1:
        return None
    answer_pos = int(answer_positions[0])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_pos": int(answer_pos),
    }


def _build_base_case_for_k(
    *,
    ex: Example,
    k: int,
    answer_token_id: int,
    max_positions: Optional[int],
    suffix_token_ids_by_k: Sequence[Sequence[int]],
    suffix_attention_by_k: Sequence[Sequence[int]],
) -> Optional[Dict]:
    if int(k) < 0 or int(k) >= len(suffix_token_ids_by_k):
        raise ValueError(f"k={k} is out of precomputed suffix range.")

    input_ids = list(ex.prefix_input_ids)
    input_ids.extend(int(x) for x in suffix_token_ids_by_k[int(k)])
    attention_mask = list(ex.prefix_attention_mask)
    attention_mask.extend(int(x) for x in suffix_attention_by_k[int(k)])

    # Need room for 5 autoregressively generated digits.
    if max_positions is not None and (len(input_ids) + 5) > max_positions:
        return None

    answer_positions = [i for i, t in enumerate(input_ids) if int(t) == int(answer_token_id)]
    if len(answer_positions) != 1:
        return None
    answer_pos = int(answer_positions[0])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_pos": int(answer_pos),
    }


def _rollout_digits_autoreg_legacy(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    cases: List[Dict],
    digit_token_ids: List[int],
    device: torch.device,
    autocast_dtype: torch.dtype,
    collect_latents: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    True autoregressive 5-digit rollout without teacher-forcing labels.

    Positional convention:
    - digit_t token is at target_pos = answer_pos + t (1-indexed over digits),
      i.e. positions answer_pos+1 ... answer_pos+5.
    - digit_t is predicted from label_pos = target_pos - 1 by appending one
      placeholder token per step and reading logits at that next-token slot.
    """
    if not cases:
        empty = torch.empty((0, 5), dtype=torch.long, device=device)
        return {
            "pred_digit_token_ids": empty,
            "pred_digit_values": empty,
            "latent_vectors_orig": None,
            "latent_vectors_orig_mask": None,
        }

    bsz = len(cases)
    pad_token_id = int(tokenizer.pad_token_id)
    placeholder_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_token_id)

    seq_ids: List[List[int]] = [list(c["input_ids"]) for c in cases]
    seq_attn: List[List[int]] = [list(c["attention_mask"]) for c in cases]
    answer_pos = [int(c["answer_pos"]) for c in cases]

    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    pred_token_ids = torch.full((bsz, 5), -100, dtype=torch.long, device=device)
    pred_values = torch.full((bsz, 5), -1, dtype=torch.long, device=device)

    latent_vecs_out: Optional[torch.Tensor] = None
    latent_mask_out: Optional[torch.Tensor] = None

    for t in range(5):
        for b in range(bsz):
            target_pos = int(answer_pos[b] + 1 + t)
            if len(seq_ids[b]) != target_pos:
                raise RuntimeError(
                    f"Autoregressive rollout invariant violated at step {t}: "
                    f"expected length {target_pos}, got {len(seq_ids[b])}."
                )
            seq_ids[b].append(placeholder_id)
            seq_attn[b].append(1)

        max_len = max(len(x) for x in seq_ids)
        input_ids_t = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask_t = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        digit_pos_t = torch.zeros((bsz, 5), dtype=torch.long, device=device)

        for b in range(bsz):
            n = len(seq_ids[b])
            input_ids_t[b, :n] = torch.tensor(seq_ids[b], dtype=torch.long, device=device)
            attention_mask_t[b, :n] = torch.tensor(seq_attn[b], dtype=torch.long, device=device)
            label_pos = int(answer_pos[b] + t)
            digit_pos_t[b, :] = label_pos

        amp_ctx = _autocast_ctx(device=device, dtype=autocast_dtype)
        with amp_ctx:
            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                digit_position_indices=digit_pos_t,
                compute_aux=False,
                collect_latents=bool(collect_latents and t == 0),
            )

        if collect_latents and t == 0:
            latent_vecs_out = out.latent_vectors_orig
            latent_mask_out = out.latent_vectors_orig_mask

        logits_step = out.logits_orig[:, 0, :]  # [B,10]
        pred_idx = logits_step.argmax(dim=-1)  # [B], class index 0..9
        pred_tid = digit_token_ids_t.index_select(0, pred_idx)

        pred_token_ids[:, t] = pred_tid
        pred_values[:, t] = pred_idx

        for b in range(bsz):
            seq_ids[b][-1] = int(pred_tid[b].item())

    return {
        "pred_digit_token_ids": pred_token_ids,
        "pred_digit_values": pred_values,
        "latent_vectors_orig": latent_vecs_out,
        "latent_vectors_orig_mask": latent_mask_out,
    }


def rollout_digits_autoreg(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    cases: List[Dict],
    digit_token_ids: List[int],
    device: torch.device,
    autocast_dtype: torch.dtype,
    collect_latents: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    True autoregressive 5-digit rollout without teacher-forcing labels.
    Optimized path:
    - preallocates padded tensors once per case-batch
    - updates attention mask and generated token slots in-place per step
    """
    if not cases:
        empty = torch.empty((0, 5), dtype=torch.long, device=device)
        return {
            "pred_digit_token_ids": empty,
            "pred_digit_values": empty,
            "latent_vectors_orig": None,
            "latent_vectors_orig_mask": None,
        }

    bsz = len(cases)
    pad_token_id = int(tokenizer.pad_token_id)
    placeholder_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_token_id)

    answer_pos_t = torch.tensor([int(c["answer_pos"]) for c in cases], dtype=torch.long, device=device)
    base_lens_t = torch.tensor([len(c["input_ids"]) for c in cases], dtype=torch.long, device=device)
    expected_base_lens = answer_pos_t + 1
    if bool((base_lens_t != expected_base_lens).any().item()):
        bad_idx = int((base_lens_t != expected_base_lens).nonzero(as_tuple=False)[0].item())
        raise RuntimeError(
            "Autoregressive rollout invariant violated before step 0: "
            f"expected base length {int(expected_base_lens[bad_idx].item())}, "
            f"got {int(base_lens_t[bad_idx].item())}."
        )

    final_lens_t = base_lens_t + 5
    max_final_len = int(final_lens_t.max().item())

    input_ids_t = torch.full((bsz, max_final_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask_t = torch.zeros((bsz, max_final_len), dtype=torch.long, device=device)

    for b, c in enumerate(cases):
        ids = torch.tensor(c["input_ids"], dtype=torch.long, device=device)
        n = int(ids.numel())
        input_ids_t[b, :n] = ids

        start = int(answer_pos_t[b].item()) + 1
        end = start + 5
        input_ids_t[b, start:end] = int(placeholder_id)

    step_offsets = torch.arange(5, dtype=torch.long, device=device).unsqueeze(0)
    label_positions_by_step = answer_pos_t.unsqueeze(1) + step_offsets
    target_positions_by_step = answer_pos_t.unsqueeze(1) + 1 + step_offsets
    seq_col_idx = torch.arange(max_final_len, dtype=torch.long, device=device).unsqueeze(0)
    batch_idx = torch.arange(bsz, dtype=torch.long, device=device)

    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    pred_token_ids = torch.full((bsz, 5), -100, dtype=torch.long, device=device)
    pred_values = torch.full((bsz, 5), -1, dtype=torch.long, device=device)

    latent_vecs_out: Optional[torch.Tensor] = None
    latent_mask_out: Optional[torch.Tensor] = None

    for t in range(5):
        target_pos = target_positions_by_step[:, t]
        current_lens = base_lens_t + int(t) + 1
        attention_mask_t = (seq_col_idx < current_lens.unsqueeze(1)).to(dtype=torch.long)
        label_pos = label_positions_by_step[:, t]
        digit_pos_t = label_pos.unsqueeze(1).expand(-1, 5)

        amp_ctx = _autocast_ctx(device=device, dtype=autocast_dtype)
        with amp_ctx:
            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                digit_position_indices=digit_pos_t,
                compute_aux=False,
                collect_latents=bool(collect_latents and t == 0),
            )

        if collect_latents and t == 0:
            latent_vecs_out = out.latent_vectors_orig
            latent_mask_out = out.latent_vectors_orig_mask

        logits_step = out.logits_orig[:, 0, :]  # [B,10]
        pred_idx = logits_step.argmax(dim=-1)  # [B], class index 0..9
        pred_tid = digit_token_ids_t.index_select(0, pred_idx)

        pred_token_ids[:, t] = pred_tid
        pred_values[:, t] = pred_idx
        input_ids_t[batch_idx, target_pos] = pred_tid

    return {
        "pred_digit_token_ids": pred_token_ids,
        "pred_digit_values": pred_values,
        "latent_vectors_orig": latent_vecs_out,
        "latent_vectors_orig_mask": latent_mask_out,
    }


def rollout_digits_autoreg_kv_cache(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    cases: List[Dict],
    digit_token_ids: List[int],
    device: torch.device,
    autocast_dtype: torch.dtype,
    collect_latents: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Autoregressive 5-digit rollout using batched KV cache.
    """
    if not cases:
        empty = torch.empty((0, 5), dtype=torch.long, device=device)
        return {
            "pred_digit_token_ids": empty,
            "pred_digit_values": empty,
            "latent_vectors_orig": None,
            "latent_vectors_orig_mask": None,
        }

    bsz = len(cases)
    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    pred_token_ids = torch.full((bsz, 5), -100, dtype=torch.long, device=device)
    pred_values = torch.full((bsz, 5), -1, dtype=torch.long, device=device)

    base_lens_t = torch.tensor([len(c["input_ids"]) for c in cases], dtype=torch.long, device=device)
    answer_pos_t = torch.tensor([int(c["answer_pos"]) for c in cases], dtype=torch.long, device=device)
    expected_base_lens = answer_pos_t + 1
    if bool((base_lens_t != expected_base_lens).any().item()):
        bad_idx = int((base_lens_t != expected_base_lens).nonzero(as_tuple=False)[0].item())
        raise RuntimeError(
            "KV rollout invariant violated before prefill: "
            f"expected base length {int(expected_base_lens[bad_idx].item())}, "
            f"got {int(base_lens_t[bad_idx].item())}."
        )

    max_base_len = int(base_lens_t.max().item())
    max_final_len = max_base_len + 5
    input_ids_t = torch.full((bsz, max_base_len), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
    attention_mask_t = torch.zeros((bsz, max_base_len), dtype=torch.long, device=device)
    for b, case in enumerate(cases):
        ids = torch.tensor(case["input_ids"], dtype=torch.long, device=device)
        attn = torch.tensor(case["attention_mask"], dtype=torch.long, device=device)
        n = int(ids.numel())
        input_ids_t[b, :n] = ids
        attention_mask_t[b, :n] = attn

    amp_ctx = _autocast_ctx(device=device, dtype=autocast_dtype)
    with amp_ctx:
        prefill = model.prefill_with_cache_for_rollout(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            label_positions=answer_pos_t,
            digit_token_ids=digit_token_ids,
            collect_latents=collect_latents,
        )

    logits_step = prefill["next_digit_logits"]  # [B,10]
    past_key_values = prefill["past_key_values"]
    next_position_ids = prefill["next_position_ids"]  # [B]

    latent_vecs_out = prefill["latent_vectors_orig"]
    latent_mask_out = prefill["latent_vectors_orig_mask"]
    if collect_latents and (latent_vecs_out is None or latent_mask_out is None):
        raise RuntimeError(
            "prefill_with_cache_for_rollout(collect_latents=True) "
            "did not return latent vectors."
        )

    seq_col_idx = torch.arange(max_final_len, dtype=torch.long, device=device).unsqueeze(0)
    pred_idx = logits_step.argmax(dim=-1)  # [B]
    pred_tid = digit_token_ids_t.index_select(0, pred_idx)  # [B]
    pred_token_ids[:, 0] = pred_tid
    pred_values[:, 0] = pred_idx
    current_token_ids = pred_tid

    for t in range(1, 5):
        # After consuming digit_{t-1}, effective length is base_len + t.
        current_lens = base_lens_t + int(t)
        current_attention_mask = (seq_col_idx < current_lens.unsqueeze(1)).to(dtype=torch.long)

        amp_ctx = _autocast_ctx(device=device, dtype=autocast_dtype)
        with amp_ctx:
            dec = model.decode_next_with_cache_for_rollout(
                token_ids=current_token_ids,
                attention_mask_with_new_token=current_attention_mask,
                next_position_ids=next_position_ids,
                past_key_values=past_key_values,
                digit_token_ids=digit_token_ids,
            )
        logits_step = dec["next_digit_logits"]  # [B,10]
        past_key_values = dec["past_key_values"]
        next_position_ids = dec["next_position_ids"]

        pred_idx = logits_step.argmax(dim=-1)  # [B]
        pred_tid = digit_token_ids_t.index_select(0, pred_idx)  # [B]
        pred_token_ids[:, t] = pred_tid
        pred_values[:, t] = pred_idx
        current_token_ids = pred_tid

    return {
        "pred_digit_token_ids": pred_token_ids,
        "pred_digit_values": pred_values,
        "latent_vectors_orig": latent_vecs_out,
        "latent_vectors_orig_mask": latent_mask_out,
    }


def _flush_shard(rows: List[Dict], output_dir: str, shard_idx: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"part-{shard_idx:05d}.parquet")
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def _prepare_examples(
    batch_rows: List[Dict],
    digit_id_map: Dict[str, int],
    tokenizer,
    base_idx: int,
) -> List[Example]:
    examples: List[Example] = []
    for i, row in enumerate(batch_rows):
        question = row.get("problem")
        if question is None:
            continue
        answer_int = _parse_answer_int(row.get("final_answer"))
        if answer_int is None:
            continue
        digits_str = f"{answer_int:05d}"
        digit_ids = [digit_id_map[ch] for ch in digits_str]
        qid_raw = row.get("id", base_idx + i)
        prompt_prefix = build_prompt(str(question), "", tokenizer)
        prefix_input_ids = list(prompt_prefix["input_ids"])
        prefix_attention_mask = list(prompt_prefix["attention_mask"])
        examples.append(
            Example(
                qid=str(qid_raw),
                question=str(question),
                answer_int=int(answer_int),
                answer_digits=[int(x) for x in digits_str],
                digit_token_ids=[int(x) for x in digit_ids],
                prefix_input_ids=prefix_input_ids,
                prefix_attention_mask=prefix_attention_mask,
            )
        )
    return examples


def _debug_assert_case_builder_parity(
    *,
    examples: Sequence[Example],
    tokenizer,
    answer_token_id: int,
    max_positions: Optional[int],
    suffix_token_ids_by_k: Sequence[Sequence[int]],
    suffix_attention_by_k: Sequence[Sequence[int]],
    k_max: int,
) -> None:
    if not examples:
        return

    rng = random.Random(0)
    sample_size = min(2, len(examples))
    sample_indices = rng.sample(range(len(examples)), sample_size)
    k_candidates = sorted({0, min(1, int(k_max)), min(3, int(k_max)), int(k_max)})

    for ex_idx in sample_indices:
        ex = examples[ex_idx]
        for k in k_candidates:
            old_case = _build_base_case_for_k_legacy(
                ex=ex,
                k=int(k),
                tokenizer=tokenizer,
                answer_token_id=int(answer_token_id),
                max_positions=max_positions,
            )
            new_case = _build_base_case_for_k(
                ex=ex,
                k=int(k),
                answer_token_id=int(answer_token_id),
                max_positions=max_positions,
                suffix_token_ids_by_k=suffix_token_ids_by_k,
                suffix_attention_by_k=suffix_attention_by_k,
            )
            if (old_case is None) != (new_case is None):
                raise RuntimeError(
                    f"Case-builder parity mismatch (None vs non-None) for qid={ex.qid}, k={k}."
                )
            if old_case is None or new_case is None:
                continue
            if old_case["input_ids"] != new_case["input_ids"]:
                print(f"old_case['input_ids']:\n{old_case['input_ids']}")
                print(f"new_case['input_ids']:\n{new_case['input_ids']}")
                raise RuntimeError(f"Case-builder input_ids mismatch for qid={ex.qid}, k={k}.")
            if old_case["attention_mask"] != new_case["attention_mask"]:
                raise RuntimeError(f"Case-builder attention_mask mismatch for qid={ex.qid}, k={k}.")
            if int(old_case["answer_pos"]) != int(new_case["answer_pos"]):
                raise RuntimeError(f"Case-builder answer_pos mismatch for qid={ex.qid}, k={k}.")


def _debug_assert_rollout_parity(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    cases: Sequence[Dict],
    digit_token_ids: Sequence[int],
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> None:
    if not cases:
        return

    legacy = _rollout_digits_autoreg_legacy(
        model=model,
        tokenizer=tokenizer,
        cases=list(cases),
        digit_token_ids=[int(x) for x in digit_token_ids],
        device=device,
        autocast_dtype=autocast_dtype,
        collect_latents=True,
    )
    fast = rollout_digits_autoreg(
        model=model,
        tokenizer=tokenizer,
        cases=list(cases),
        digit_token_ids=[int(x) for x in digit_token_ids],
        device=device,
        autocast_dtype=autocast_dtype,
        collect_latents=True,
    )

    if not torch.equal(legacy["pred_digit_token_ids"], fast["pred_digit_token_ids"]):
        raise RuntimeError("Rollout parity mismatch: pred_digit_token_ids.")
    if not torch.equal(legacy["pred_digit_values"], fast["pred_digit_values"]):
        raise RuntimeError("Rollout parity mismatch: pred_digit_values.")

    legacy_lat = legacy["latent_vectors_orig"]
    fast_lat = fast["latent_vectors_orig"]
    legacy_mask = legacy["latent_vectors_orig_mask"]
    fast_mask = fast["latent_vectors_orig_mask"]

    if (legacy_lat is None) != (fast_lat is None):
        raise RuntimeError("Rollout parity mismatch: latent_vectors_orig None state differs.")
    if (legacy_mask is None) != (fast_mask is None):
        raise RuntimeError("Rollout parity mismatch: latent_vectors_orig_mask None state differs.")
    if legacy_lat is not None and fast_lat is not None:
        if legacy_lat.shape != fast_lat.shape:
            raise RuntimeError("Rollout parity mismatch: latent_vectors_orig shape differs.")
        if autocast_dtype == torch.float32:
            atol = 1e-6
            rtol = 1e-5
        else:
            atol = 1e-3
            rtol = 1e-2
        if not torch.allclose(legacy_lat, fast_lat, atol=atol, rtol=rtol):
            raise RuntimeError("Rollout parity mismatch: latent_vectors_orig values differ.")
    if legacy_mask is not None and fast_mask is not None:
        if not torch.equal(legacy_mask, fast_mask):
            raise RuntimeError("Rollout parity mismatch: latent_vectors_orig_mask differs.")


def _precompute_suffix_tokens_by_k(
    *,
    tokenizer,
    k_max: int,
    latent_token_id: int,
    answer_token_id: int,
) -> tuple[List[List[int]], List[List[int]]]:
    """
    Pre-tokenize suffix text once per k using the exact legacy surface form:
    "<LATENT> ... <LATENT> <ANSWER>" (space-separated join).
    This preserves tokenizer-specific whitespace/special-token behavior.
    """
    suffix_token_ids_by_k: List[List[int]] = []
    suffix_attention_by_k: List[List[int]] = []
    for k in range(int(k_max) + 1):
        answer_text = "".join([LATENT_TOKEN] * int(k) + [ANSWER_TOKEN])
        enc = tokenizer(
            answer_text,
            add_special_tokens=False,
            padding=False,
            return_attention_mask=True,
        )
        ids = [int(x) for x in enc["input_ids"]]
        attn = [int(x) for x in enc["attention_mask"]]
        if len(ids) != len(attn):
            raise RuntimeError("Suffix tokenization produced mismatched ids/attention lengths.")
        if sum(1 for x in ids if int(x) == int(answer_token_id)) != 1:
            raise RuntimeError(
                f"Suffix tokenization must contain exactly one {ANSWER_TOKEN} token for k={k}."
            )
        if sum(1 for x in ids if int(x) == int(latent_token_id)) != int(k):
            raise RuntimeError(
                f"Suffix tokenization must contain exactly k={k} {LATENT_TOKEN} tokens."
            )
        suffix_token_ids_by_k.append(ids)
        suffix_attention_by_k.append(attn)
    return suffix_token_ids_by_k, suffix_attention_by_k


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = _resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    base_model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, torch_dtype=load_dtype)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    answer_token_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    if latent_token_id is None or int(latent_token_id) < 0:
        raise RuntimeError(f"Latent token {LATENT_TOKEN} missing from tokenizer.")
    if answer_token_id is None or int(answer_token_id) < 0:
        raise RuntimeError(f"Answer token {ANSWER_TOKEN} missing from tokenizer.")

    digit_id_map = _build_digit_id_map(tokenizer)
    digit_token_ids = [int(digit_id_map[str(d)]) for d in range(10)]

    model = Phase1CoconutModel(
        base_model=base_model,
        latent_token_id=int(latent_token_id),
        answer_token_id=int(answer_token_id),
        digit_token_ids=digit_token_ids,
    ).to(device)
    model.eval()
    if hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = bool(args.use_kv_cache)

    ds = load_dataset(args.dataset_name, split=args.split)
    total_rows = len(ds)
    if args.eval_rows_limit and args.eval_rows_limit > 0:
        total_rows = min(total_rows, int(args.eval_rows_limit))
    if args.max_rows and args.max_rows > 0:
        total_rows = min(total_rows, int(args.max_rows))
    ds = ds.select(range(total_rows))

    max_positions = getattr(model.base_model.config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and max_positions <= 0:
        max_positions = None

    suffix_token_ids_by_k, suffix_attention_by_k = _precompute_suffix_tokens_by_k(
        tokenizer=tokenizer,
        k_max=int(args.k_max),
        latent_token_id=int(latent_token_id),
        answer_token_id=int(answer_token_id),
    )

    seen = 0
    solved = 0
    skipped_attempts = 0
    skipped_examples = 0
    shard_idx = 0
    buffer: List[Dict] = []
    k_hist = Counter()
    printed_dry = 0
    printed_rollout_debug = False
    checked_case_builder_parity = False
    checked_rollout_parity = False

    with torch.no_grad():
        for batch_id, start in enumerate(range(0, total_rows, int(args.batch_size)), start=1):
            end = min(start + int(args.batch_size), total_rows)
            batch_rows = [ds[i] for i in range(start, end)]
            seen += len(batch_rows)

            examples = _prepare_examples(
                batch_rows,
                digit_id_map,
                tokenizer=tokenizer,
                base_idx=start,
            )
            if args.debug_parity_checks and not checked_case_builder_parity and examples:
                _debug_assert_case_builder_parity(
                    examples=examples,
                    tokenizer=tokenizer,
                    answer_token_id=int(answer_token_id),
                    max_positions=max_positions,
                    suffix_token_ids_by_k=suffix_token_ids_by_k,
                    suffix_attention_by_k=suffix_attention_by_k,
                    k_max=int(args.k_max),
                )
                checked_case_builder_parity = True
            active = list(range(len(examples)))
            solved_in_batch = [False] * len(examples)
            skipped_in_batch = [False] * len(examples)

            def mark_skipped_example(ex_idx: int) -> None:
                nonlocal skipped_examples
                if not solved_in_batch[ex_idx] and not skipped_in_batch[ex_idx]:
                    skipped_in_batch[ex_idx] = True
                    skipped_examples += 1

            for k in range(0, int(args.k_max) + 1):
                if not active:
                    break

                cases: List[Dict] = []
                map_local_to_ex: List[int] = []
                for ex_idx in active:
                    built = _build_base_case_for_k(
                        ex=examples[ex_idx],
                        k=k,
                        answer_token_id=int(answer_token_id),
                        max_positions=max_positions,
                        suffix_token_ids_by_k=suffix_token_ids_by_k,
                        suffix_attention_by_k=suffix_attention_by_k,
                    )
                    if built is None:
                        skipped_attempts += 1
                        mark_skipped_example(ex_idx)
                        continue
                    cases.append(built)
                    map_local_to_ex.append(ex_idx)

                if not cases:
                    active = []
                    break

                if args.debug_parity_checks and not checked_rollout_parity:
                    _debug_assert_rollout_parity(
                        model=model,
                        tokenizer=tokenizer,
                        cases=cases[: min(4, len(cases))],
                        digit_token_ids=digit_token_ids,
                        device=device,
                        autocast_dtype=load_dtype,
                    )
                    checked_rollout_parity = True

                if batch_id == 1 and not printed_rollout_debug:
                    for local_idx in range(min(2, len(cases))):
                        ex_idx = map_local_to_ex[local_idx]
                        ex = examples[ex_idx]
                        c = cases[local_idx]
                        ans_pos = int(c["answer_pos"])
                        seq = c["input_ids"]
                        tail = tokenizer.decode(seq, skip_special_tokens=False)
                        suffix_after_answer = seq[ans_pos + 1:]
                        suffix_has_true_digit = any(
                            int(tid) in set(int(x) for x in ex.digit_token_ids)
                            for tid in suffix_after_answer
                        )
                        print(
                            f"[rollout debug prompt] qid={ex.qid} k={k} "
                            f"answer_pos={ans_pos} tail={tail!r} "
                            f"suffix_len={len(suffix_after_answer)} suffix_has_true_digit={suffix_has_true_digit}"
                        )

                if args.use_kv_cache:
                    rollout = rollout_digits_autoreg_kv_cache(
                        model=model,
                        tokenizer=tokenizer,
                        cases=cases,
                        digit_token_ids=digit_token_ids,
                        device=device,
                        autocast_dtype=load_dtype,
                        collect_latents=True,
                    )
                else:
                    rollout = rollout_digits_autoreg(
                        model=model,
                        tokenizer=tokenizer,
                        cases=cases,
                        digit_token_ids=digit_token_ids,
                        device=device,
                        autocast_dtype=load_dtype,
                        collect_latents=True,
                    )
                pred_digit_token_ids = rollout["pred_digit_token_ids"]
                pred_digit_values = rollout["pred_digit_values"]
                latent_vecs_batch = rollout["latent_vectors_orig"]
                latent_mask_batch = rollout["latent_vectors_orig_mask"]

                if latent_vecs_batch is None or latent_mask_batch is None:
                    raise RuntimeError(
                        "rollout_digits_autoreg(collect_latents=True) did not return latent vectors."
                    )

                true_digit_targets = torch.tensor(
                    [examples[ex_idx].digit_token_ids for ex_idx in map_local_to_ex],
                    dtype=torch.long,
                    device=device,
                )
                correct = (pred_digit_token_ids == true_digit_targets).all(dim=1)

                if batch_id == 1 and not printed_rollout_debug:
                    for local_idx in range(min(2, pred_digit_values.size(0))):
                        ex_idx = map_local_to_ex[local_idx]
                        ex = examples[ex_idx]
                        pred_str = "".join(str(int(x)) for x in pred_digit_values[local_idx].tolist())
                        true_str = "".join(str(int(x)) for x in ex.answer_digits)
                        print(
                            f"[rollout debug pred] qid={ex.qid} k={k} pred={pred_str} true={true_str}"
                        )
                    # printed_rollout_debug = True

                next_active: List[int] = []
                for local_idx, ex_idx in enumerate(map_local_to_ex):
                    if not bool(correct[local_idx].item()):
                        next_active.append(ex_idx)
                        continue

                    ex = examples[ex_idx]
                    latent_vecs = latent_vecs_batch[local_idx]
                    latent_mask = latent_mask_batch[local_idx]
                    if int(k) > int(latent_vecs.shape[0]):
                        skipped_attempts += 1
                        mark_skipped_example(ex_idx)
                        continue
                    if not bool(latent_mask[:k].all().item()):
                        skipped_attempts += 1
                        mark_skipped_example(ex_idx)
                        continue

                    vec = latent_vecs[:k].float().cpu()
                    row = {
                        "qid": ex.qid,
                        "question": ex.question,
                        "answer_int": int(ex.answer_int),
                        "answer_digits": [int(x) for x in ex.answer_digits],
                        "K_star": int(k),
                        "k_max": int(args.k_max),
                        "latent_vectors": vec.tolist(),
                    }
                    if len(row["latent_vectors"]) != int(k):
                        raise RuntimeError("latent_vectors length mismatch with K_star.")
                    buffer.append(row)
                    solved += 1
                    solved_in_batch[ex_idx] = True
                    k_hist[int(k)] += 1

                    if args.max_rows and int(args.max_rows) <= 32 and printed_dry < 2:
                        print(
                            f"[dry-run sample] qid={row['qid']} K_star={row['K_star']} "
                            f"shape=({len(row['latent_vectors'])}, {len(row['latent_vectors'][0]) if row['latent_vectors'] else 0})"
                        )
                        printed_dry += 1

                active = next_active

            # Remaining active examples were attempted but never solved by k_max.
            for ex_idx in active:
                mark_skipped_example(ex_idx)

            while len(buffer) >= int(args.shard_size):
                chunk = buffer[: int(args.shard_size)]
                buffer = buffer[int(args.shard_size) :]
                path = _flush_shard(chunk, args.output_dir, shard_idx)
                print(f"[flush] shard={shard_idx} rows={len(chunk)} path={path}")
                shard_idx += 1

            if batch_id % 20 == 0:
                solve_rate = (float(solved) / float(seen)) if seen > 0 else 0.0
                print(
                    f"[progress] batches={batch_id} seen={seen} solved={solved} "
                    f"skipped_examples={skipped_examples} skipped_attempts={skipped_attempts} "
                    f"solve_rate={solve_rate:.4f}"
                )
            if batch_id % 100 == 0 and k_hist:
                hist_items = ", ".join(f"{k}:{k_hist[k]}" for k in sorted(k_hist.keys()))
                print(f"[K_star histogram] {hist_items}")

    if buffer:
        path = _flush_shard(buffer, args.output_dir, shard_idx)
        print(f"[flush] shard={shard_idx} rows={len(buffer)} path={path}")

    solve_rate = (float(solved) / float(seen)) if seen > 0 else 0.0
    hist_items = ", ".join(f"{k}:{k_hist[k]}" for k in sorted(k_hist.keys())) if k_hist else "(empty)"
    print(
        f"[done] seen={seen} solved={solved} "
        f"skipped_examples={skipped_examples} skipped_attempts={skipped_attempts} "
        f"solve_rate={solve_rate:.4f} histogram={hist_items}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase2 codebook dataset from Phase1 Coconut model.")
    p.add_argument("--ckpt_dir", required=True, type=str)
    p.add_argument("--dataset_name", required=True, type=str)
    p.add_argument("--split", default="train", type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--k_max", default=20, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--max_rows", default=0, type=int)
    p.add_argument("--shard_size", default=1000, type=int)
    p.add_argument("--eval_rows_limit", default=0, type=int)
    p.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Use KV-cache based autoregressive rollout for digit generation.",
    )
    p.add_argument(
        "--debug_parity_checks",
        action="store_true",
        help=(
            "Run one-time parity assertions against legacy case-builder and rollout "
            "implementations."
        ),
    )
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
