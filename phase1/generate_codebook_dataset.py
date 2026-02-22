from __future__ import annotations

import argparse
import os
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def _build_input_for_k(
    *,
    ex: Example,
    k: int,
    tokenizer,
    answer_token_id: int,
    max_positions: Optional[int],
) -> Optional[Dict]:
    answer_text = ' '.join([LATENT_TOKEN] * int(k) + [ANSWER_TOKEN])
    enc = build_prompt(ex.question, answer_text, tokenizer)
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])

    input_ids.extend(ex.digit_token_ids)
    attention_mask.extend([1] * 5)

    if max_positions is not None and len(input_ids) > max_positions:
        return None

    answer_positions = [i for i, t in enumerate(input_ids) if int(t) == int(answer_token_id)]
    if len(answer_positions) != 1:
        return None
    answer_pos = int(answer_positions[0])
    if answer_pos + 5 >= len(input_ids):
        return None

    digit_position_indices = [answer_pos + i for i in range(5)]
    digit_target_token_ids = [int(input_ids[answer_pos + 1 + i]) for i in range(5)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "digit_position_indices": digit_position_indices,
        "digit_target_token_ids": digit_target_token_ids,
    }


def _collate_cases(cases: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    bsz = len(cases)
    max_len = max(len(c["input_ids"]) for c in cases)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    digit_pos = torch.full((bsz, 5), -1, dtype=torch.long)
    digit_targets = torch.full((bsz, 5), -100, dtype=torch.long)

    for i, c in enumerate(cases):
        n = len(c["input_ids"])
        input_ids[i, :n] = torch.tensor(c["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(c["attention_mask"], dtype=torch.long)
        digit_pos[i] = torch.tensor(c["digit_position_indices"], dtype=torch.long)
        digit_targets[i] = torch.tensor(c["digit_target_token_ids"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "digit_position_indices": digit_pos,
        "digit_target_token_ids": digit_targets,
    }


def _flush_shard(rows: List[Dict], output_dir: str, shard_idx: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"part-{shard_idx:05d}.parquet")
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def _prepare_examples(batch_rows: List[Dict], digit_id_map: Dict[str, int], base_idx: int) -> List[Example]:
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
        examples.append(
            Example(
                qid=str(qid_raw),
                question=str(question),
                answer_int=int(answer_int),
                answer_digits=[int(x) for x in digits_str],
                digit_token_ids=[int(x) for x in digit_ids],
            )
        )
    return examples


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
        model.base_model.config.use_cache = False

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

    pad_id = int(tokenizer.pad_token_id)
    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)

    seen = 0
    solved = 0
    skipped_attempts = 0
    skipped_examples = 0
    shard_idx = 0
    buffer: List[Dict] = []
    k_hist = Counter()
    printed_dry = 0

    with torch.no_grad():
        for batch_id, start in enumerate(range(0, total_rows, int(args.batch_size)), start=1):
            end = min(start + int(args.batch_size), total_rows)
            batch_rows = [ds[i] for i in range(start, end)]
            seen += len(batch_rows)

            examples = _prepare_examples(batch_rows, digit_id_map, base_idx=start)
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
                    built = _build_input_for_k(
                        ex=examples[ex_idx],
                        k=k,
                        tokenizer=tokenizer,
                        answer_token_id=int(answer_token_id),
                        max_positions=max_positions,
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

                collated = _collate_cases(cases, pad_token_id=pad_id)
                input_ids = collated["input_ids"].to(device)
                attention_mask = collated["attention_mask"].to(device)
                digit_pos = collated["digit_position_indices"].to(device)
                digit_targets = collated["digit_target_token_ids"].to(device)

                amp_ctx = _autocast_ctx(device=device, dtype=load_dtype)
                with amp_ctx:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        digit_position_indices=digit_pos,
                        compute_aux=False,
                        collect_latents=True,
                    )
                if out.latent_vectors_orig is None or out.latent_vectors_orig_mask is None:
                    raise RuntimeError("collect_latents=True but latent vectors were not returned.")

                pred_digit_idx = out.logits_orig.argmax(dim=-1)
                token_table = digit_token_ids_t.unsqueeze(0).expand(pred_digit_idx.size(0), -1)
                preds_5 = torch.gather(token_table, 1, pred_digit_idx)
                correct = (preds_5 == digit_targets).all(dim=1)

                next_active: List[int] = []
                for local_idx, ex_idx in enumerate(map_local_to_ex):
                    if not bool(correct[local_idx].item()):
                        next_active.append(ex_idx)
                        continue

                    ex = examples[ex_idx]
                    latent_vecs = out.latent_vectors_orig[local_idx]
                    latent_mask = out.latent_vectors_orig_mask[local_idx]
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
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
