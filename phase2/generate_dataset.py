#!/usr/bin/env python
# phase2/make_latent_states_dataset.py
#
# Creates a dataset for Phase-2:
# For each question, find the minimal K* in [0..k_max] for which the model predicts
# the correct integer answer (from digit-head 5 digits). Save decision-point states:
#
#   h0: hidden state at the token just before the first latent insertion point
#       (i.e., position answer_pos-1 in the k=0 sequence)
#   h1..hK*: hidden states at each <LATENT> token position at the time it existed
#            (captured BEFORE replacing that latent token embedding with hidden[p-1])
#
# Saved latent_states shape per row: [(K*+1), H] in float32
#
# Output schema (parquet shards):
# {
#   "qid": str,
#   "question": str,
#   "latent_states": list[list[float32]]  # (K*+1) x H
#   "answer_digits": list[int]           # length 5
#   "num_latents": int                   # K*
#   "k_max": int
#   "K_star": int
#   "model_id": str
# }
#
# Notes:
# - Streaming/sharded write (no full RAM accumulation).
# - Uses Phase0 forward directly + digit_heads for evaluation.
# - Uses the same latent execution rule as Phase1: latent_embed[p] = hidden[p-1].
# - Captures hi at the latent position BEFORE replacement.
#
# Usage example:
#   python phase2/make_latent_states_dataset.py \
#       --model_dir /path/to/phase1_ckpt \
#       --output_dir /path/to/out_parquet \
#       --k_max 20 --batch_size 64 --shard_size 10000 \
#       --dataset_name omrisap/LMMS_numina_250K --split train \
#       --model_id phase1_qwen25math_15b
#
# You must set start_tokens and end_tokens below (token-id lists).

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

# --------------------------------------------------------------------
# User-provided token-id lists (fill these in)
# --------------------------------------------------------------------
# start_tokens: question framing + question tokens (NO <LATENT>, NO <ANSWER>)
# end_tokens: includes exactly one <ANSWER> token id, nothing after it
START_TOKENS: List[int] = []  # <-- FILL ME
END_TOKENS: List[int] = []    # <-- FILL ME

# --------------------------------------------------------------------
# Hard IDs (you used these already; keep consistent)
# --------------------------------------------------------------------
LATENT_ID = 151666
ANSWER_ID = 151665


def int_to_5digits(n: int) -> List[int]:
    n_int = int(n)
    s = f"{n_int:05d}"
    return [int(ch) for ch in s]


def digits_to_int(digits5: Sequence[int]) -> int:
    # digits5 like [0,0,1,2,3] -> 123
    return int("".join(str(int(x)) for x in digits5))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_input_ids(problem: str, tokenizer) -> List[int]:
    # You said you will provide START_TOKENS and END_TOKENS (already token IDs).
    return START_TOKENS + tokenizer.encode(problem) + END_TOKENS


def build_attention_mask_from_pad(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (input_ids != pad_id).long()


def write_parquet_shard(
    rows: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    rows: list of dicts; latent_states must be list-of-list of float32
    """
    if not rows:
        return

    # Build Arrow arrays explicitly to avoid slow inference on large nested lists
    qid_arr = pa.array([r["qid"] for r in rows], pa.string())
    question_arr = pa.array([r["question"] for r in rows], pa.string())
    num_latents_arr = pa.array([r["num_latents"] for r in rows], pa.int32())
    kmax_arr = pa.array([r["k_max"] for r in rows], pa.int32())
    kstar_arr = pa.array([r["K_star"] for r in rows], pa.int32())
    model_id_arr = pa.array([r["model_id"] for r in rows], pa.string())

    # answer_digits: list<int32>[5]
    answer_digits_arr = pa.array([r["answer_digits"] for r in rows], pa.list_(pa.int32()))

    # latent_states: list<list<float32>>
    # Each row: (K*+1) x H
    latent_states_arr = pa.array(
        [r["latent_states"] for r in rows],
        pa.list_(pa.list_(pa.float32())),
    )

    table = pa.Table.from_arrays(
        [
            qid_arr,
            question_arr,
            latent_states_arr,
            answer_digits_arr,
            num_latents_arr,
            kmax_arr,
            kstar_arr,
            model_id_arr,
        ],
        names=[
            "qid",
            "question",
            "latent_states",
            "answer_digits",
            "num_latents",
            "k_max",
            "K_star",
            "model_id",
        ],
    )
    pq.write_table(table, out_path)


def compute_digit_logits_from_answer_positions(
    phase0_model,
    digit_heads,
    hidden_last: torch.Tensor,          # [B, T, H]
    answer_positions: torch.Tensor,     # [B]
) -> torch.Tensor:
    """
    Return logits [B, 5, 10] for the answer token positions.
    """
    bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
    answer_hidden = hidden_last[bidx, answer_positions]  # [B, H]
    logits = torch.stack([head(answer_hidden) for head in digit_heads], dim=1)
    return logits


def make_latent_token_embedding(embedding_layer, device: torch.device) -> torch.Tensor:
    """
    Return embedding vector [H] for the LATENT_ID token.
    """
    latent_id_t = torch.tensor([LATENT_ID], device=device, dtype=torch.long)
    emb = embedding_layer(latent_id_t).view(-1)  # [H]
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="omrisap/LMMS_numina_250K")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to your Phase-1 checkpoint dir or Phase-0 base dir")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--shard_size", type=int, default=10_000)
    ap.add_argument("--model_id", type=str, default="")
    ap.add_argument("--hf_tokenizer_repo", type=str, default="", help="Optional tokenizer repo/name; if empty, loads from model_dir")
    ap.add_argument("--max_rows", type=int, default=-1, help="Debug: limit number of rows processed")
    args = ap.parse_args()

    if not START_TOKENS or not END_TOKENS:
        raise RuntimeError("You must set START_TOKENS and END_TOKENS at top of script (token-id lists).")

    ensure_dir(args.output_dir)

    # Load dataset
    ds = load_dataset(args.dataset_name, split=args.split)

    # Load tokenizer
    tok_src = args.hf_tokenizer_repo if args.hf_tokenizer_repo else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    # You must have your trained Phase1 model object load here.
    # If you already have a loader in your repo, import & use it.
    #
    # Example placeholder:
    # from phase1.load_model_phase1 import load_phase1_model
    # model = load_phase1_model(args.model_dir).to(device)
    #
    # For this script, we assume you will provide `load_phase1_model` that returns Phase1CoconutModel.
    from phase1.load_model_phase1 import load_phase1_model  # <-- adjust import to your repo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_phase1_model(args.model_dir).to(device)
    model.eval()

    # Phase0 components
    phase0 = model.phase0
    base = phase0.model
    digit_heads = phase0.digit_heads
    embedding_layer = base.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or 0

    latent_token_emb = make_latent_token_embedding(embedding_layer, device=device)  # [H]

    # Streaming writer buffers
    buf: List[Dict[str, Any]] = []
    shard_idx = 0
    solved_total = 0
    seen_total = 0

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        out_path = os.path.join(args.output_dir, f"latent_states-{shard_idx:05d}.parquet")
        write_parquet_shard(buf, out_path)
        buf = []
        shard_idx += 1

    # Iterate in batches
    n_rows = len(ds)
    if args.max_rows > 0:
        n_rows = min(n_rows, args.max_rows)

    # Track solved qids to avoid reprocessing across batches (you asked for this).
    # Here qid is just the dataset index as string.
    solved_qids: set[str] = set()

    for start in range(0, n_rows, args.batch_size):
        end = min(start + args.batch_size, n_rows)

        # Build batch lists, skipping already-solved qids
        qids: List[str] = []
        questions: List[str] = []
        labels_int: List[int] = []
        labels_digits: List[List[int]] = []
        input_ids_list: List[List[int]] = []

        for i in range(start, end):
            qid = str(i)
            if qid in solved_qids:
                continue

            ex = ds[i]
            question = ex["problem"]
            final_answer = ex["final_answer"]

            # label as int (handles leading zeros by digit-compare)
            lab_int = int(final_answer)
            lab_digits = int_to_5digits(lab_int)

            ids = build_input_ids(question, tokenizer)

            # sanity: ensure exactly one ANSWER_ID at end
            if ids[-1] != ANSWER_ID:
                raise RuntimeError(f"Expected <ANSWER> as last token. Got last token {ids[-1]} for qid={qid}")
            if ids.count(ANSWER_ID) != 1:
                raise RuntimeError(f"Expected exactly one <ANSWER>. Found {ids.count(ANSWER_ID)} for qid={qid}")

            qids.append(qid)
            questions.append(question)
            labels_int.append(lab_int)
            labels_digits.append(lab_digits)
            input_ids_list.append(ids)

        if not qids:
            continue

        B = len(qids)
        seen_total += B

        # We'll maintain a dynamic "active set" inside the batch:
        active_idx = list(range(B))  # indices into qids/questions/etc
        # decision states to save per sample once solved: list of tensors [K*+1, H]
        saved_states: Dict[int, torch.Tensor] = {}
        saved_kstar: Dict[int, int] = {}

        # ------------------------------------------------------------------
        # Build k=0 base: remove all existing latent tokens and keep only ... + <ANSWER>
        # (Your get_k_thoughts_per_row logic, but for k=0)
        # ------------------------------------------------------------------
        base_ids_per_sample: List[List[int]] = []
        answer_pos_per_sample: List[int] = []  # answer position index in base sequence

        for ids in input_ids_list:
            answer_idx = ids.index(ANSWER_ID)
            prefix = [t for t in ids[:answer_idx] if t != LATENT_ID]
            base_ids = prefix + [ANSWER_ID]
            base_ids_per_sample.append(base_ids)
            answer_pos_per_sample.append(len(prefix))  # answer index

        # Pad base batch
        max_len0 = max(len(x) for x in base_ids_per_sample)
        padded0 = [x + [pad_id] * (max_len0 - len(x)) for x in base_ids_per_sample]
        input_ids0 = torch.tensor(padded0, device=device, dtype=torch.long)
        attn0 = build_attention_mask_from_pad(input_ids0, pad_id=pad_id)

        # Get embeddings
        with torch.no_grad():
            embeds = embedding_layer(input_ids0)  # [B, T0, H]

        # We'll maintain per-sample current embeddings + attention masks + answer positions
        # as Python lists of tensors because length changes as we insert latents.
        cur_embeds: List[torch.Tensor] = [embeds[b, :attn0[b].sum().item(), :].contiguous() for b in range(B)]
        cur_masks: List[torch.Tensor] = [torch.ones(cur_embeds[b].shape[0], device=device, dtype=torch.long) for b in range(B)]
        cur_answer_pos: List[int] = answer_pos_per_sample[:]  # shifts as we insert latents

        # ------------------------------------------------------------------
        # Evaluate k=0 and capture h0 (decision point before first latent insertion)
        # h0 is hidden at position (answer_pos - 1) in the k=0 sequence.
        # ------------------------------------------------------------------
        def run_eval_on_active(active: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
              logits_digits: [len(active), 5, 10]
              hidden_last: [len(active), T, H]
            """
            # pad active
            maxL = max(cur_embeds[i].shape[0] for i in active)
            H = cur_embeds[active[0]].shape[1]

            batch_emb = torch.stack([
                torch.cat([cur_embeds[i], cur_embeds[i].new_zeros(maxL - cur_embeds[i].shape[0], H)], dim=0)
                for i in active
            ], dim=0)

            batch_mask = torch.stack([
                torch.cat([cur_masks[i], cur_masks[i].new_zeros(maxL - cur_masks[i].shape[0])], dim=0)
                for i in active
            ], dim=0)

            ans_pos = torch.tensor([cur_answer_pos[i] for i in active], device=device, dtype=torch.long)

            with torch.no_grad():
                out = base(
                    inputs_embeds=batch_emb,
                    attention_mask=batch_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_last = out.hidden_states[-1]

            logits = compute_digit_logits_from_answer_positions(
                phase0_model=base,
                digit_heads=digit_heads,
                hidden_last=hidden_last,
                answer_positions=ans_pos,
            )
            return logits, hidden_last

        # Evaluate k=0 for all active
        logits0, hidden0 = run_eval_on_active(active_idx)
        pred_digits0 = torch.argmax(logits0, dim=-1).tolist()

        # Capture h0 for each sample from k=0 hidden:
        # h0 position is (answer_pos - 1). If answer_pos==0, that's invalid (shouldn't happen).
        # We'll store later only for solved ones, but we need to compute it now.
        h0_all: Dict[int, torch.Tensor] = {}
        for j, bi in enumerate(active_idx):
            ap0 = cur_answer_pos[bi]
            if ap0 <= 0:
                # This would mean answer is at position 0; not expected given your framing.
                raise RuntimeError(f"Answer position <=0 at k=0 for qid={qids[bi]}")
            h0_all[bi] = hidden0[j, ap0 - 1, :].detach()

        # Check correctness at k=0
        still_active: List[int] = []
        for j, bi in enumerate(active_idx):
            pred_int = digits_to_int(pred_digits0[j])
            if pred_int == labels_int[bi]:
                saved_states[bi] = h0_all[bi].unsqueeze(0)  # [1, H]
                saved_kstar[bi] = 0
            else:
                still_active.append(bi)
        active_idx = still_active

        # ------------------------------------------------------------------
        # For k = 1..k_max: insert one latent token before <ANSWER>,
        # capture h_k at the latent token position BEFORE replacement,
        # then replace embedding with hidden[p-1] (Phase1 rule),
        # then evaluate answer and early-stop those solved.
        # ------------------------------------------------------------------
        for k in range(1, args.k_max + 1):
            if not active_idx:
                break

            # Insert latent token embedding before answer for each active sample
            for bi in active_idx:
                p = cur_answer_pos[bi]  # insertion position (before answer)
                emb = cur_embeds[bi]
                H = emb.shape[1]

                # Insert a latent token embedding (NOT zeros)
                latent_row = latent_token_emb.view(1, H).to(dtype=emb.dtype)
                cur_embeds[bi] = torch.cat([emb[:p], latent_row, emb[p:]], dim=0)
                cur_masks[bi] = torch.ones(cur_embeds[bi].shape[0], device=device, dtype=torch.long)

                # answer shifts right by 1
                cur_answer_pos[bi] = p + 1

            # Run forward to get hidden states for current sequences (with latent token present)
            logits_k, hidden_k = run_eval_on_active(active_idx)

            # For each active sample, capture h_k at latent position (p),
            # and compute replacement z_k = hidden[p-1], then write it into embedding at p.
            # Need to know latent position p for each active sample in THIS step: it's answer_pos-1.
            # Because we inserted latent right before answer.
            # So latent position = cur_answer_pos - 1.
            h_k_captured: Dict[int, torch.Tensor] = {}
            for j, bi in enumerate(active_idx):
                latent_pos = cur_answer_pos[bi] - 1
                h_k_captured[bi] = hidden_k[j, latent_pos, :].detach()

                # Replace latent embedding with hidden at latent_pos-1
                z = hidden_k[j, latent_pos - 1, :].detach()
                # Write into cur_embeds
                cur_embeds[bi][latent_pos, :] = z

            # Re-evaluate answer AFTER replacement (because answer depends on the new latent embedding).
            logits_k2, _hidden_k2 = run_eval_on_active(active_idx)
            pred_digits = torch.argmax(logits_k2, dim=-1).tolist()

            # Early-stop solved
            still_active = []
            for j, bi in enumerate(active_idx):
                pred_int = digits_to_int(pred_digits[j])
                if pred_int == labels_int[bi]:
                    # Build states: [h0, h1..hk]
                    # h0 from k=0; h1..hk captured during each step; we have only hk now.
                    # So we must accumulate hk across steps.
                    #
                    # We'll store per-sample rolling list in a dict of tensors.
                    # Initialize if first time we reach here:
                    if bi not in saved_states:
                        # not solved earlier; start with h0
                        rolling = [h0_all[bi]]
                    else:
                        # shouldn't happen: if bi in saved_states it's already solved
                        rolling = [h0_all[bi]]

                    # To avoid keeping per-step history for all actives, we reconstruct by
                    # storing rolling history in a dict during the loop:
                    pass
                else:
                    still_active.append(bi)

            # We need to actually maintain rolling h1..hK for active samples.
            # We'll do it via a dict: hist_states[bi] = [h0, h1, ... current]
            # Initialize it after k=0 for those still active.
            if k == 1:
                hist_states: Dict[int, List[torch.Tensor]] = {bi: [h0_all[bi]] for bi in active_idx}
            # Append hk for everyone active in this k
            for bi in active_idx:
                if bi in hist_states:
                    hist_states[bi].append(h_k_captured[bi])
                else:
                    # should not happen, but be safe
                    hist_states[bi] = [h0_all[bi], h_k_captured[bi]]

            # Now evaluate solved based on logits_k2
            active_idx_next: List[int] = []
            for j, bi in enumerate(active_idx):
                pred_int = digits_to_int(pred_digits[j])
                if pred_int == labels_int[bi]:
                    states = torch.stack(hist_states[bi], dim=0)  # [(k+1), H]
                    saved_states[bi] = states
                    saved_kstar[bi] = k
                    # remove from hist_states to free memory
                    hist_states.pop(bi, None)
                else:
                    active_idx_next.append(bi)

            active_idx = active_idx_next

        # ------------------------------------------------------------------
        # Write solved examples from this batch to buffer/shards; drop unsolved.
        # ------------------------------------------------------------------
        for bi in range(B):
            if bi not in saved_states:
                continue

            K_star = saved_kstar[bi]
            states = saved_states[bi]  # [(K*+1), H] in model dtype

            row = {
                "qid": qids[bi],
                "question": questions[bi],
                "latent_states": states.to(torch.float32).cpu().numpy().tolist(),
                "answer_digits": labels_digits[bi],
                "num_latents": int(K_star),
                "k_max": int(args.k_max),
                "K_star": int(K_star),
                "model_id": args.model_id,
            }
            buf.append(row)
            solved_qids.add(qids[bi])

            solved_total += 1

            if len(buf) >= args.shard_size:
                flush()

        # Optional: print progress occasionally
        if (start // args.batch_size) % 50 == 0:
            print(
                f"[{start:>7}/{n_rows}] "
                f"seen={seen_total} solved_total={solved_total} "
                f"buffer={len(buf)} shards={shard_idx}"
            )

    flush()
    print(f"Done. Seen={seen_total} Solved={solved_total}. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
