# DEBUG_GIT_CHANGE_DO_NOT_REMOVE

from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


# --------------------------------------------------------------------
# User-provided token-id lists (fill these in)
# --------------------------------------------------------------------
# start_tokens: question framing + question tokens (NO <LATENT>, NO <ANSWER>)
# end_tokens: includes exactly one <ANSWER> token id, nothing after it
START_TOKENS: List[int] = [151644, 8948, 198, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878, 1124, 79075, 46391, 151645, 198, 151644, 872, 198]  # <-- FILL ME
MID_TOKENS: List[int] = [151645, 198, 151644, 77091, 198]    # <-- FILL ME
END_TOKENS: List[int] = [151665, 151643]    # <-- FILL ME

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
    # START + Question + MID + END  (no latents here; those are added dynamically by k)
    return START_TOKENS + tokenizer.encode(problem) + MID_TOKENS + END_TOKENS


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
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--shard_size", type=int, default=2_000)
    ap.add_argument("--max_rows", type=int, default=-1, help="Debug: limit number of rows processed")
    args = ap.parse_args()

    if not START_TOKENS or not END_TOKENS:
        raise RuntimeError("You must set START_TOKENS and END_TOKENS at top of script (token-id lists).")

    ensure_dir(args.output_dir)

    # Load dataset
    ds = load_dataset(args.dataset_name, split=args.split)

    # Load tokenizer + model via loader
    from load_model_phase1 import load_phase1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, _meta = load_phase1(device=str(device))
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

    # Track solved qids to avoid reprocessing across batches
    solved_qids: set[str] = set()

    # Helper to get fields from dataset row robustly
    def get_question_and_answer(ex: Dict[str, Any]) -> Tuple[str, int]:
        # Prefer common keys, with fallbacks
        q = ex.get("question") or ex.get("problem") or ex.get("prompt")
        if not isinstance(q, str):
            # Try to coerce if it's a list of strings
            if isinstance(q, list):
                q = " ".join(map(str, q))
            else:
                q = str(q)
        ans_raw = ex.get("generated_final_answer") or ex.get("final_answer") or ex.get("answer")
        # Convert to int safely
        if isinstance(ans_raw, int):
            ans_int = ans_raw
        elif isinstance(ans_raw, str):
            ans_int = int("".join(ch for ch in ans_raw if ch.isdigit()) or 0)
        else:
            try:
                ans_int = int(ans_raw)
            except Exception:
                ans_int = 0
        ans_int = max(0, min(ans_int, 99999))
        return q, ans_int

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
            question, lab_int = get_question_and_answer(ex)

            lab_digits = int_to_5digits(lab_int)

            ids = build_input_ids(question, tokenizer)

            # sanity: ensure exactly one ANSWER_ID at end
            if ids.count(ANSWER_ID) != 1:
                raise RuntimeError("input sequence must contain exactly one <ANSWER> token id")

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
        # rolling latent history per sample (initialize when needed)
        hist_states: Dict[int, List[torch.Tensor]] = {}

        # ------------------------------------------------------------------
        # Build k=0 base: remove all existing latent tokens and keep only ... + <ANSWER>
        # ------------------------------------------------------------------
        base_ids_per_sample: List[List[int]] = []
        answer_pos_per_sample: List[int] = []  # answer position index in base sequence

        for ids in input_ids_list:
            # Remove any latent placeholders that might already exist (stage-8 style)
            ids_no_latents = [t for t in ids if t != LATENT_ID]
            # Recompute answer_idx after removal (because indices shift)
            answer_idx = ids_no_latents.index(ANSWER_ID)

            base_ids_per_sample.append(ids_no_latents)
            answer_pos_per_sample.append(answer_idx)  # position of <ANSWER> in k=0 sequence

        # Pad base batch
        max_len0 = max(len(x) for x in base_ids_per_sample)
        padded0 = [x + [pad_id] * (max_len0 - len(x)) for x in base_ids_per_sample]
        input_ids0 = torch.tensor(padded0, device=device, dtype=torch.long)
        attn0 = build_attention_mask_from_pad(input_ids0, pad_id=pad_id)

        # Get embeddings
        with torch.no_grad():
            embeds = embedding_layer(input_ids0)  # [B, T0, H]

        # We'll maintain per-sample current embeddings + attention masks + answer positions
        cur_embeds: List[torch.Tensor] = [embeds[b, :attn0[b].sum().item(), :].contiguous() for b in range(B)]
        cur_masks: List[torch.Tensor] = [torch.ones(cur_embeds[b].shape[0], device=device, dtype=torch.long) for b in range(B)]
        cur_answer_pos: List[int] = answer_pos_per_sample[:]  # shifts as we insert latents

        # ------------------------------------------------------------------
        # Evaluate k=0 and capture h0 (decision point before first latent insertion)
        # ------------------------------------------------------------------
        def run_eval_on_active(active: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
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

        h0_all: Dict[int, torch.Tensor] = {}
        for j, bi in enumerate(active_idx):
            ap0 = cur_answer_pos[bi]
            if ap0 <= 0:
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
        # For k = 1..k_max: insert one latent token before <ANSWER>, capture h_k, replace, eval
        # ------------------------------------------------------------------
        for k in range(1, args.k_max + 1):
            if not active_idx:
                break

            # Insert latent token embedding before answer for each active sample
            for bi in active_idx:
                p = cur_answer_pos[bi]  # insertion position (before answer)
                emb = cur_embeds[bi]
                H = emb.shape[1]

                latent_row = latent_token_emb.view(1, H).to(dtype=emb.dtype)
                cur_embeds[bi] = torch.cat([emb[:p], latent_row, emb[p:]], dim=0)
                cur_masks[bi] = torch.ones(cur_embeds[bi].shape[0], device=device, dtype=torch.long)

                # answer shifts right by 1
                cur_answer_pos[bi] = p + 1

            # Run forward to get hidden states for current sequences (with latent token present)
            logits_k, hidden_k = run_eval_on_active(active_idx)

            # Capture h_k at latent position and replace embedding with hidden at latent_pos-1
            h_k_captured: Dict[int, torch.Tensor] = {}
            for j, bi in enumerate(active_idx):
                latent_pos = cur_answer_pos[bi] - 1
                h_k_captured[bi] = hidden_k[j, latent_pos, :].detach()
                z = hidden_k[j, latent_pos - 1, :].detach()
                cur_embeds[bi][latent_pos, :] = z

            # Maintain rolling history
            for bi in active_idx:
                if bi not in hist_states:
                    hist_states[bi] = [h0_all[bi]]
                hist_states[bi].append(h_k_captured[bi])

            # Re-evaluate answer AFTER replacement
            logits_k2, _hidden_k2 = run_eval_on_active(active_idx)
            pred_digits = torch.argmax(logits_k2, dim=-1).tolist()

            # Early-stop solved
            active_idx_next: List[int] = []
            for j, bi in enumerate(active_idx):
                pred_int = digits_to_int(pred_digits[j])
                if pred_int == labels_int[bi]:
                    states = torch.stack(hist_states[bi], dim=0)  # [(k+1), H]
                    saved_states[bi] = states
                    saved_kstar[bi] = k
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
