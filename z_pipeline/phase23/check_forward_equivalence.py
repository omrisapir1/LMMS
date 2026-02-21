from __future__ import annotations

import argparse
from typing import List

import torch

from z_pipeline.phase23.model import UnifiedZSoftModel


def _build_batch(
    *,
    tokenizer,
    model: UnifiedZSoftModel,
    prompts: List[str],
    ks: List[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    seqs = []
    for prompt, k in zip(prompts, ks):
        p = tokenizer.encode(prompt, add_special_tokens=False)
        seq = p + [model.latent_token_id] * int(k) + [model.answer_token_id]
        seqs.append(seq)

    max_len = max(len(x) for x in seqs)
    input_ids = torch.full((len(seqs), max_len), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        n = len(seq)
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :n] = 1
    return input_ids, attention_mask


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser("Check phase23 forward equivalence (streaming vs prefix-recompute).")
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--v_z", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype.lower() == "bfloat16" else torch.float32
    bundle = UnifiedZSoftModel.from_phase1(
        phase1_dir=args.phase1_dir,
        v_z=args.v_z,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.eval()

    prompts = [
        "Compute 13 plus 9.",
        "What is 7 * 8?",
        "Return the final number for 100-1.",
    ]
    ks = [1, 3, 2]
    input_ids, attention_mask = _build_batch(tokenizer=tokenizer, model=model, prompts=prompts, ks=ks)
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        out_prefix = model._forward_prefix_recompute(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_gs=False,
            return_distributions=True,
        )
        out_stream = model._forward_streaming_with_kv_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_gs=False,
            return_distributions=True,
        )

    keys = [
        "digit_logits",
        "answer_next_logits",
        "p_student",
        "p_student_det",
        "latent_answer_logit_allowed",
        "latent_logsumexp_allowed",
    ]
    for k in keys:
        d = _max_abs(out_stream[k], out_prefix[k])
        print(f"{k}: max_abs_diff={d:.6g}")
        if d > args.tol:
            raise AssertionError(f"{k} diff {d:.6g} exceeds tol {args.tol:.6g}")

    if not torch.equal(out_stream["slot_mask"], out_prefix["slot_mask"]):
        raise AssertionError("slot_mask mismatch")
    print("Forward equivalence PASSED")


if __name__ == "__main__":
    main()
