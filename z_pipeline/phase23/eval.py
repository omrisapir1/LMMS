from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .conf import Config
from .dataset import UnifiedDataset, collate_fn
from .loss import (
    AnswerDigitLoss,
    AnswerTokenSFTLoss,
    CounterfactualAnswerLoss,
)
from .model import UnifiedZSoftModel
from .utils import build_prompt, effective_vocab_size


def _digits_to_int(d: torch.Tensor) -> int:
    out = 0
    for x in d.tolist():
        out = out * 10 + int(x)
    return out


def _randomize_z_tokens(seqs: torch.Tensor, z_ids: torch.Tensor) -> torch.Tensor:
    if z_ids.numel() == 0:
        return seqs.clone()
    seqs_rand = seqs.clone()
    is_z = torch.isin(seqs_rand, z_ids)
    num_z = int(is_z.sum().item())
    if num_z == 0:
        return seqs_rand
    rand_idx = torch.randint(0, z_ids.numel(), size=(num_z,), device=seqs_rand.device)
    seqs_rand[is_z] = z_ids[rand_idx]
    return seqs_rand


def _digit_preds_from_sequences(
    *,
    model: UnifiedZSoftModel,
    sequences: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
) -> List[Optional[int]]:
    full_attn = (sequences != pad_token_id).long()
    answer_mask = (sequences == model.answer_token_id)
    has_answer = answer_mask.any(dim=1)
    preds: List[Optional[int]] = [None] * int(sequences.size(0))
    if not bool(has_answer.any().item()):
        return preds

    valid_idx = torch.nonzero(has_answer, as_tuple=False).squeeze(1)
    seqs_valid = sequences.index_select(0, valid_idx)
    attn_valid = full_attn.index_select(0, valid_idx)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        out = model.base(
            input_ids=seqs_valid,
            attention_mask=attn_valid,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_last = out.hidden_states[-1]
        answer_mask_valid = (seqs_valid == model.answer_token_id)
        pos = answer_mask_valid.float().argmax(dim=1)
        bidx = torch.arange(hidden_last.size(0), device=hidden_last.device)
        h = hidden_last[bidx, pos]
        digit_logits = torch.stack([head(h) for head in model.digit_heads], dim=1)
        digit_preds = digit_logits.argmax(dim=-1)

    for j, bi in enumerate(valid_idx.tolist()):
        preds[bi] = _digits_to_int(digit_preds[j])
    return preds


def decode_full_upto_answer(tokenizer, seq: torch.Tensor, answer_token_id: int) -> str:
    ids = seq.tolist()
    if answer_token_id in ids:
        ids = ids[: ids.index(answer_token_id) + 1]
    return tokenizer.decode(ids, skip_special_tokens=False)


def evaluate(
    *,
    model: UnifiedZSoftModel,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate an existing model on an existing eval loader.
    This is the function train.py calls every eval interval.
    """
    del global_step
    if cfg.loss.lambda_sft <= 0:
        raise ValueError("Phase23 requires AnswerTokenSFTLoss: set loss.lambda_sft > 0")

    model.eval()

    ans_loss_fn = AnswerDigitLoss(keep_prob=cfg.loss.keep_prob)
    sft_loss_fn = AnswerTokenSFTLoss(
        answer_token_id=model.answer_token_id,
        lambda_no_answer_on_latent=cfg.loss.lambda_no_answer_on_latent,
    )
    cf_loss_fn = CounterfactualAnswerLoss(
        permute_prob=cfg.loss.counterfactual_schedule,
        digit_temperature=cfg.loss.digit_temperature,
        seed=cfg.train.seed,
    )

    totals: Dict[str, float] = {"n": 0.0, "eff_vocab": 0.0}
    if cfg.loss.lambda_ans > 0:
        totals["ans"] = 0.0
    if cfg.loss.lambda_sft > 0:
        totals["sft"] = 0.0
        totals["sft_ce"] = 0.0
        totals["sft_no_answer_latent"] = 0.0
        totals["sft_latent_p_ans"] = 0.0
    if cfg.loss.lambda_cf > 0:
        totals["cf_det"] = 0.0
    if cfg.loss.lambda_batch > 0:
        totals["batch"] = 0.0
    if cfg.loss.lambda_consistency > 0:
        totals["consistency"] = 0.0
    totals["loss"] = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_labels = batch["digit_labels"].to(device)
            k_vals = batch["K"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gumbel_tau=cfg.model.gumbel_tau_end,
                    use_gs=False,
                    return_distributions=True,
                )

                digit_logits = out["digit_logits"]
                answer_next_logits = out["answer_next_logits"]
                latent_answer_logit = out.get("latent_answer_logit_allowed")
                latent_logsumexp = out.get("latent_logsumexp_allowed")
                latent_slot_mask = out.get("latent_slot_mask", out.get("slot_mask"))
                p_student = out.get("p_student")

                loss_ans = ans_loss_fn(digit_logits, digit_labels)
                sft_terms = sft_loss_fn(
                    answer_next_logits,
                    latent_answer_logits=latent_answer_logit,
                    latent_logsumexp=latent_logsumexp,
                    latent_slot_mask=latent_slot_mask,
                    return_details=True,
                )
                loss_sft = sft_terms["loss_total"]
                loss_sft_ce = sft_terms["loss_answer_ce"]
                loss_sft_no_answer_latent = sft_terms["loss_no_answer_latent"]
                sft_latent_p_ans = sft_terms["latent_p_ans_mean"]

                if cfg.loss.lambda_cf > 0 and p_student is not None:
                    p_student_det_idx = p_student.detach().argmax(dim=-1)
                    cf_terms = cf_loss_fn(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        p_z=None,
                        p_z_idx_det=p_student_det_idx,
                        k_vals=k_vals,
                        cf_bias_scale=0.0,
                        cf_attention_bias_strength=0.0,
                        apply_cf_answer_z_bias=False,
                        cf_mode="det",
                        return_details=True,
                    )
                    loss_cf_det = cf_terms["loss_cf"]
                else:
                    loss_cf_det = torch.zeros((), device=device)

                loss_batch = torch.zeros((), device=device)
                loss_consistency = torch.zeros((), device=device)

                total = (
                    cfg.loss.lambda_ans * loss_ans
                    + cfg.loss.lambda_sft * loss_sft
                    + cfg.loss.lambda_cf * loss_cf_det
                    + cfg.loss.lambda_batch * loss_batch
                    + cfg.loss.lambda_consistency * loss_consistency
                )

            bsz = float(input_ids.size(0))
            totals["loss"] += float(total.detach().cpu()) * bsz
            if "ans" in totals:
                totals["ans"] += float(loss_ans.detach().cpu()) * bsz
            if "sft" in totals:
                totals["sft"] += float(loss_sft.detach().cpu()) * bsz
                totals["sft_ce"] += float(loss_sft_ce.detach().cpu()) * bsz
                totals["sft_no_answer_latent"] += float(loss_sft_no_answer_latent.detach().cpu()) * bsz
                totals["sft_latent_p_ans"] += float(sft_latent_p_ans.detach().cpu()) * bsz
            if "cf_det" in totals:
                totals["cf_det"] += float(loss_cf_det.detach().cpu()) * bsz
            if "batch" in totals:
                totals["batch"] += float(loss_batch.detach().cpu()) * bsz
            if "consistency" in totals:
                totals["consistency"] += float(loss_consistency.detach().cpu()) * bsz
            if p_student is not None:
                totals["eff_vocab"] += float(effective_vocab_size(p_student.detach()).mean().cpu()) * bsz
            totals["n"] += bsz

    n = max(totals["n"], 1.0)
    metrics: Dict[str, float] = {k: v / n for k, v in totals.items() if k != "n"}
    metrics["n"] = float(totals["n"])
    return metrics


def evaluate_generate_table(
    *,
    model: UnifiedZSoftModel,
    loader: DataLoader,
    tokenizer,
    cfg: Config,
    device: torch.device,
    step: int,
    pad_token_id: int,
) -> str:
    model.eval()
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_generations_step_{step}.jsonl")

    z_ids = torch.tensor(model.z_token_ids, device=device, dtype=torch.long)
    rows_written = 0
    greedy_with_answer = 0
    sample_with_answer = 0

    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for batch in loader:
            questions = [str(x) for x in batch.get("question", [])]
            prompts = [build_prompt(tokenizer, q) for q in questions]
            tok = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            prefix_lens = mask.sum(dim=1).tolist()
            answer_digits = [_digits_to_int(x) for x in batch["digit_labels"]]

            for i in range(len(questions)):
                prefix_len = int(prefix_lens[i])
                prefix_ids = ids[i:i + 1, :prefix_len]
                prefix_mask = mask[i:i + 1, :prefix_len]
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    greedy_out = model.generate_with_digits(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        max_new_tokens=cfg.train.eval_generate_max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                    sample_out = model.generate_with_digits(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        max_new_tokens=cfg.train.eval_generate_max_new_tokens,
                        do_sample=True,
                        temperature=cfg.train.eval_generate_temperature,
                        top_p=cfg.train.eval_generate_top_p,
                        pad_token_id=pad_token_id,
                    )

                greedy_seqs = greedy_out["sequences"]
                sample_seqs = sample_out["sequences"]
                greedy_suffix = greedy_seqs[0, prefix_len:]
                sample_suffix = sample_seqs[0, prefix_len:]

                if (greedy_suffix == model.latent_token_id).any() or (sample_suffix == model.latent_token_id).any():
                    raise RuntimeError("Generated <|latent|> in suffix - generation masking is broken.")

                greedy_has_answer = bool((greedy_suffix == model.answer_token_id).any().item())
                sample_has_answer = bool((sample_suffix == model.answer_token_id).any().item())
                greedy_with_answer += int(greedy_has_answer)
                sample_with_answer += int(sample_has_answer)

                greedy_pred = _digits_to_int(greedy_out["digit_preds"][0]) if greedy_has_answer else None
                sample_pred = _digits_to_int(sample_out["digit_preds"][0]) if sample_has_answer else None

                greedy_suffix_rand = _randomize_z_tokens(greedy_suffix.unsqueeze(0), z_ids)[0]
                sample_suffix_rand = _randomize_z_tokens(sample_suffix.unsqueeze(0), z_ids)[0]
                greedy_rand = torch.cat([prefix_ids[0], greedy_suffix_rand], dim=0).unsqueeze(0)
                sample_rand = torch.cat([prefix_ids[0], sample_suffix_rand], dim=0).unsqueeze(0)

                greedy_rand_pred = _digit_preds_from_sequences(
                    model=model,
                    sequences=greedy_rand,
                    pad_token_id=pad_token_id,
                    device=device,
                )[0]
                sample_rand_pred = _digit_preds_from_sequences(
                    model=model,
                    sequences=sample_rand,
                    pad_token_id=pad_token_id,
                    device=device,
                )[0]

                greedy_text = decode_full_upto_answer(
                    tokenizer, greedy_seqs[0].detach().cpu(), model.answer_token_id
                )
                sample_text = decode_full_upto_answer(
                    tokenizer, sample_seqs[0].detach().cpu(), model.answer_token_id
                )
                greedy_rand_text = decode_full_upto_answer(
                    tokenizer, greedy_rand[0].detach().cpu(), model.answer_token_id
                )
                sample_rand_text = decode_full_upto_answer(
                    tokenizer, sample_rand[0].detach().cpu(), model.answer_token_id
                )

                row = {
                    "greedy_full_text": greedy_text,
                    "greedy_digit_pred": greedy_pred,
                    "sample_full_text": sample_text,
                    "sample_digit_pred": sample_pred,
                    "greedy_randomized_full_text": greedy_rand_text,
                    "greedy_randomized_digit_pred": greedy_rand_pred,
                    "sample_randomized_full_text": sample_rand_text,
                    "sample_randomized_digit_pred": sample_rand_pred,
                    "question": questions[i],
                    "answer_digits": answer_digits[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(
        f"eval_generate@step {step} | path={out_path} rows={rows_written} "
        f"greedy_with_answer={greedy_with_answer} sample_with_answer={sample_with_answer}"
    )
    torch.cuda.empty_cache()
    model.train()
    return out_path


def evaluate_from_config(cfg: Config) -> Dict[str, float]:
    """Standalone eval entrypoint that loads model + eval dataset from config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = UnifiedZSoftModel.from_base(
        base_model_id=cfg.model.base_model_id,
        v_z=cfg.model.v_z,
        device=device,
        torch_dtype=torch.bfloat16,
        z_prefix=cfg.model.z_prefix,
        latent_token=cfg.model.latent_token,
        answer_token=cfg.model.answer_token,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    assert len(model.z_token_ids) == int(cfg.model.v_z), "len(model.z_token_ids) must equal cfg.model.v_z"
    assert model.latent_token_id == tokenizer.convert_tokens_to_ids(
        cfg.model.latent_token
    ), "latent_token_id mismatch"
    assert model.answer_token_id == tokenizer.convert_tokens_to_ids(
        cfg.model.answer_token
    ), "answer_token_id mismatch"

    ds = UnifiedDataset(
        tokenizer=tokenizer,
        latent_token_id=model.latent_token_id,
        answer_token_id=model.answer_token_id,
        k_max=cfg.data.k_max,
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.eval_split,
        data_path=cfg.data.data_path,
        max_length=cfg.data.max_length,
    )
    assert ds.k_max == cfg.data.k_max, "eval dataset.k_max must match cfg.data.k_max"

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        drop_last=False,
    )
    return evaluate(model=model, loader=loader, cfg=cfg, device=device)


if __name__ == "__main__":
    metrics = evaluate_from_config(Config())
    print(metrics)
