from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SFTConfig
from .dataset import ANSWER_TOKEN, SFTCollator, SFTDataset
from .eval_vllm import evaluate_with_vllm
from .losses import compute_counterfactual_regularizer, compute_weighted_loss, extract_digit_logits
import gc

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported torch dtype string: {name}")
    return table[key]


def _log(msg: str, log_path: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts} | {msg}"
    print(line)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _ensure_sft_tokens(tokenizer, model, vocab_size: int) -> Dict[str, List[int]]:
    z_tokens = [f"<z_{i}>" for i in range(int(vocab_size))]

    existing_vocab = tokenizer.get_vocab()
    to_add = [t for t in z_tokens if t not in existing_vocab]
    if ANSWER_TOKEN not in existing_vocab:
        to_add.append(ANSWER_TOKEN)

    added = 0
    if to_add:
        added = tokenizer.add_tokens(to_add)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added += 1

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    answer_token_id = int(tokenizer.convert_tokens_to_ids(ANSWER_TOKEN))
    if answer_token_id < 0:
        raise RuntimeError("Failed to resolve <ANSWER> token id")

    z_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in z_tokens]
    if any(i < 0 for i in z_token_ids):
        raise RuntimeError("Failed to resolve one or more Z token ids")

    digit_token_ids = []
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Digit tokenization check failed for '{d}' -> {ids}")
        digit_token_ids.append(int(ids[0]))

    return {
        "z_token_ids": z_token_ids,
        "digit_token_ids": digit_token_ids,
        "answer_token_id": [answer_token_id],
    }


def _apply_warmup_freeze(
    *,
    model,
    z_token_ids: Sequence[int],
    warmup_active: bool,
) -> List[torch.utils.hooks.RemovableHandle]:
    for p in model.parameters():
        p.requires_grad = True

    if not warmup_active:
        return []

    for p in model.parameters():
        p.requires_grad = False

    embed = model.get_input_embeddings().weight
    embed.requires_grad = True

    lm_head = model.get_output_embeddings().weight
    same_param = lm_head.data_ptr() == embed.data_ptr()
    if not same_param:
        lm_head.requires_grad = True

    allowed = torch.zeros(embed.shape[0], dtype=torch.bool, device=embed.device)
    allowed[torch.as_tensor(list(z_token_ids), dtype=torch.long, device=embed.device)] = True

    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[~allowed] = 0
        return out

    handles.append(embed.register_hook(_mask_grad))
    if not same_param:
        handles.append(lm_head.register_hook(_mask_grad))

    return handles


def _make_run_dir(cfg: SFTConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}__V{cfg.vocab_size}__bs{cfg.batch_size}__lr{cfg.learning_rate}"
    run_dir = os.path.join(cfg.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "last"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def _save_model_dir(model, tokenizer, out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _retain_periodic(ckpt_root: str, keep_last_k: int) -> None:
    all_steps = []
    for name in os.listdir(ckpt_root):
        if not name.startswith("step_"):
            continue
        path = os.path.join(ckpt_root, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("_")[1])
        except Exception:
            continue
        all_steps.append((step, path))
    all_steps.sort(key=lambda x: x[0], reverse=True)
    for _, path in all_steps[int(keep_last_k) :]:
        shutil.rmtree(path)


def _save_last(
    *,
    run_dir: str,
    model,
    tokenizer,
    cfg: SFTConfig,
    step: int,
    best_pass_at_n: float,
) -> str:
    out_dir = os.path.join(run_dir, "last")
    _save_model_dir(model, tokenizer, out_dir)
    payload = {
        "step": int(step),
        "best_pass_at_n": float(best_pass_at_n),
        "config": asdict(cfg),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_dir


def _save_periodic(
    *,
    run_dir: str,
    model,
    tokenizer,
    step: int,
    keep_last_k: int,
) -> str:
    out_dir = os.path.join(run_dir, "checkpoints", f"step_{step:05d}")
    _save_model_dir(model, tokenizer, out_dir)
    _retain_periodic(os.path.join(run_dir, "checkpoints"), keep_last_k=keep_last_k)
    return out_dir


def _save_best(*, run_dir: str, model, tokenizer, step: int, metric: float) -> str:
    out_dir = os.path.join(run_dir, "checkpoints", "best")
    _save_model_dir(model, tokenizer, out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"step": int(step), "pass_at_n": float(metric)}, f, indent=2)
    return out_dir


def _save_ppo_init(*, run_dir: str, model, tokenizer) -> str:
    out_dir = os.path.join(run_dir, "ppo_init")
    _save_model_dir(model, tokenizer, out_dir)
    return out_dir


def _load_hf_records(dataset_name: str, split: str) -> List[Dict]:
    ds = load_dataset(dataset_name, split=split)
    return [dict(x) for x in ds]


def _build_loader(
    *,
    records: Iterable[Dict],
    tokenizer,
    cfg: SFTConfig,
    shuffle: bool,
    train: bool,
) -> DataLoader:
    ds = SFTDataset(records=records, tokenizer=tokenizer, vocab_size=cfg.vocab_size)
    collator = SFTCollator(tokenizer=tokenizer, max_length=cfg.max_length)
    num_workers = int(cfg.dataloader_num_workers if train else cfg.eval_dataloader_num_workers)
    loader_kwargs = {
        "batch_size": int(cfg.batch_size if train else cfg.eval_batch_size),
        "shuffle": bool(shuffle),
        "collate_fn": collator,
        "drop_last": False,
        "num_workers": max(0, num_workers),
        "pin_memory": bool(cfg.dataloader_pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(max(1, cfg.dataloader_prefetch_factor))
    return DataLoader(ds, **loader_kwargs)


def _append_metrics_csv(path: str, row: Dict[str, float]) -> None:
    exists = os.path.isfile(path)
    fields = list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_prob_tuple(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated probabilities, got: {text}")
    vals = tuple(float(x) for x in parts)
    s = vals[0] + vals[1] + vals[2]
    if min(vals) < 0.0 or abs(s - 1.0) > 1e-6:
        raise ValueError(f"cf_prob_tuple must be non-negative and sum to 1, got: {vals}")
    return vals


def _parse_range_tuple(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 2 comma-separated floats, got: {text}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo <= hi):
        raise ValueError(f"invalid range (expected 0 <= lo <= hi), got: {(lo, hi)}")
    return lo, hi


def _validate_cf_config(cfg: SFTConfig) -> None:
    if int(cfg.cf_every_n_steps) <= 0:
        raise ValueError("cf_every_n_steps must be > 0")
    if float(cfg.cf_lambda) < 0.0:
        raise ValueError("cf_lambda must be >= 0")
    if float(cfg.cf_eps) <= 0.0:
        raise ValueError("cf_eps must be > 0")
    if int(cfg.cf_min_z_len) < 0:
        raise ValueError("cf_min_z_len must be >= 0")
    p_trunc, p_reverse, p_random = cfg.cf_prob_tuple
    if min(float(p_trunc), float(p_reverse), float(p_random)) < 0.0:
        raise ValueError("cf_prob_tuple probabilities must be non-negative")
    if abs(float(p_trunc + p_reverse + p_random) - 1.0) > 1e-6:
        raise ValueError("cf_prob_tuple must sum to 1")
    lo, hi = cfg.cf_trunc_range
    if not (0.0 <= float(lo) <= float(hi)):
        raise ValueError("cf_trunc_range must satisfy 0 <= lo <= hi")


def _sample_cf_variant(cfg: SFTConfig, rng: random.Random) -> str:
    p_trunc, p_reverse, _ = cfg.cf_prob_tuple
    u = float(rng.random())
    if u < float(p_trunc):
        return "truncate"
    if u < float(p_trunc + p_reverse):
        return "reverse"
    return "random"


def _build_counterfactual_batch(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_token_id: int,
    z_token_ids: Sequence[int],
    pad_token_id: int,
    cf_min_z_len: int,
    variant_name: str,
    trunc_range: Tuple[float, float],
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_cf = input_ids.clone()
    attention_mask_cf = attention_mask.clone()

    z_set = set(int(x) for x in z_token_ids)
    bsz, _ = input_ids.shape
    eligible_mask = torch.zeros((bsz,), dtype=torch.bool, device=input_ids.device)
    visible_z_counts = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)

    lo, hi = float(trunc_range[0]), float(trunc_range[1])

    for b in range(bsz):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 0:
            continue
        row = input_ids[b, :valid_len]
        ans_idx = (row == int(answer_token_id)).nonzero(as_tuple=False).view(-1)
        if int(ans_idx.numel()) == 0:
            continue
        ans_pos = int(ans_idx[0].item())
        z_pos: List[int] = []
        for j in range(ans_pos):
            if int(row[j].item()) in z_set:
                z_pos.append(j)
        lz = len(z_pos)
        if lz < int(cf_min_z_len):
            continue

        eligible_mask[b] = True

        if variant_name == "reverse":
            src = input_ids[b, z_pos].clone()
            input_ids_cf[b, z_pos] = src.flip(0)
            visible_z_counts[b] = int(lz)
        elif variant_name == "random":
            vals = [int(rng.choice(z_token_ids)) for _ in z_pos]
            if vals:
                input_ids_cf[b, z_pos] = torch.as_tensor(vals, dtype=input_ids.dtype, device=input_ids.device)
            visible_z_counts[b] = int(lz)
        elif variant_name == "truncate":
            r = float(rng.uniform(lo, hi))
            remove_count = int(math.ceil(r * lz))
            keep_k = max(0, lz - remove_count)
            remove_pos = z_pos[keep_k:]
            if remove_pos:
                remove_tensor = torch.as_tensor(remove_pos, dtype=torch.long, device=input_ids.device)
                input_ids_cf[b, remove_tensor] = int(pad_token_id)
                attention_mask_cf[b, remove_tensor] = 0
            visible_z_counts[b] = int(keep_k)
        else:
            raise ValueError(f"unknown counterfactual variant: {variant_name}")

    return input_ids_cf, attention_mask_cf, eligible_mask, visible_z_counts


def _log_full_sequence_example(*, tokenizer, batch: Dict[str, torch.Tensor], log_path: str, step: int) -> None:
    if "input_ids" not in batch or "attention_mask" not in batch:
        return
    if int(batch["input_ids"].shape[0]) == 0:
        return
    ids = batch["input_ids"][0].detach().cpu()
    mask = batch["attention_mask"][0].detach().cpu()
    seq_len = int(mask.sum().item())
    if seq_len <= 0:
        return
    text = tokenizer.decode(ids[:seq_len].tolist(), skip_special_tokens=False)
    _log(f"[example@warmup_end step={step}] {text}", log_path)


def train(cfg: SFTConfig) -> str:
    if not cfg.base_model_or_checkpoint.strip():
        raise ValueError("config.base_model_or_checkpoint is empty; fill with Phase1 checkpoint path")
    if not cfg.train_dataset_name.strip():
        raise ValueError("config.train_dataset_name is empty; fill with HF dataset path")
    if not cfg.eval_dataset_name.strip():
        raise ValueError("config.eval_dataset_name is empty; fill with HF dataset path")
    if int(cfg.vocab_size) <= 0:
        raise ValueError("config.vocab_size must be > 0")
    _validate_cf_config(cfg)

    _set_seed(cfg.seed)

    run_dir = _make_run_dir(cfg)
    log_path = os.path.join(run_dir, "logs", "train.log")
    metrics_csv = os.path.join(run_dir, "logs", "metrics.csv")
    eval_jsonl = os.path.join(run_dir, "logs", "eval.jsonl")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_or_checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_or_checkpoint,#
        torch_dtype=_dtype_from_str("bfloat16") if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    token_info = _ensure_sft_tokens(tokenizer, model, cfg.vocab_size)
    z_token_ids = token_info["z_token_ids"]
    digit_token_ids = token_info["digit_token_ids"]
    answer_token_id = token_info["answer_token_id"][0]
    z_and_answer_allowed = list(z_token_ids) + [int(answer_token_id)]

    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))

    train_records = _load_hf_records(cfg.train_dataset_name, cfg.train_dataset_split)
    eval_records = _load_hf_records(cfg.eval_dataset_name, cfg.eval_dataset_split)

    train_loader = _build_loader(records=train_records, tokenizer=tokenizer, cfg=cfg, shuffle=True, train=True)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_rng = random.Random(int(cfg.seed))

    step = 0
    micro = 0
    best_pass = -math.inf
    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    _log(f"run_dir={run_dir}", log_path)
    _log(f"train_size={len(train_records)} eval_size={len(eval_records)}", log_path)
    _log(
        "counterfactual enabled={} every_n_steps={} prob={} lambda={} kl_margin={} min_z_len={} trunc_range={}".format(
            bool(cfg.cf_enabled),
            int(cfg.cf_every_n_steps),
            tuple(float(x) for x in cfg.cf_prob_tuple),
            float(cfg.cf_lambda),
            float(cfg.cf_kl_margin),
            int(cfg.cf_min_z_len),
            tuple(float(x) for x in cfg.cf_trunc_range),
        ),
        log_path,
    )

    warmup_hooks = _apply_warmup_freeze(model=model, z_token_ids=z_token_ids, warmup_active=cfg.warmup_steps > 0)

    while step < int(cfg.max_steps):
        for batch in train_loader:
            if step >= int(cfg.max_steps):
                break

            model.train()
            for k in ("input_ids", "attention_mask", "labels", "target_class"):
                batch[k] = batch[k].to(device)

            accum_steps = max(1, int(cfg.gradient_accumulation_steps))
            will_step = ((micro + 1) % accum_steps) == 0
            cf_trigger = bool(cfg.cf_enabled) and will_step and (((step + 1) % int(cfg.cf_every_n_steps)) == 0)
            cf_applied = False
            cf_variant_name = "none"
            cf_loss_scalar = 0.0
            cf_mean_sym_kl = 0.0
            cf_mean_entropy = 0.0
            cf_visible_z_mean = 0.0
            cf_eligible_count = 0.0
            cf_loss_value = torch.zeros((), dtype=torch.float32, device=device)

            with scaler_ctx:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss_out = compute_weighted_loss(
                    logits=out.logits,
                    labels=batch["labels"],
                    target_class=batch["target_class"],
                    z_allowed_ids=z_and_answer_allowed,
                    digit_allowed_ids=digit_token_ids,
                    w_z=cfg.w_z,
                    w_answer=cfg.w_answer,
                    w_digits=cfg.w_digits,
                    z_label_smoothing=cfg.z_label_smoothing,
                    keep_prob=cfg.keep_prob,
                )
                total_loss = loss_out.total

                if cf_trigger:
                    cf_variant_name = _sample_cf_variant(cfg, train_rng)
                    input_ids_cf, attention_mask_cf, eligible_mask, visible_z_counts = _build_counterfactual_batch(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        answer_token_id=answer_token_id,
                        z_token_ids=z_token_ids,
                        pad_token_id=int(tokenizer.pad_token_id),
                        cf_min_z_len=int(cfg.cf_min_z_len),
                        variant_name=cf_variant_name,
                        trunc_range=cfg.cf_trunc_range,
                        rng=train_rng,
                    )
                    cf_eligible_count = float(eligible_mask.float().sum().item())
                    if cf_eligible_count > 0:
                        cf_applied = True
                        out_cf = model(input_ids=input_ids_cf, attention_mask=attention_mask_cf)
                        clean_digit_logits, digit_valid_mask = extract_digit_logits(
                            logits=out.logits,
                            target_class=batch["target_class"],
                            digit_allowed_ids=digit_token_ids,
                        )
                        cf_digit_logits, cf_digit_valid_mask = extract_digit_logits(
                            logits=out_cf.logits,
                            target_class=batch["target_class"],
                            digit_allowed_ids=digit_token_ids,
                        )
                        del out_cf
                        if clean_digit_logits.shape != cf_digit_logits.shape:
                            raise RuntimeError(
                                "counterfactual digit logits shape mismatch: "
                                f"{tuple(clean_digit_logits.shape)} vs {tuple(cf_digit_logits.shape)}"
                            )
                        if digit_valid_mask.shape != cf_digit_valid_mask.shape or not bool(
                            torch.equal(digit_valid_mask, cf_digit_valid_mask)
                        ):
                            raise RuntimeError("counterfactual digit mask mismatch")
                        cf_out = compute_counterfactual_regularizer(
                            clean_digit_logits=clean_digit_logits,
                            cf_digit_logits=cf_digit_logits,
                            digit_valid_mask=digit_valid_mask,
                            eligible_mask=eligible_mask,
                            variant_name=cf_variant_name,
                            kl_margin=float(cfg.cf_kl_margin),
                            eps=float(cfg.cf_eps),
                        )
                        cf_loss_value = cf_out.loss
                        total_loss = total_loss + float(cfg.cf_lambda) * cf_loss_value
                        cf_loss_scalar = float(cf_out.loss.detach().item())
                        cf_mean_sym_kl = float(cf_out.mean_sym_kl)
                        cf_mean_entropy = float(cf_out.mean_entropy)
                        del cf_digit_logits
                        del clean_digit_logits
                        del cf_digit_valid_mask
                        del digit_valid_mask
                        del input_ids_cf
                        del attention_mask_cf
                        if cf_variant_name == "truncate":
                            v = visible_z_counts[eligible_mask]
                            cf_visible_z_mean = float(v.float().mean().item()) if int(v.numel()) > 0 else 0.0

                loss = total_loss / accum_steps

            loss.backward()
            micro += 1

            if micro % accum_steps != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == int(cfg.warmup_steps):
                _log_full_sequence_example(tokenizer=tokenizer, batch=batch, log_path=log_path, step=step)
                for h in warmup_hooks:
                    h.remove()
                warmup_hooks = _apply_warmup_freeze(
                    model=model,
                    z_token_ids=z_token_ids,
                    warmup_active=False,
                )
                _log(f"warmup ended at step={step}; full model unfrozen", log_path)



            if step % int(cfg.log_interval_steps) == 0:
                mean_z_len = float(batch["z_lens"].float().mean().item())
                row = {
                    "step": float(step),
                    "L_total": float(total_loss.detach().item()),
                    "L_z": float(loss_out.l_z.detach().item()),
                    "L_answer": float(loss_out.l_answer.detach().item()),
                    "L_digits": float(loss_out.l_digits.detach().item()),
                    "z_acc": float(loss_out.z_acc),
                    "digit_exact_match": float(loss_out.digit_exact_match),
                    "avg_z_len": float(mean_z_len),
                    "no_answer_before_kmax": 0.0,
                    "cf_enabled": float(bool(cfg.cf_enabled)),
                    "cf_applied": float(bool(cf_applied)),
                    "cf_loss": float(cf_loss_scalar),
                    "cf_variant_truncate": float(cf_variant_name == "truncate"),
                    "cf_variant_reverse": float(cf_variant_name == "reverse"),
                    "cf_variant_random": float(cf_variant_name == "random"),
                    "cf_mean_sym_kl": float(cf_mean_sym_kl),
                    "cf_mean_entropy": float(cf_mean_entropy),
                    "cf_eligible_count": float(cf_eligible_count),
                    "cf_visible_z_mean": float(cf_visible_z_mean),
                }
                _append_metrics_csv(metrics_csv, row)
                _log(
                    "step={} L={:.4f} Lz={:.4f} La={:.4f} Ld={:.4f} z_acc={:.3f} d_em={:.3f} z_len={:.2f} cf_on={} cf_applied={} cf_variant={} cf_loss={:.4f} cf_kl={:.4f} cf_H={:.4f}".format(
                        step,
                        row["L_total"],
                        row["L_z"],
                        row["L_answer"],
                        row["L_digits"],
                        row["z_acc"],
                        row["digit_exact_match"],
                        row["avg_z_len"],
                        int(bool(cfg.cf_enabled)),
                        int(bool(cf_applied)),
                        cf_variant_name,
                        row["cf_loss"],
                        row["cf_mean_sym_kl"],
                        row["cf_mean_entropy"],
                    ),
                    log_path,
                )

            if step % int(cfg.save_interval_steps) == 0:
                _save_last(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    step=step,
                    best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
                )
                _log(f"saved last checkpoint at step={step}", log_path)

            if step % int(cfg.save_every_steps) == 0:
                p = _save_periodic(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    step=step,
                    keep_last_k=int(cfg.keep_last_k),
                )
                _log(f"saved periodic checkpoint {p}", log_path)

            warmup_steps = int(cfg.warmup_steps)
            eval_ready = (warmup_steps <= 0) or (step >= warmup_steps)
            eval_on_interval = step % int(cfg.eval_interval_steps) == 0
            if eval_ready and eval_on_interval:
                eval_model_path = _save_last(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    step=step,
                    best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
                )
                model.train(False)
                torch.cuda.empty_cache()
                metrics = evaluate_with_vllm(
                    model_path=eval_model_path,
                    records=eval_records,
                    pass_at_n=cfg.pass_at_n,
                    k_max=cfg.k_max,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    vocab_size=cfg.vocab_size,
                    output_jsonl_path=eval_jsonl,
                )

                gc.collect()
                torch.cuda.empty_cache()
                _log(
                    "eval step={} pass@{}={:.4f} greedy={:.4f} z_len={:.2f} no_answer={:.4f}".format(
                        step,
                        cfg.pass_at_n,
                        metrics.pass_at_n,
                        metrics.greedy_exact_match,
                        metrics.mean_z_len,
                        metrics.no_answer_before_kmax_rate,
                    ),
                    log_path,
                )
                _log(f"eval generations appended to {eval_jsonl}", log_path)

                _append_metrics_csv(
                    metrics_csv,
                    {
                        "step": float(step),
                        "pass_at_n": float(metrics.pass_at_n),
                        "greedy_exact_match": float(metrics.greedy_exact_match),
                        "eval_mean_z_len": float(metrics.mean_z_len),
                        "eval_no_answer_before_kmax": float(metrics.no_answer_before_kmax_rate),
                    },
                )

                if cfg.save_best and metrics.pass_at_n > best_pass:
                    best_pass = metrics.pass_at_n
                    best_path = _save_best(
                        run_dir=run_dir,
                        model=model,
                        tokenizer=tokenizer,
                        step=step,
                        metric=best_pass,
                    )
                    _log(f"new best pass@{cfg.pass_at_n}={best_pass:.4f}; saved {best_path}", log_path)

    _save_last(
        run_dir=run_dir,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        step=step,
        best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
    )
    if cfg.save_ppo_init:
        p = _save_ppo_init(run_dir=run_dir, model=model, tokenizer=tokenizer)
        _log(f"saved ppo_init snapshot at {p}", log_path)

    _log("training complete", log_path)
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3 SFT over discrete latent Z-programs")
    p.add_argument("--base_model_or_checkpoint", type=str, default="")
    p.add_argument("--train_dataset_name", type=str, default="")
    p.add_argument("--train_dataset_split", type=str, default="train")
    p.add_argument("--eval_dataset_name", type=str, default="")
    p.add_argument("--eval_dataset_split", type=str, default="eval")
    p.add_argument("--vocab_size", type=int, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--z_label_smoothing", type=float, default=0.05)
    p.add_argument("--w_z", type=float, default=0.1)
    p.add_argument("--w_answer", type=float, default=0.5)
    p.add_argument("--w_digits", type=float, default=1.0)
    p.add_argument("--cf_enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cf_every_n_steps", type=int, default=4)
    p.add_argument("--cf_prob_tuple", type=str, default="0.5,0.25,0.25")
    p.add_argument("--cf_lambda", type=float, default=0.1)
    p.add_argument("--cf_kl_margin", type=float, default=0.5)
    p.add_argument("--cf_eps", type=float, default=1e-8)
    p.add_argument("--cf_min_z_len", type=int, default=2)
    p.add_argument("--cf_trunc_range", type=str, default="0.5,1.0")

    p.add_argument("--eval_interval_steps", type=int, default=500)
    p.add_argument("--pass_at_n", type=int, default=16)
    p.add_argument("--k_max", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--run_root", type=str, default="runs/sft_z")
    p.add_argument("--log_interval_steps", type=int, default=20)
    p.add_argument("--save_interval_steps", type=int, default=200)
    p.add_argument("--save_every_steps", type=int, default=2000)
    p.add_argument("--keep_last_k", type=int, default=3)
    p.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_ppo_init", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cf_prob_tuple = _parse_prob_tuple(args.cf_prob_tuple)
    cf_trunc_range = _parse_range_tuple(args.cf_trunc_range)
    cfg = SFTConfig(
        base_model_or_checkpoint=args.base_model_or_checkpoint,
        train_dataset_name=args.train_dataset_name,
        train_dataset_split=args.train_dataset_split,
        eval_dataset_name=args.eval_dataset_name,
        eval_dataset_split=args.eval_dataset_split,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        z_label_smoothing=args.z_label_smoothing,
        w_z=args.w_z,
        w_answer=args.w_answer,
        w_digits=args.w_digits,
        cf_enabled=bool(args.cf_enabled),
        cf_every_n_steps=args.cf_every_n_steps,
        cf_prob_tuple=cf_prob_tuple,
        cf_lambda=args.cf_lambda,
        cf_kl_margin=args.cf_kl_margin,
        cf_eps=args.cf_eps,
        cf_min_z_len=args.cf_min_z_len,
        cf_trunc_range=cf_trunc_range,
        eval_interval_steps=args.eval_interval_steps,
        pass_at_n=args.pass_at_n,
        k_max=args.k_max,
        temperature=args.temperature,
        top_p=args.top_p,
        run_root=args.run_root,
        log_interval_steps=args.log_interval_steps,
        save_interval_steps=args.save_interval_steps,
        save_every_steps=args.save_every_steps,
        keep_last_k=args.keep_last_k,
        save_best=bool(args.save_best),
        save_ppo_init=bool(args.save_ppo_init),
    )
    train(cfg)


if __name__ == "__main__":
    main()
