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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PhaseConfig, SFTConfig
from .dataset import THINK_CLOSE_TOKEN, CurriculumSFTDataset, SFTCollator
from .losses import compute_counterfactual_regularizer, compute_weighted_loss, extract_digit_logits

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


def _ensure_sft_tokens(tokenizer, model, vocab_size: int) -> Dict[str, object]:
    z_tokens = [f"<z_{i}>" for i in range(int(vocab_size))]

    existing_vocab = tokenizer.get_vocab()
    to_add = [t for t in z_tokens if t not in existing_vocab]

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

    closing_think_ids = tokenizer.encode(THINK_CLOSE_TOKEN, add_special_tokens=False)
    if len(closing_think_ids) != 1:
        raise RuntimeError(
            f"Tokenization contract violated for {THINK_CLOSE_TOKEN}: expected 1 token, got {closing_think_ids}"
        )
    closing_think_token_id = int(closing_think_ids[0])
    closing_think_from_vocab = int(tokenizer.convert_tokens_to_ids(THINK_CLOSE_TOKEN))
    if closing_think_from_vocab >= 0 and closing_think_from_vocab != closing_think_token_id:
        raise RuntimeError(
            f"Tokenizer id mismatch for {THINK_CLOSE_TOKEN}: "
            f"convert_tokens_to_ids={closing_think_from_vocab}, encode={closing_think_token_id}"
        )

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
        "closing_think_token_id": closing_think_token_id,
    }


def _make_run_dir(cfg: SFTConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"{ts}__V{cfg.vocab_size}__lr{cfg.learning_rate}"
    run_dir = os.path.join(cfg.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "last"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def _resolve_resume_checkpoint(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isdir(os.path.join(p, "last")):
        p = os.path.join(p, "last")
    state_path = os.path.join(p, "trainer_state.json")
    if not os.path.isfile(state_path):
        raise ValueError(
            f"resume_from does not point to a checkpoint with trainer_state.json: {path}"
        )
    return p


def _save_model_dir(model, tokenizer, out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _save_checkpoint(
    *,
    out_dir: str,
    model,
    tokenizer,
    optimizer,
    state: Dict[str, object],
) -> None:
    # Resume is coarse: phase/epoch boundary only, not exact batch-level replay.
    _save_model_dir(model, tokenizer, out_dir)
    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    with open(os.path.join(out_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _load_hf_records(dataset_name: str, split: str) -> List[Dict]:
    ds = load_dataset(dataset_name, split=split)
    return [dict(x) for x in ds]


def _filter_records_for_phase(*, records: Sequence[Dict], max_tokens: int) -> List[Dict]:
    out: List[Dict] = []
    for i, row in enumerate(records):
        if "tokens_count" not in row:
            raise ValueError(f"Row {i} is missing required column 'tokens_count'")
        try:
            count = int(row["tokens_count"])
        except Exception as e:
            raise ValueError(f"Row {i} has non-integer tokens_count={row['tokens_count']}") from e
        if count <= int(max_tokens):
            out.append(row)
    return out


def _build_loader(
    *,
    records: Iterable[Dict],
    tokenizer,
    cfg: SFTConfig,
    phase: PhaseConfig,
    shuffle: bool,
) -> tuple[DataLoader, CurriculumSFTDataset]:
    ds = CurriculumSFTDataset(
        records=records,
        tokenizer=tokenizer,
        vocab_size=cfg.vocab_size,
        z_ratio=phase.z_ratio,
        min_z_tokens=phase.min_z_tokens,
    )
    if len(ds) == 0:
        raise ValueError("Curriculum dataset has 0 usable samples after preprocessing.")

    collator = SFTCollator(tokenizer=tokenizer, max_length=cfg.max_length)
    loader_kwargs = {
        "batch_size": int(phase.batch_size),
        "shuffle": bool(shuffle),
        "collate_fn": collator,
        "drop_last": False,
        "num_workers": max(0, int(cfg.dataloader_num_workers)),
        "pin_memory": bool(cfg.dataloader_pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(max(1, cfg.dataloader_prefetch_factor))
    return DataLoader(ds, **loader_kwargs), ds


def _build_optimizer(trainable_params, cfg: SFTConfig):
    if cfg.optimizer_name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer_name == "adamw_8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as e:
            raise RuntimeError("bitsandbytes is required for optimizer_name='adamw_8bit'") from e
        return AdamW8bit(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {cfg.optimizer_name}. Supported values: 'adamw', 'adamw_8bit'")


def _append_metrics_csv(path: str, row: Dict[str, float]) -> None:
    exists = os.path.isfile(path)
    fields = list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def _validate_phases(phases: Sequence[PhaseConfig]) -> None:
    if not phases:
        raise ValueError("config.phases must be non-empty")
    ratios = [float(p.z_ratio) for p in phases]
    for i, r in enumerate(ratios):
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"phase[{i}].z_ratio must be in [0,1], got {r}")
    for i in range(1, len(ratios)):
        if not (ratios[i] > ratios[i - 1]):
            raise ValueError(
                "z_ratio schedule must be strictly increasing. "
                f"Found phase[{i - 1}]={ratios[i - 1]} and phase[{i}]={ratios[i]}"
            )
    for i, phase in enumerate(phases):
        if int(phase.batch_size) <= 0:
            raise ValueError(f"phase[{i}].batch_size must be > 0")
        if int(phase.gradient_accumulation_steps) <= 0:
            raise ValueError(f"phase[{i}].gradient_accumulation_steps must be > 0")
        if int(phase.max_tokens) <= 0:
            raise ValueError(f"phase[{i}].max_tokens must be > 0")
        if float(phase.epochs) <= 0:
            raise ValueError(f"phase[{i}].epochs must be > 0")
        if int(phase.min_z_tokens) < 1:
            raise ValueError(f"phase[{i}].min_z_tokens must be >= 1")


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
    closing_think_token_id: int,
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
        ans_idx = (row == int(closing_think_token_id)).nonzero(as_tuple=False).view(-1)
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


def _decode_sequence(tokenizer, input_ids: Sequence[int], attention_mask: Optional[Sequence[int]] = None) -> str:
    if attention_mask is None:
        valid = list(input_ids)
    else:
        n = int(sum(int(x) for x in attention_mask))
        valid = list(input_ids)[:n]
    return tokenizer.decode(valid, skip_special_tokens=False)


def _count_z_before_closing_think(
    *,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    closing_think_token_id: int,
    z_token_ids: Sequence[int],
) -> int:
    valid_len = int(sum(int(x) for x in attention_mask))
    row = list(input_ids)[:valid_len]
    z_set = set(int(x) for x in z_token_ids)
    try:
        ans_pos = row.index(int(closing_think_token_id))
    except ValueError:
        return 0
    return sum(1 for tok in row[:ans_pos] if int(tok) in z_set)


def _log_phase_debug_examples(
    *,
    cfg: SFTConfig,
    phase: PhaseConfig,
    phase_idx: int,
    dataset: CurriculumSFTDataset,
    tokenizer,
    closing_think_token_id: int,
    z_token_ids: Sequence[int],
    log_path: str,
) -> None:
    _log(f"[phase_debug] phase={phase_idx} dataset_len={len(dataset)}", log_path)
    if len(dataset) == 0:
        _log(f"[phase_debug] phase={phase_idx} has empty dataset after filtering.", log_path)
        return

    first = dataset[0]
    decoded_full = _decode_sequence(
        tokenizer=tokenizer,
        input_ids=first["input_ids"],
        attention_mask=first["attention_mask"],
    )
    _log(f"[phase_debug] phase={phase_idx} full_example_decoded:\n{decoded_full}", log_path)

    if not bool(phase.cf_loss):
        return

    eligible = None
    for i in range(len(dataset)):
        ex = dataset[i]
        z_count = _count_z_before_closing_think(
            input_ids=ex["input_ids"],
            attention_mask=ex["attention_mask"],
            closing_think_token_id=closing_think_token_id,
            z_token_ids=z_token_ids,
        )
        if z_count >= int(cfg.cf_min_z_len):
            eligible = ex
            break

    if eligible is None:
        _log(
            f"[phase_debug] phase={phase_idx} cf_examples_skipped: no sample with >= {cfg.cf_min_z_len} Z tokens before </think>.",
            log_path,
        )
        return

    orig_decoded = _decode_sequence(
        tokenizer=tokenizer,
        input_ids=eligible["input_ids"],
        attention_mask=eligible["attention_mask"],
    )
    base_ids = torch.tensor([eligible["input_ids"]], dtype=torch.long)
    base_mask = torch.tensor([eligible["attention_mask"]], dtype=torch.long)
    cf_rng = random.Random(int(cfg.seed) + int(phase_idx))

    for variant in ("truncate", "random", "reverse"):
        cf_ids, cf_mask, _, _ = _build_counterfactual_batch(
            input_ids=base_ids,
            attention_mask=base_mask,
            closing_think_token_id=int(closing_think_token_id),
            z_token_ids=z_token_ids,
            pad_token_id=int(tokenizer.pad_token_id),
            cf_min_z_len=int(cfg.cf_min_z_len),
            variant_name=variant,
            trunc_range=cfg.cf_trunc_range,
            rng=cf_rng,
        )
        cf_decoded = _decode_sequence(
            tokenizer=tokenizer,
            input_ids=cf_ids[0].tolist(),
            attention_mask=cf_mask[0].tolist(),
        )
        _log(
            f"[phase_debug][cf_variant={variant}] original_decoded:\n{orig_decoded}\n"
            f"[phase_debug][cf_variant={variant}] counterfactual_decoded:\n{cf_decoded}",
            log_path,
        )


def _run_phase_batches(
    *,
    cfg: SFTConfig,
    phase: PhaseConfig,
    loader: DataLoader,
    model,
    optimizer,
    device: torch.device,
    scaler_ctx,
    step: int,
    train_rng: random.Random,
    tokenizer,
    z_token_ids: Sequence[int],
    digit_token_ids: Sequence[int],
    closing_think_token_id: int,
    z_and_boundary_allowed: Sequence[int],
    log_path: str,
    metrics_csv: str,
    phase_idx: int,
    epoch_tag: float,
    max_batches: Optional[int] = None,
) -> tuple[int, bool]:
    accum_steps = int(phase.gradient_accumulation_steps)
    if accum_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    batches_seen = 0
    micro = 0
    stop_due_to_cap = False
    clip_drop_total = 0.0
    agg = {
        "micro_count": 0.0,
        "L": 0.0,
        "Lan": 0.0,
        "Lae": 0.0,
        "Ld": 0.0,
        "an_acc": 0.0,
        "d_em": 0.0,
        "z_len": 0.0,
        "cf_loss": 0.0,
        "cf_kl": 0.0,
        "cf_H": 0.0,
        "cf_applied": 0.0,
        "clip_drop_batch": 0.0,
    }

    for batch in loader:
        if cfg.max_steps is not None and step >= int(cfg.max_steps):
            stop_due_to_cap = True
            break
        if max_batches is not None and batches_seen >= int(max_batches):
            break
        batches_seen += 1

        model.train()
        for k in ("input_ids", "attention_mask", "labels", "target_class"):
            batch[k] = batch[k].to(device)
        batch["z_lens"] = batch["z_lens"].to(device)
        batch["text_thought_counts"] = batch["text_thought_counts"].to(device)

        will_step = ((micro + 1) % accum_steps) == 0
        phase_cf_enabled = bool(cfg.cf_enabled) and bool(phase.cf_loss)
        cf_trigger = phase_cf_enabled and will_step and (((step + 1) % int(cfg.cf_every_n_steps)) == 0)
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
                z_allowed_ids=z_and_boundary_allowed,
                digit_allowed_ids=digit_token_ids,
                alpha_z=cfg.alpha_z,
                alpha_answer=cfg.alpha_answer,
                alpha_digits=cfg.alpha_digits,
                z_label_smoothing=cfg.z_label_smoothing,
                keep_prob=cfg.keep_prob,
            )
            total_loss = loss_out.total

            if cf_trigger:
                cf_variant_name = _sample_cf_variant(cfg, train_rng)
                input_ids_cf, attention_mask_cf, eligible_mask, visible_z_counts = _build_counterfactual_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    closing_think_token_id=closing_think_token_id,
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
                    if cf_variant_name == "truncate":
                        v = visible_z_counts[eligible_mask]
                        cf_visible_z_mean = float(v.float().mean().item()) if int(v.numel()) > 0 else 0.0

            loss = total_loss / accum_steps

        micro_z_len = float(batch["z_lens"].float().mean().item())
        agg["micro_count"] += 1.0
        agg["L"] += float(total_loss.detach().item())
        agg["Lan"] += float(loss_out.l_z.detach().item())
        agg["Lae"] += float(loss_out.l_answer.detach().item())
        agg["Ld"] += float(loss_out.l_digits.detach().item())
        agg["an_acc"] += float(loss_out.z_acc)
        agg["d_em"] += float(loss_out.digit_exact_match)
        agg["z_len"] += float(micro_z_len)
        agg["cf_loss"] += float(cf_loss_scalar)
        agg["cf_kl"] += float(cf_mean_sym_kl)
        agg["cf_H"] += float(cf_mean_entropy)
        agg["cf_applied"] += 1.0 if cf_applied else 0.0
        agg["clip_drop_batch"] += float(loss_out.clip_drop_count)
        clip_drop_total += float(loss_out.clip_drop_count)

        loss.backward()
        micro += 1

        if micro % accum_steps != 0:
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        denom = max(1.0, float(agg["micro_count"]))
        eff = {
            "L": agg["L"] / denom,
            "Lan": agg["Lan"] / denom,
            "Lae": agg["Lae"] / denom,
            "Ld": agg["Ld"] / denom,
            "an_acc": agg["an_acc"] / denom,
            "d_em": agg["d_em"] / denom,
            "z_len": agg["z_len"] / denom,
            "cf_loss": agg["cf_loss"] / denom,
            "cf_kl": agg["cf_kl"] / denom,
            "cf_H": agg["cf_H"] / denom,
            "cf_applied": agg["cf_applied"] / denom,
            "clip_drop_batch": agg["clip_drop_batch"],
        }

        if step % int(cfg.log_interval_steps) == 0:
            mean_text_thoughts = float(batch["text_thought_counts"].float().mean().item())
            row = {
                "step": float(step),
                "phase_idx": float(phase_idx),
                "phase_z_ratio": float(phase.z_ratio),
                "phase_epoch_tag": float(epoch_tag),
                "L": float(eff["L"]),
                "Lan": float(eff["Lan"]),
                "Lae": float(eff["Lae"]),
                "Ld": float(eff["Ld"]),
                "an_acc": float(eff["an_acc"]),
                "d_em": float(eff["d_em"]),
                "z_len": float(eff["z_len"]),
                "cf_loss": float(eff["cf_loss"]),
                "cf_kl": float(eff["cf_kl"]),
                "cf_H": float(eff["cf_H"]),
                "cf_applied": float(eff["cf_applied"]),
                "clip_drop_batch": float(eff["clip_drop_batch"]),
                "clip_drop_total": float(clip_drop_total),
                "avg_text_thoughts": float(mean_text_thoughts),
                "cf_phase_enabled": float(phase_cf_enabled),
                "cf_variant_truncate": float(cf_variant_name == "truncate"),
                "cf_variant_reverse": float(cf_variant_name == "reverse"),
                "cf_variant_random": float(cf_variant_name == "random"),
                "cf_eligible_count": float(cf_eligible_count),
                "cf_visible_z_mean": float(cf_visible_z_mean),
            }
            _append_metrics_csv(metrics_csv, row)
            _log(
                "step={} phase={} z_ratio={} epoch_tag={:.2f} L={:.4f} Lan={:.4f} Lae={:.4f} Ld={:.4f} an_acc={:.3f} d_em={:.3f} z_len={:.2f} cf_loss={:.4f} cf_kl={:.4f} cf_H={:.4f} cf_applied={:.3f} clip_drop_batch={:.1f} clip_drop_total={:.1f} txt={:.2f} cf_on={} cf_variant={}".format(
                    step,
                    phase_idx,
                    phase.z_ratio,
                    epoch_tag,
                    row["L"],
                    row["Lan"],
                    row["Lae"],
                    row["Ld"],
                    row["an_acc"],
                    row["d_em"],
                    row["z_len"],
                    row["cf_loss"],
                    row["cf_kl"],
                    row["cf_H"],
                    row["cf_applied"],
                    row["clip_drop_batch"],
                    row["clip_drop_total"],
                    row["avg_text_thoughts"],
                    int(phase_cf_enabled),
                    cf_variant_name,
                ),
                log_path,
            )
        agg = {
            "micro_count": 0.0,
            "L": 0.0,
            "Lan": 0.0,
            "Lae": 0.0,
            "Ld": 0.0,
            "an_acc": 0.0,
            "d_em": 0.0,
            "z_len": 0.0,
            "cf_loss": 0.0,
            "cf_kl": 0.0,
            "cf_H": 0.0,
            "cf_applied": 0.0,
            "clip_drop_batch": 0.0,
        }

    return step, stop_due_to_cap


def _build_config_from_yaml_dict(data: Dict) -> SFTConfig:
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object.")
    for key in ("base_model_or_checkpoint", "train_dataset_name", "vocab_size"):
        if key not in data:
            raise ValueError(f"Config is missing required field '{key}'")

    if "phases" not in data:
        raise ValueError("Config must include a top-level 'phases' list.")

    phase_list = data["phases"]
    if not isinstance(phase_list, list):
        raise ValueError("Config field 'phases' must be a list.")

    phases: List[PhaseConfig] = []
    for i, p in enumerate(phase_list):
        if not isinstance(p, dict):
            raise ValueError(f"phases[{i}] must be a mapping/object.")
        required = ("z_ratio", "batch_size", "gradient_accumulation_steps", "max_tokens", "epochs", "cf_loss")
        missing = [k for k in required if k not in p]
        if missing:
            raise ValueError(f"phases[{i}] missing required fields: {missing}")
        phases.append(
            PhaseConfig(
                z_ratio=float(p["z_ratio"]),
                batch_size=int(p["batch_size"]),
                gradient_accumulation_steps=int(p["gradient_accumulation_steps"]),
                max_tokens=int(p["max_tokens"]),
                epochs=float(p["epochs"]),
                cf_loss=bool(p["cf_loss"]),
                min_z_tokens=int(p.get("min_z_tokens", 1)),
            )
        )

    keep_prob_raw = data.get("keep_prob", [0.2, 0.3, 0.45, 0.75, 1.0])
    keep_prob = tuple(float(x) for x in keep_prob_raw)

    cf_prob_raw = data.get("cf_prob_tuple", [0.5, 0.25, 0.25])
    cf_prob_tuple = tuple(float(x) for x in cf_prob_raw)

    cf_trunc_raw = data.get("cf_trunc_range", [0.5, 1.0])
    cf_trunc_range = tuple(float(x) for x in cf_trunc_raw)
    if len(cf_trunc_range) != 2:
        raise ValueError("cf_trunc_range must have exactly 2 values")

    max_steps_raw = data.get("max_steps", None)
    max_steps = None if max_steps_raw is None else int(max_steps_raw)

    return SFTConfig(
        base_model_or_checkpoint=str(data["base_model_or_checkpoint"]),
        train_dataset_name=str(data["train_dataset_name"]),
        train_dataset_split=str(data.get("train_dataset_split", "train")),
        vocab_size=int(data["vocab_size"]),
        seed=int(data.get("seed", 42)),
        learning_rate=float(data.get("learning_rate", 2e-5)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        optimizer_name=str(data.get("optimizer_name", "adamw_8bit")),
        max_length=int(data.get("max_length", 16000)),
        torch_device=str(data.get("torch_device", "cuda:0")),
        max_steps=max_steps,
        z_label_smoothing=float(data.get("z_label_smoothing", 0.05)),
        alpha_z=float(data.get("alpha_z", 0.1)),
        alpha_answer=float(data.get("alpha_answer", 0.5)),
        alpha_digits=float(data.get("alpha_digits", 1.0)),
        keep_prob=keep_prob,
        cf_enabled=bool(data.get("cf_enabled", True)),
        cf_every_n_steps=int(data.get("cf_every_n_steps", 2)),
        cf_prob_tuple=cf_prob_tuple,  # type: ignore[arg-type]
        cf_lambda=float(data.get("cf_lambda", 1.0)),
        cf_kl_margin=float(data.get("cf_kl_margin", 0.5)),
        cf_eps=float(data.get("cf_eps", 1e-8)),
        cf_min_z_len=int(data.get("cf_min_z_len", 2)),
        cf_trunc_range=cf_trunc_range,  # type: ignore[arg-type]
        phases=tuple(phases),
        run_root=str(data.get("run_root", "runs/sft_curriculum")),
        run_name=(None if data.get("run_name") in (None, "") else str(data.get("run_name"))),
        log_interval_steps=int(data.get("log_interval_steps", 20)),
        save_every_epoch=bool(data.get("save_every_epoch", True)),
        save_phase_end=bool(data.get("save_phase_end", True)),
        resume_from=(None if data.get("resume_from") in (None, "") else str(data.get("resume_from"))),
        dataloader_num_workers=int(data.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=bool(data.get("dataloader_pin_memory", True)),
        dataloader_prefetch_factor=int(data.get("dataloader_prefetch_factor", 2)),
    )


def _load_config_yaml(path: str) -> SFTConfig:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from e
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return _build_config_from_yaml_dict(payload)


def train(cfg: SFTConfig) -> str:
    if not cfg.base_model_or_checkpoint.strip():
        raise ValueError("config.base_model_or_checkpoint is empty.")
    if not cfg.train_dataset_name.strip():
        raise ValueError("config.train_dataset_name is empty.")
    if int(cfg.vocab_size) <= 0:
        raise ValueError("config.vocab_size must be > 0")
    if len(cfg.keep_prob) != 5:
        raise ValueError(f"keep_prob must have length 5, got {len(cfg.keep_prob)}")
    _validate_cf_config(cfg)
    _validate_phases(cfg.phases)

    _set_seed(cfg.seed)
    train_rng = random.Random(int(cfg.seed))

    if torch.cuda.is_available():
        device = torch.device(str(cfg.torch_device))
    else:
        device = torch.device("cpu")

    resume_ckpt = None
    start_phase_idx = 0
    start_epoch_idx = 0
    step = 0

    if cfg.resume_from:
        resume_ckpt = _resolve_resume_checkpoint(cfg.resume_from)
        state_path = os.path.join(resume_ckpt, "trainer_state.json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        run_dir = str(state.get("run_dir", os.path.abspath(os.path.join(resume_ckpt, "..", ".."))))
        start_phase_idx = int(state.get("next_phase_idx", 0))
        start_epoch_idx = int(state.get("next_epoch_idx_in_phase", 0))
        step = int(state.get("global_step", 0))
        tokenizer = AutoTokenizer.from_pretrained(resume_ckpt, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            resume_ckpt,
            torch_dtype=_dtype_from_str("bfloat16") if torch.cuda.is_available() else torch.float32,
        )
    else:
        run_dir = _make_run_dir(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_or_checkpoint, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_or_checkpoint,
            torch_dtype=_dtype_from_str("bfloat16") if torch.cuda.is_available() else torch.float32,
        )

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "last"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    log_path = os.path.join(run_dir, "logs", "train.log")
    metrics_csv = os.path.join(run_dir, "logs", "metrics.csv")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    token_info = _ensure_sft_tokens(tokenizer, model, cfg.vocab_size)
    z_token_ids = token_info["z_token_ids"]
    digit_token_ids = token_info["digit_token_ids"]
    closing_think_token_id = int(token_info["closing_think_token_id"])
    z_and_boundary_allowed = list(z_token_ids) + [int(closing_think_token_id)]
    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))

    train_records = _load_hf_records(cfg.train_dataset_name, cfg.train_dataset_split)
    optimizer = _build_optimizer(model.parameters(), cfg)
    if resume_ckpt:
        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        _log(
            f"resumed from {resume_ckpt} at phase={start_phase_idx} epoch={start_epoch_idx} step={step}",
            log_path,
        )
        _log(
            "resume_granularity=coarse (phase/epoch boundary). "
            "If interrupted mid-epoch or mid-partial-epoch, that segment restarts from its beginning.",
            log_path,
        )

    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    _log(f"run_dir={run_dir}", log_path)
    _log(f"train_size={len(train_records)}", log_path)
    _log(
        "counterfactual global_enabled={} every_n_steps={} prob={} lambda={} kl_margin={} min_z_len={} trunc_range={}".format(
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

    reached_cap = False
    stop_next_phase_idx = int(len(cfg.phases))
    stop_next_epoch_idx = 0
    for phase_idx in range(start_phase_idx, len(cfg.phases)):
        phase = cfg.phases[phase_idx]
        phase_records = _filter_records_for_phase(records=train_records, max_tokens=phase.max_tokens)
        loader, phase_ds = _build_loader(
            records=phase_records,
            tokenizer=tokenizer,
            cfg=cfg,
            phase=phase,
            shuffle=True,
        )
        avg_z = float(sum(len(x.z_ids) for x in phase_ds.samples) / len(phase_ds.samples))
        avg_text = float(sum(len(x.text_thoughts) for x in phase_ds.samples) / len(phase_ds.samples))

        _log(
            "phase={} z_ratio={} max_tokens={} rows_kept={} avg_z={:.2f} avg_text={:.2f} batch={} grad_accum={} epochs={} cf_phase={}".format(
                phase_idx,
                phase.z_ratio,
                phase.max_tokens,
                len(phase_records),
                avg_z,
                avg_text,
                phase.batch_size,
                phase.gradient_accumulation_steps,
                phase.epochs,
                int(bool(phase.cf_loss)),
            ),
            log_path,
        )
        _log_phase_debug_examples(
            cfg=cfg,
            phase=phase,
            phase_idx=phase_idx,
            dataset=phase_ds,
            tokenizer=tokenizer,
            closing_think_token_id=closing_think_token_id,
            z_token_ids=z_token_ids,
            log_path=log_path,
        )

        full_epochs = int(math.floor(float(phase.epochs)))
        partial_frac = float(phase.epochs) - float(full_epochs)
        epoch_start = start_epoch_idx if phase_idx == start_phase_idx else 0
        if partial_frac > 0.0:
            _log(
                f"phase={phase_idx} uses fractional epoch={partial_frac:.4f}; "
                "resume for this segment is coarse and restarts the partial segment from batch 0.",
                log_path,
            )

        for epoch_idx in range(epoch_start, full_epochs):
            step, hit_cap = _run_phase_batches(
                cfg=cfg,
                phase=phase,
                loader=loader,
                model=model,
                optimizer=optimizer,
                device=device,
                scaler_ctx=scaler_ctx,
                step=step,
                train_rng=train_rng,
                tokenizer=tokenizer,
                z_token_ids=z_token_ids,
                digit_token_ids=digit_token_ids,
                closing_think_token_id=closing_think_token_id,
                z_and_boundary_allowed=z_and_boundary_allowed,
                log_path=log_path,
                metrics_csv=metrics_csv,
                phase_idx=phase_idx,
                epoch_tag=float(epoch_idx + 1),
            )
            if cfg.save_every_epoch:
                state = {
                    "run_dir": run_dir,
                    "global_step": int(step),
                    "next_phase_idx": int(phase_idx),
                    "next_epoch_idx_in_phase": int(epoch_idx + 1),
                }
                epoch_dir = os.path.join(run_dir, "checkpoints", f"phase_{phase_idx:02d}_epoch_{epoch_idx + 1:03d}")
                _save_checkpoint(out_dir=epoch_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, state=state)
                _save_checkpoint(
                    out_dir=os.path.join(run_dir, "last"),
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    state=state,
                )
                _log(f"saved epoch checkpoint {epoch_dir}", log_path)
            if hit_cap:
                reached_cap = True
                stop_next_phase_idx = int(phase_idx)
                stop_next_epoch_idx = int(epoch_idx)
                break
        if reached_cap:
            break

        if partial_frac > 0.0:
            partial_batches = int(math.ceil(partial_frac * len(loader)))
            step, hit_cap = _run_phase_batches(
                cfg=cfg,
                phase=phase,
                loader=loader,
                model=model,
                optimizer=optimizer,
                device=device,
                scaler_ctx=scaler_ctx,
                step=step,
                train_rng=train_rng,
                tokenizer=tokenizer,
                z_token_ids=z_token_ids,
                digit_token_ids=digit_token_ids,
                closing_think_token_id=closing_think_token_id,
                z_and_boundary_allowed=z_and_boundary_allowed,
                log_path=log_path,
                metrics_csv=metrics_csv,
                phase_idx=phase_idx,
                epoch_tag=float(full_epochs) + partial_frac,
                max_batches=partial_batches,
            )
            if hit_cap:
                reached_cap = True
                stop_next_phase_idx = int(phase_idx)
                stop_next_epoch_idx = int(full_epochs)
                break

        if cfg.save_phase_end:
            state = {
                "run_dir": run_dir,
                "global_step": int(step),
                "next_phase_idx": int(phase_idx + 1),
                "next_epoch_idx_in_phase": 0,
            }
            phase_dir = os.path.join(run_dir, "checkpoints", f"phase_{phase_idx:02d}_end")
            _save_checkpoint(out_dir=phase_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, state=state)
            _save_checkpoint(
                out_dir=os.path.join(run_dir, "last"),
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                state=state,
            )
            _log(f"saved phase-end checkpoint {phase_dir}", log_path)

    if reached_cap:
        # Save a recoverable checkpoint at the current boundary state.
        state = {
            "run_dir": run_dir,
            "global_step": int(step),
            "next_phase_idx": int(stop_next_phase_idx),
            "next_epoch_idx_in_phase": int(stop_next_epoch_idx),
        }
        _save_checkpoint(
            out_dir=os.path.join(run_dir, "last"),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            state=state,
        )
        _log(f"stopped by max_steps safety cap at step={step}", log_path)
    else:
        _log("training complete", log_path)

    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3 Curriculum SFT")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
