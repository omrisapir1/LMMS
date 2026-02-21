from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .config import Phase1Config
from .dataset import ANSWER_TOKEN, Phase1Collator, Phase1Dataset
from .model import Phase1CoconutModel

MAX_EVAL_ROWS = 500


@dataclass
class EvalMetrics:
    acc: float
    acc_perm: float
    total: int
    total_perm: int


def make_stage_num_latent_fn(stage: int) -> Callable[[int], int]:
    stage_i = int(stage)
    if stage_i < 1 or stage_i > 8:
        raise ValueError(f"Stage must be in [1,8], got {stage_i}.")

    if stage_i <= 7:
        return lambda K: min(stage_i, max(0, int(K) - 1))
    return lambda K: int(K)


def make_stage_k_filter(stage: int) -> Callable[[int], bool]:
    stage_i = int(stage)
    if stage_i <= 7:
        return lambda K: int(K) > 1
    return lambda K: True


def _extract_digit_token_predictions(
    *,
    logits: torch.Tensor,
    digit_position_indices: torch.Tensor,
) -> torch.Tensor:
    token_preds = logits.argmax(dim=-1)  # [B,T]
    if digit_position_indices.ndim != 2:
        raise ValueError("digit_position_indices must have shape [B,5].")
    if digit_position_indices.shape[0] != token_preds.shape[0]:
        raise ValueError(
            "digit_position_indices batch size does not match logits batch size."
        )
    t = int(token_preds.shape[1])
    if bool((digit_position_indices < 0).any().item()) or bool((digit_position_indices >= t).any().item()):
        bad = (digit_position_indices < 0) | (digit_position_indices >= t)
        bad_count = int(bad.sum().item())
        raise ValueError(
            f"digit_position_indices out of bounds before gather: "
            f"valid range is [0,{t - 1}], found {bad_count} invalid positions."
        )
    return torch.gather(token_preds, 1, digit_position_indices)


def _limit_eval_records(records: Iterable[Dict], limit: int = MAX_EVAL_ROWS):
    if limit <= 0:
        return records
    if hasattr(records, "select") and hasattr(records, "__len__"):
        n = min(int(limit), int(len(records)))
        return records.select(range(n))
    if isinstance(records, list):
        return records[:limit]
    out = []
    for i, row in enumerate(records):
        if i >= limit:
            break
        out.append(row)
    return out


def build_eval_loader(
    *,
    records: Iterable[Dict],
    tokenizer,
    config: Phase1Config,
    stage: int,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, Phase1Dataset, Phase1Collator]:
    records = _limit_eval_records(records, limit=MAX_EVAL_ROWS)
    ds = Phase1Dataset(
        records=records,
        tokenizer=tokenizer,
        num_latent_fn=make_stage_num_latent_fn(stage),
        k_filter=make_stage_k_filter(stage),
        max_thoughts=config.max_thoughts,
        answer_token=ANSWER_TOKEN,
    )
    collator = Phase1Collator(
        tokenizer=tokenizer,
        max_length=config.max_length,
        answer_token=ANSWER_TOKEN,
    )
    num_workers = max(0, int(getattr(config, "eval_dataloader_num_workers", 0)))
    pin_memory = bool(getattr(config, "dataloader_pin_memory", True))
    prefetch_factor = max(1, int(getattr(config, "dataloader_prefetch_factor", 2)))
    loader_kwargs = {
        "batch_size": int(batch_size or config.batch_size),
        "shuffle": False,
        "collate_fn": collator,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        ds,
        **loader_kwargs,
    )
    return loader, ds, collator


def evaluate(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    records: Iterable[Dict],
    config: Phase1Config,
    stage: int,
    device: torch.device,
    seed_base: int,
    batch_size: Optional[int] = None,
) -> EvalMetrics:
    loader, _, _ = build_eval_loader(
        records=records,
        tokenizer=tokenizer,
        config=config,
        stage=stage,
        batch_size=batch_size,
    )

    was_training = model.training
    model.eval()

    total = 0
    correct = 0
    total_perm = 0
    correct_perm = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch["input_ids"].numel() == 0:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            digit_pos = batch["digit_position_indices"].to(device)
            digit_targets = batch["digit_target_token_ids"].to(device)
            latent_count = batch["latent_count"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    compute_aux=True,
                    aux_seed=int(seed_base + batch_idx),
                )

            preds = _extract_digit_token_predictions(
                logits=out.logits_orig,
                digit_position_indices=digit_pos,
            )
            match = (preds == digit_targets).all(dim=1)
            correct += int(match.sum().item())
            total += int(match.numel())

            if out.logits_aux is not None:
                eligible = out.aux_enabled_mask.to(device) & (latent_count >= 2)
                if bool(eligible.any().item()):
                    preds_aux = _extract_digit_token_predictions(
                        logits=out.logits_aux,
                        digit_position_indices=digit_pos,
                    )
                    aux_match = (preds_aux == digit_targets).all(dim=1)
                    correct_perm += int(aux_match[eligible].sum().item())
                    total_perm += int(eligible.sum().item())

    if was_training:
        model.train()

    acc = float(correct) / float(total) if total > 0 else 0.0
    acc_perm = float(correct_perm) / float(total_perm) if total_perm > 0 else 0.0
    return EvalMetrics(
        acc=acc,
        acc_perm=acc_perm,
        total=total,
        total_perm=total_perm,
    )
