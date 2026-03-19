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
from .dataset import (
    HarmonyTemplateBuilder,
    SFTCollator,
    SFTDataset,
    TARGET_ANALYSIS,
    TARGET_ANALYSIS_END,
    TARGET_IGNORE,
    resolve_digit_token_ids,
)
from .eval_vllm import evaluate_with_vllm
from .losses import compute_counterfactual_regularizer, compute_weighted_loss, extract_digit_logits
import gc

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    z_token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in z_tokens]
    if any(i < 0 for i in z_token_ids):
        raise RuntimeError("Failed to resolve one or more Z token ids")

    digit_token_ids = resolve_digit_token_ids(tokenizer)
    harmony = HarmonyTemplateBuilder(tokenizer=tokenizer)

    return {
        "z_token_ids": z_token_ids,
        "digit_token_ids": digit_token_ids,
        "analysis_end_token_id": [int(harmony.analysis_end_token_id)],
    }


def _discover_moe_target_parameters(model, cfg: SFTConfig) -> List[str]:
    if not bool(cfg.lora_enable_moe_target_parameters):
        return []
    substrs = tuple(str(x) for x in cfg.lora_moe_param_substrings)
    out: List[str] = []
    for name, param in model.named_parameters():
        if ".experts." not in name:
            continue
        if param.ndim < 2:
            continue
        if substrs and not any(s in name for s in substrs):
            continue
        out.append(str(name))
    return sorted(set(out))


def _attach_lora(model, cfg: SFTConfig) -> tuple[torch.nn.Module, List[str]]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError("PEFT is required for LoRA training but is not available.") from exc

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }
    task_type_key = str(cfg.lora_task_type).upper().strip()
    if task_type_key not in task_type_map:
        raise ValueError(f"Unsupported lora_task_type={cfg.lora_task_type}")

    target_parameters = _discover_moe_target_parameters(model, cfg)
    lora_cfg = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias=str(cfg.lora_bias),
        task_type=task_type_map[task_type_key],
        target_modules=str(cfg.lora_target_modules),
        target_parameters=target_parameters if target_parameters else None,
    )
    model = get_peft_model(model, lora_cfg)
    return model, target_parameters


class RowDeltaAdapter(torch.nn.Module):
    def __init__(
        self,
        *,
        z_token_ids: Sequence[int],
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if len(z_token_ids) <= 0:
            raise ValueError("z_token_ids must be non-empty")
        z_ids = torch.as_tensor([int(x) for x in z_token_ids], dtype=torch.long, device=device)
        self.register_buffer("z_token_ids", z_ids, persistent=True)

        token_to_compact = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
        token_to_compact[z_ids] = torch.arange(int(z_ids.numel()), dtype=torch.long, device=device)
        self.register_buffer("token_to_compact", token_to_compact, persistent=True)

        self.embedding_row_deltas = torch.nn.Parameter(
            torch.zeros((int(z_ids.numel()), int(hidden_size)), dtype=dtype, device=device)
        )
        self.lm_head_row_deltas = torch.nn.Parameter(
            torch.zeros((int(z_ids.numel()), int(hidden_size)), dtype=dtype, device=device)
        )


def _setup_trainable_parameters(
    model: torch.nn.Module,
    *,
    z_token_ids: Sequence[int],
    device: torch.device,
) -> RowDeltaAdapter:
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)

    emb_w = model.get_input_embeddings().weight
    lm_w = model.get_output_embeddings().weight
    emb_w.requires_grad_(False)
    lm_w.requires_grad_(False)
    adapter = RowDeltaAdapter(
        z_token_ids=z_token_ids,
        vocab_size=int(emb_w.shape[0]),
        hidden_size=int(emb_w.shape[1]),
        dtype=emb_w.dtype,
        device=device,
    )
    return adapter


def _gather_trainable_parameters(model: torch.nn.Module, row_adapter: RowDeltaAdapter) -> List[torch.nn.Parameter]:
    out: List[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    out.extend([p for p in row_adapter.parameters() if p.requires_grad])
    return out


def _log_trainable_summary(
    *,
    model: torch.nn.Module,
    row_adapter: RowDeltaAdapter,
    log_path: str,
) -> None:
    total_params = int(sum(p.numel() for p in model.parameters())) + int(
        sum(p.numel() for p in row_adapter.parameters())
    )
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad)) + int(
        sum(p.numel() for p in row_adapter.parameters() if p.requires_grad)
    )
    trainable_pct = (100.0 * float(trainable_params) / float(max(1, total_params)))

    lora_params = 0
    trainable_names: List[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_names.append(name)
            if "lora_" in name:
                lora_params += int(p.numel())

    emb_rows = int(row_adapter.embedding_row_deltas.numel())
    head_rows = int(row_adapter.lm_head_row_deltas.numel())

    _log(
        f"params total={total_params} trainable={trainable_params} trainable_pct={trainable_pct:.4f}%",
        log_path,
    )
    _log(
        "trainable_breakdown "
        f"lora={int(lora_params)} "
        f"embedding_rows={emb_rows} "
        f"lm_head_rows={head_rows} "
        f"row_local_trainables={emb_rows + head_rows}",
        log_path,
    )

    trainable_names.extend(
        [
            "row_adapter.embedding_row_deltas",
            "row_adapter.lm_head_row_deltas",
        ]
    )
    preview = trainable_names[:40]
    _log(f"trainable_param_names_preview(count={len(trainable_names)}): {preview}", log_path)


def _collect_row_effective_state(
    model: torch.nn.Module,
    row_adapter: RowDeltaAdapter,
) -> Dict[str, torch.Tensor]:
    emb_w = model.get_input_embeddings().weight.detach()
    head_w = model.get_output_embeddings().weight.detach()
    z_ids = row_adapter.z_token_ids.to(emb_w.device)
    emb_base_rows = emb_w.index_select(0, z_ids)
    head_base_rows = head_w.index_select(0, z_ids.to(head_w.device))
    emb_eff = emb_base_rows + row_adapter.embedding_row_deltas.detach().to(emb_base_rows.dtype)
    head_eff = head_base_rows + row_adapter.lm_head_row_deltas.detach().to(head_base_rows.dtype)
    return {
        "z_token_ids": row_adapter.z_token_ids.detach().cpu(),
        "embedding_rows_effective": emb_eff.detach().cpu(),
        "lm_head_rows_effective": head_eff.detach().cpu(),
    }


def _build_base_load_kwargs(cfg: SFTConfig) -> Dict[str, object]:
    try:
        from transformers import Mxfp4Config
    except Exception as exc:
        raise RuntimeError("transformers.Mxfp4Config is required for GPT-OSS loading.") from exc
    quant_cfg = Mxfp4Config(dequantize=bool(cfg.dequantize_mxfp4))
    kwargs: Dict[str, object] = {
        "quantization_config": quant_cfg,
        "attn_implementation": str(cfg.attn_implementation),
    }
    if bool(cfg.force_bfloat16):
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def _build_full_model_for_save(
    *,
    model: torch.nn.Module,
    row_adapter: RowDeltaAdapter,
    tokenizer,
    base_model_name: str,
    cfg: SFTConfig,
    adapter_dir: str,
) -> torch.nn.Module:
    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError("PEFT is required to reconstruct merged full model.") from exc

    load_kwargs = _build_base_load_kwargs(cfg)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    if int(base_model.get_input_embeddings().weight.shape[0]) != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    row_state = _collect_row_effective_state(model, row_adapter)
    z_ids_t = torch.as_tensor(row_state["z_token_ids"], dtype=torch.long)
    emb_rows = torch.as_tensor(
        row_state["embedding_rows_effective"],
        dtype=merged_model.get_input_embeddings().weight.dtype,
    )
    head_rows = torch.as_tensor(
        row_state["lm_head_rows_effective"],
        dtype=merged_model.get_output_embeddings().weight.dtype,
    )
    if int(emb_rows.shape[0]) != int(z_ids_t.numel()):
        raise RuntimeError(
            f"embedding rows mismatch: emb_rows.shape[0]={int(emb_rows.shape[0])} "
            f"vs z_ids={int(z_ids_t.numel())}"
        )
    if int(head_rows.shape[0]) != int(z_ids_t.numel()):
        raise RuntimeError(
            f"lm_head rows mismatch: head_rows.shape[0]={int(head_rows.shape[0])} "
            f"vs z_ids={int(z_ids_t.numel())}"
        )
    emb_hidden = int(merged_model.get_input_embeddings().weight.shape[1])
    head_hidden = int(merged_model.get_output_embeddings().weight.shape[1])
    if int(emb_rows.shape[1]) != emb_hidden:
        raise RuntimeError(
            f"embedding hidden dim mismatch: emb_rows.shape[1]={int(emb_rows.shape[1])} "
            f"vs destination={emb_hidden}"
        )
    if int(head_rows.shape[1]) != head_hidden:
        raise RuntimeError(
            f"lm_head hidden dim mismatch: head_rows.shape[1]={int(head_rows.shape[1])} "
            f"vs destination={head_hidden}"
        )
    emb_dev = merged_model.get_input_embeddings().weight.device
    head_dev = merged_model.get_output_embeddings().weight.device
    merged_model.get_input_embeddings().weight.data.index_copy_(
        0, z_ids_t.to(emb_dev), emb_rows.to(emb_dev)
    )
    merged_model.get_output_embeddings().weight.data.index_copy_(
        0, z_ids_t.to(head_dev), head_rows.to(head_dev)
    )
    return merged_model


def _save_checkpoint_bundle(
    *,
    model: torch.nn.Module,
    row_adapter: RowDeltaAdapter,
    tokenizer,
    out_dir: str,
    base_model_name: str,
    cfg: SFTConfig,
    step: int,
    kind: str,
    best_pass_at_n: Optional[float] = None,
    metric: Optional[float] = None,
) -> str:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    adapter_dir = os.path.join(out_dir, "adapter")
    full_model_dir = os.path.join(out_dir, "full_model")
    tokenizer_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(full_model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Standard PEFT adapter artifact.
    if not hasattr(model, "peft_config"):
        raise RuntimeError("Expected PEFT model for adapter save, but peft_config is missing.")
    model.save_pretrained(adapter_dir)

    # Explicit tokenizer artifact.
    tokenizer.save_pretrained(tokenizer_dir)

    # Build standard full model artifact (no PEFT wrapper required for loading).
    full_model = _build_full_model_for_save(
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        base_model_name=base_model_name,
        cfg=cfg,
        adapter_dir=adapter_dir,
    )
    if int(full_model.get_input_embeddings().weight.shape[0]) != len(tokenizer):
        raise RuntimeError("full_model vocab size does not match tokenizer length")
    full_model.save_pretrained(full_model_dir)
    tokenizer.save_pretrained(full_model_dir)
    del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta_payload: Dict[str, object] = {
        "step": int(step),
        "kind": str(kind),
        "base_model_or_checkpoint": str(base_model_name),
        "save_format": "full_model_plus_lora_adapter",
        "config": asdict(cfg),
    }
    if best_pass_at_n is not None:
        meta_payload["best_pass_at_n"] = float(best_pass_at_n)
    if metric is not None:
        meta_payload["metric"] = float(metric)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    return out_dir


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
    row_adapter: RowDeltaAdapter,
    tokenizer,
    base_model_name: str,
    cfg: SFTConfig,
    step: int,
    best_pass_at_n: float,
) -> str:
    out_dir = os.path.join(run_dir, "last")
    _save_checkpoint_bundle(
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        out_dir=out_dir,
        base_model_name=base_model_name,
        cfg=cfg,
        step=int(step),
        kind="last",
        best_pass_at_n=float(best_pass_at_n),
    )
    return out_dir


def _save_periodic(
    *,
    run_dir: str,
    model,
    row_adapter: RowDeltaAdapter,
    tokenizer,
    base_model_name: str,
    cfg: SFTConfig,
    step: int,
    keep_last_k: int,
) -> str:
    out_dir = os.path.join(run_dir, "checkpoints", f"step_{step:05d}")
    _save_checkpoint_bundle(
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        out_dir=out_dir,
        base_model_name=base_model_name,
        cfg=cfg,
        step=int(step),
        kind="periodic",
    )
    _retain_periodic(os.path.join(run_dir, "checkpoints"), keep_last_k=keep_last_k)
    return out_dir


def _save_best(
    *,
    run_dir: str,
    model,
    row_adapter: RowDeltaAdapter,
    tokenizer,
    base_model_name: str,
    cfg: SFTConfig,
    step: int,
    metric: float,
) -> str:
    out_dir = os.path.join(run_dir, "checkpoints", "best")
    _save_checkpoint_bundle(
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        out_dir=out_dir,
        base_model_name=base_model_name,
        cfg=cfg,
        step=int(step),
        kind="best",
        metric=float(metric),
    )
    return out_dir


def _save_ppo_init(
    *,
    run_dir: str,
    model,
    row_adapter: RowDeltaAdapter,
    tokenizer,
    base_model_name: str,
    cfg: SFTConfig,
) -> str:
    out_dir = os.path.join(run_dir, "ppo_init")
    _save_checkpoint_bundle(
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        out_dir=out_dir,
        base_model_name=base_model_name,
        cfg=cfg,
        step=0,
        kind="ppo_init",
    )
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
    if len(ds) == 0:
        raise ValueError(
            "SFT dataset has 0 usable samples after preprocessing "
            f"(kept={ds.stats.get('kept', 0)}, dropped={ds.stats.get('dropped', 0)}). "
            "Check required fields: question, z_ids, and either valid answer_digits (5 digits) "
            "or answer_int in [0, 99999]."
        )
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


def _get_scheduled_loss_weights(cfg: SFTConfig, step: int) -> tuple[float, float]:
    start = int(cfg.start_weights_steps)
    ramp = int(cfg.goes_up_weights_steps)

    if step < start:
        return float(cfg.w_start_answer), float(cfg.w_start_digits)

    if ramp <= 0:
        return float(cfg.w_end_answer), float(cfg.w_end_digits)

    ramp_end = start + ramp

    if step >= ramp_end:
        return float(cfg.w_end_answer), float(cfg.w_end_digits)

    progress = float(step - start) / float(ramp)

    w_answer = float(cfg.w_start_answer) + progress * (
        float(cfg.w_end_answer) - float(cfg.w_start_answer)
    )
    w_digits = float(cfg.w_start_digits) + progress * (
        float(cfg.w_end_digits) - float(cfg.w_start_digits)
    )

    return w_answer, w_digits


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
    target_class: torch.Tensor,
    z_token_ids: Sequence[int],
    pad_token_id: int,
    cf_min_z_len: int,
    variant_name: str,
    trunc_range: Tuple[float, float],
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_cf = input_ids.clone()
    attention_mask_cf = attention_mask.clone()
    target_class_cf = target_class.clone()

    bsz, _ = input_ids.shape
    eligible_mask = torch.zeros((bsz,), dtype=torch.bool, device=input_ids.device)
    visible_z_counts = torch.zeros((bsz,), dtype=torch.long, device=input_ids.device)

    lo, hi = float(trunc_range[0]), float(trunc_range[1])
    analysis_token_mask = torch.zeros_like(target_class, dtype=torch.bool)
    analysis_token_mask[:, 1:] = target_class[:, :-1] == int(TARGET_ANALYSIS)
    analysis_end_token_mask = torch.zeros_like(target_class, dtype=torch.bool)
    analysis_end_token_mask[:, 1:] = target_class[:, :-1] == int(TARGET_ANALYSIS_END)

    def _token_class_from_target_row(tc_row: torch.Tensor, valid_len: int) -> List[int]:
        out = [int(TARGET_IGNORE)] * int(valid_len)
        for j in range(1, int(valid_len)):
            out[j] = int(tc_row[j - 1].item())
        return out

    def _target_from_token_class(token_class: List[int], row_len: int) -> torch.Tensor:
        t = torch.full((int(row_len),), int(TARGET_IGNORE), dtype=torch.long, device=input_ids.device)
        n = len(token_class)
        for j in range(max(0, n - 1)):
            t[j] = int(token_class[j + 1])
        return t

    def _validate_truncate_row(
        *,
        row_ids: torch.Tensor,
        row_attn: torch.Tensor,
        row_target: torch.Tensor,
        row_pad_id: int,
    ) -> None:
        valid_len_local = int(row_attn.sum().item())
        if valid_len_local <= 0:
            raise RuntimeError("truncate validation failed: empty valid sequence")
        valid_ids = row_ids[:valid_len_local]
        valid_target = row_target[:valid_len_local]
        local_analysis_end = (valid_target == int(TARGET_ANALYSIS_END)).sum().item()
        if int(local_analysis_end) != 1:
            raise RuntimeError(
                f"truncate validation failed: expected 1 analysis-end target, got {int(local_analysis_end)}"
            )
        local_digits = int((valid_target == 3).sum().item())
        if local_digits != 5:
            raise RuntimeError(
                f"truncate validation failed: expected 5 digit targets, got {local_digits}"
            )
        if bool((valid_ids == int(row_pad_id)).any()):
            raise RuntimeError("truncate validation failed: pad token found inside valid region")
        if valid_len_local < int(row_ids.numel()):
            tail = row_ids[valid_len_local:]
            if not bool((tail == int(row_pad_id)).all()):
                raise RuntimeError("truncate validation failed: non-pad token found after valid region")
            attn_tail = row_attn[valid_len_local:]
            if bool((attn_tail != 0).any()):
                raise RuntimeError("truncate validation failed: non-zero attention after valid region")

    for b in range(bsz):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 0:
            continue
        z_pos = analysis_token_mask[b, :valid_len].nonzero(as_tuple=False).view(-1).tolist()
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
            row_len = int(input_ids.shape[1])
            row_ids = input_ids[b]
            row_attn = attention_mask[b]
            row_target = target_class[b]
            valid_row_ids = row_ids[:valid_len]
            token_class = _token_class_from_target_row(tc_row=row_target, valid_len=valid_len)

            analysis_pos = z_pos
            analysis_start = int(analysis_pos[0])
            analysis_end_pos_vec = analysis_end_token_mask[b, :valid_len].nonzero(as_tuple=False).view(-1)
            if int(analysis_end_pos_vec.numel()) != 1:
                raise RuntimeError(
                    f"truncate expects exactly one analysis-end token position; got {int(analysis_end_pos_vec.numel())}"
                )
            analysis_end_pos = int(analysis_end_pos_vec[0].item())
            if analysis_end_pos <= analysis_pos[-1]:
                raise RuntimeError("analysis-end token must come after analysis z span")

            prefix_ids = valid_row_ids[:analysis_start].tolist()
            kept_z_ids = valid_row_ids[analysis_start : analysis_start + keep_k].tolist()
            analysis_end_token_id = int(valid_row_ids[analysis_end_pos].item())
            final_segment_ids = valid_row_ids[analysis_end_pos + 1 :].tolist()

            prefix_tc = token_class[:analysis_start]
            final_tc = token_class[analysis_end_pos + 1 : valid_len]
            if len(final_tc) != len(final_segment_ids):
                raise RuntimeError("truncate internal mismatch: final token_class length mismatch")

            new_valid_ids = prefix_ids + kept_z_ids + [analysis_end_token_id] + final_segment_ids
            new_token_class = prefix_tc + [int(TARGET_ANALYSIS)] * int(keep_k) + [int(TARGET_ANALYSIS_END)] + final_tc
            if len(new_valid_ids) != len(new_token_class):
                raise RuntimeError("truncate internal mismatch: new ids/token_class length mismatch")
            if len(final_segment_ids) > 0:
                if new_valid_ids[-len(final_segment_ids) :] != final_segment_ids:
                    raise RuntimeError("truncate failed to preserve final segment verbatim")

            new_valid_len = len(new_valid_ids)
            if new_valid_len <= 0 or new_valid_len > row_len:
                raise RuntimeError(
                    f"truncate produced invalid new length: {new_valid_len} (row_len={row_len})"
                )

            rebuilt_ids = torch.full((row_len,), int(pad_token_id), dtype=input_ids.dtype, device=input_ids.device)
            rebuilt_attn = torch.zeros((row_len,), dtype=attention_mask.dtype, device=input_ids.device)
            rebuilt_ids[:new_valid_len] = torch.as_tensor(new_valid_ids, dtype=input_ids.dtype, device=input_ids.device)
            rebuilt_attn[:new_valid_len] = 1
            rebuilt_target = _target_from_token_class(new_token_class, row_len=row_len)

            input_ids_cf[b] = rebuilt_ids
            attention_mask_cf[b] = rebuilt_attn
            target_class_cf[b] = rebuilt_target

            _validate_truncate_row(
                row_ids=input_ids_cf[b],
                row_attn=attention_mask_cf[b],
                row_target=target_class_cf[b],
                row_pad_id=int(pad_token_id),
            )
            visible_z_counts[b] = int(keep_k)
        else:
            raise ValueError(f"unknown counterfactual variant: {variant_name}")

    return input_ids_cf, attention_mask_cf, target_class_cf, eligible_mask, visible_z_counts


def _print_startup_examples(
    *,
    tokenizer,
    batch: Dict[str, torch.Tensor],
    z_token_ids: Sequence[int],
    pad_token_id: int,
    cf_min_z_len: int,
    cf_trunc_range: Tuple[float, float],
    seed: int,
) -> None:
    if int(batch["input_ids"].shape[0]) <= 0:
        print("===== STARTUP DEBUG: EMPTY BATCH =====")
        return

    row_idx = 0
    input_ids_row = batch["input_ids"][row_idx].detach().cpu().tolist()
    attention_row = batch["attention_mask"][row_idx].detach().cpu().tolist()
    labels_row = batch["labels"][row_idx].detach().cpu().tolist()
    target_class_row = batch["target_class"][row_idx].detach().cpu().tolist()
    valid_len = int(sum(int(x) for x in attention_row))
    valid_ids = input_ids_row[:valid_len]
    valid_labels = labels_row[:valid_len]
    valid_target_class = target_class_row[:valid_len]

    print("===== STARTUP REGULAR EXAMPLE =====")
    print("decoded_text:")
    print(tokenizer.decode(valid_ids, skip_special_tokens=False))
    print("tokens:")
    print(tokenizer.convert_ids_to_tokens(valid_ids))
    print("input_ids:")
    print(input_ids_row)
    print("target_class:")
    print(valid_target_class)
    print("labels:")
    print(valid_labels)

    def _print_cf_variant(variant_name: str, rng_seed: int) -> None:
        debug_rng = random.Random(int(rng_seed))
        input_ids_cf, attention_cf, target_class_cf, _eligible_mask, _visible_z_counts = _build_counterfactual_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_class=batch["target_class"],
            z_token_ids=z_token_ids,
            pad_token_id=int(pad_token_id),
            cf_min_z_len=int(cf_min_z_len),
            variant_name=str(variant_name),
            trunc_range=cf_trunc_range,
            rng=debug_rng,
        )

        input_ids_cf_row = input_ids_cf[row_idx].detach().cpu().tolist()
        attention_cf_row = attention_cf[row_idx].detach().cpu().tolist()
        target_class_cf_row = target_class_cf[row_idx].detach().cpu().tolist()
        valid_cf_len = int(sum(int(x) for x in attention_cf_row))
        valid_cf_ids = input_ids_cf_row[:valid_cf_len]
        valid_cf_target_class = target_class_cf_row[:valid_cf_len]

        print("===== STARTUP COUNTERFACTUAL EXAMPLE =====")
        print("counterfactual_variant:")
        print(str(variant_name))
        print("decoded_text:")
        print(tokenizer.decode(valid_cf_ids, skip_special_tokens=False))
        print("tokens:")
        print(tokenizer.convert_ids_to_tokens(valid_cf_ids))
        print("input_ids:")
        print(input_ids_cf_row)
        print("attention_mask:")
        print(attention_cf_row)
        print("target_class:")
        print(valid_cf_target_class)

    _print_cf_variant("truncate", int(seed) + 101)
    _print_cf_variant("reverse", int(seed) + 202)
    _print_cf_variant("random", int(seed) + 303)


def _forward_with_row_deltas(
    *,
    model: torch.nn.Module,
    row_adapter: RowDeltaAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)

    compact = row_adapter.token_to_compact.index_select(0, input_ids.view(-1)).view_as(input_ids)
    mask = compact >= 0
    if bool(mask.any()):
        inputs_embeds = inputs_embeds.clone()
        delta = row_adapter.embedding_row_deltas.index_select(0, compact[mask])
        inputs_embeds[mask] = inputs_embeds[mask] + delta

    output_head = model.get_output_embeddings()
    captured_hidden: Dict[str, torch.Tensor] = {}

    def _capture_lm_head_input(_module, args):
        if len(args) == 0:
            raise RuntimeError("LM head pre-hook received empty args")
        captured_hidden["x"] = args[0]

    hook = output_head.register_forward_pre_hook(_capture_lm_head_input)
    try:
        out = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
    finally:
        hook.remove()

    if "x" not in captured_hidden:
        raise RuntimeError("Failed to capture LM head input activations for row-delta correction")
    hidden = captured_hidden["x"]
    corr = torch.matmul(hidden, row_adapter.lm_head_row_deltas.transpose(0, 1))  # [B,L,Z]
    logits = out.logits
    z_ids = row_adapter.z_token_ids.to(logits.device)
    z_logits = logits.index_select(-1, z_ids)
    z_logits = z_logits + corr
    logits.index_copy_(-1, z_ids, z_logits)
    out.logits = logits
    return out


def train(cfg: SFTConfig) -> str:
    if not cfg.base_model_or_checkpoint.strip():
        raise ValueError("config.base_model_or_checkpoint is empty; fill with GPT-OSS model path")
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

    if torch.cuda.is_available():
        device = torch.device(str(cfg.torch_device))
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_or_checkpoint, use_fast=True)
    try:
        from transformers import Mxfp4Config
    except Exception as exc:
        raise RuntimeError(
            "transformers.Mxfp4Config is required for GPT-OSS loading but is unavailable."
        ) from exc

    quant_cfg = Mxfp4Config(dequantize=bool(cfg.dequantize_mxfp4))
    load_kwargs = {
        "quantization_config": quant_cfg,
        "attn_implementation": str(cfg.attn_implementation),
    }
    if bool(cfg.force_bfloat16):
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_or_checkpoint, **load_kwargs)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    token_info = _ensure_sft_tokens(tokenizer, model, cfg.vocab_size)
    z_token_ids = token_info["z_token_ids"]
    digit_token_ids = token_info["digit_token_ids"]
    analysis_end_token_id = token_info["analysis_end_token_id"][0]
    z_and_analysis_end_allowed = list(z_token_ids) + [int(analysis_end_token_id)]

    model, moe_target_parameters = _attach_lora(model, cfg)
    row_adapter = _setup_trainable_parameters(model, z_token_ids=z_token_ids, device=device)
    _log(f"moe_target_parameters_count={len(moe_target_parameters)}", log_path)
    _log_trainable_summary(model=model, row_adapter=row_adapter, log_path=log_path)

    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))

    train_records = _load_hf_records(cfg.train_dataset_name, cfg.train_dataset_split)
    eval_records = _load_hf_records(cfg.eval_dataset_name, cfg.eval_dataset_split)

    train_loader = _build_loader(records=train_records, tokenizer=tokenizer, cfg=cfg, shuffle=True, train=True)

    startup_batch = next(iter(train_loader))
    _print_startup_examples(
        tokenizer=tokenizer,
        batch=startup_batch,
        z_token_ids=z_token_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        cf_min_z_len=int(cfg.cf_min_z_len),
        cf_trunc_range=cfg.cf_trunc_range,
        seed=int(cfg.seed),
    )

    trainable_params = _gather_trainable_parameters(model, row_adapter)
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found after LoRA/row-selective setup.")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_rng = random.Random(int(cfg.seed))

    step = 0
    micro = 0
    best_pass = -math.inf
    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    _log(f"run_dir={run_dir}", log_path)
    _log(f"train_size={len(train_records)} eval_size={len(eval_records)}", log_path)
    _log(
        f"torch_device={device} vllm_cuda_visible_devices={cfg.vllm_cuda_visible_devices}",
        log_path,
    )
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
            cur_w_answer, cur_w_digits = _get_scheduled_loss_weights(cfg, step)

            with scaler_ctx:
                out = _forward_with_row_deltas(
                    model=model,
                    row_adapter=row_adapter,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss_out = compute_weighted_loss(
                    logits=out.logits,
                    labels=batch["labels"],
                    target_class=batch["target_class"],
                    analysis_allowed_ids=z_and_analysis_end_allowed,
                    digit_allowed_ids=digit_token_ids,
                    w_z=cfg.w_z,
                    w_answer=cur_w_answer,
                    w_digits=cur_w_digits,
                    z_label_smoothing=cfg.z_label_smoothing,
                    keep_prob=cfg.keep_prob,
                )
                total_loss = loss_out.total

                if cf_trigger:
                    cf_variant_name = _sample_cf_variant(cfg, train_rng)
                    input_ids_cf, attention_mask_cf, target_class_cf, eligible_mask, visible_z_counts = _build_counterfactual_batch(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        target_class=batch["target_class"],
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
                        out_cf = _forward_with_row_deltas(
                            model=model,
                            row_adapter=row_adapter,
                            input_ids=input_ids_cf,
                            attention_mask=attention_mask_cf,
                        )
                        clean_digit_logits, digit_valid_mask = extract_digit_logits(
                            logits=out.logits,
                            target_class=batch["target_class"],
                            digit_allowed_ids=digit_token_ids,
                        )
                        cf_digit_logits, cf_digit_valid_mask = extract_digit_logits(
                            logits=out_cf.logits,
                            target_class=target_class_cf,
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



            if step % int(cfg.log_interval_steps) == 0:
                mean_z_len = float(batch["z_lens"].float().mean().item())
                row = {
                    "step": float(step),
                    "L_total": float(total_loss.detach().item()),
                    "L_analysis": float(loss_out.l_analysis.detach().item()),
                    "L_analysis_end": float(loss_out.l_analysis_end.detach().item()),
                    "L_digits": float(loss_out.l_digits.detach().item()),
                    "analysis_acc": float(loss_out.analysis_acc),
                    "digit_exact_match": float(loss_out.digit_exact_match),
                    "avg_z_len": float(mean_z_len),
                    "no_analysis_end_before_kmax": 0.0,
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
                    "w_answer": float(cur_w_answer),
                    "w_digits": float(cur_w_digits),
                    "batch_seen": float(int(batch.get("batch_seen", batch["input_ids"].shape[0]))),
                    "batch_kept": float(int(batch.get("batch_kept", batch["input_ids"].shape[0]))),
                    "batch_dropped_invalid_after_clip": float(int(batch.get("batch_dropped_invalid_after_clip", 0))),
                    "total_dropped_invalid_after_clip": float(int(batch.get("total_dropped_invalid_after_clip", 0))),
                }
                _append_metrics_csv(metrics_csv, row)
                _log(
                    "step={} L={:.4f} Lan={:.4f} Lae={:.4f} Ld={:.4f} an_acc={:.3f} d_em={:.3f} z_len={:.2f} wa={:.4f} wd={:.4f} clip_drop_batch={} clip_drop_total={} cf_on={} cf_applied={} cf_variant={} cf_loss={:.4f} cf_kl={:.4f} cf_H={:.4f}".format(
                        step,
                        row["L_total"],
                        row["L_analysis"],
                        row["L_analysis_end"],
                        row["L_digits"],
                        row["analysis_acc"],
                        row["digit_exact_match"],
                        row["avg_z_len"],
                        row["w_answer"],
                        row["w_digits"],
                        int(row["batch_dropped_invalid_after_clip"]),
                        int(row["total_dropped_invalid_after_clip"]),
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
                    row_adapter=row_adapter,
                    tokenizer=tokenizer,
                    base_model_name=cfg.base_model_or_checkpoint,
                    cfg=cfg,
                    step=step,
                    best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
                )
                _log(f"saved last checkpoint at step={step}", log_path)

            if step % int(cfg.save_every_steps) == 0:
                p = _save_periodic(
                    run_dir=run_dir,
                    model=model,
                    row_adapter=row_adapter,
                    tokenizer=tokenizer,
                    base_model_name=cfg.base_model_or_checkpoint,
                    cfg=cfg,
                    step=step,
                    keep_last_k=int(cfg.keep_last_k),
                )
                _log(f"saved periodic checkpoint {p}", log_path)

            eval_on_interval = step % int(cfg.eval_interval_steps) == 0
            if eval_on_interval:
                eval_model_path = _save_last(
                    run_dir=run_dir,
                    model=model,
                    row_adapter=row_adapter,
                    tokenizer=tokenizer,
                    base_model_name=cfg.base_model_or_checkpoint,
                    cfg=cfg,
                    step=step,
                    best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
                )
                eval_model_path = os.path.join(eval_model_path, "full_model")
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
                    vllm_cuda_visible_devices=cfg.vllm_cuda_visible_devices,
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
                        "eval_no_analysis_end_before_kmax": float(metrics.no_answer_before_kmax_rate),
                    },
                )

                if cfg.save_best and metrics.pass_at_n > best_pass:
                    best_pass = metrics.pass_at_n
                    best_path = _save_best(
                        run_dir=run_dir,
                        model=model,
                        row_adapter=row_adapter,
                        tokenizer=tokenizer,
                        base_model_name=cfg.base_model_or_checkpoint,
                        cfg=cfg,
                        step=step,
                        metric=best_pass,
                    )
                    _log(f"new best pass@{cfg.pass_at_n}={best_pass:.4f}; saved {best_path}", log_path)

    _save_last(
        run_dir=run_dir,
        model=model,
        row_adapter=row_adapter,
        tokenizer=tokenizer,
        base_model_name=cfg.base_model_or_checkpoint,
        cfg=cfg,
        step=step,
        best_pass_at_n=best_pass if best_pass > -math.inf else 0.0,
    )
    if cfg.save_ppo_init:
        p = _save_ppo_init(
            run_dir=run_dir,
            model=model,
            row_adapter=row_adapter,
            tokenizer=tokenizer,
            base_model_name=cfg.base_model_or_checkpoint,
            cfg=cfg,
        )
        _log(f"saved ppo_init snapshot at {p}", log_path)

    _log("training complete", log_path)
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3 SFT over discrete latent Z-programs")
    p.add_argument("--base_model_or_checkpoint", type=str, default="openai/gpt-oss-20b")
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
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--torch_device", type=str, default="cuda:0")
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--dequantize_mxfp4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_bfloat16", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_enable_moe_target_parameters", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--lora_moe_param_substrings",
        type=str,
        default="gate_up_proj,down_proj,up_proj,gate_proj,w1,w2,w3",
    )

    p.add_argument("--z_label_smoothing", type=float, default=0.05)
    p.add_argument("--w_z", type=float, default=0.1)
    p.add_argument("--w_start_answer", type=float, default=0.05)
    p.add_argument("--w_start_digits", type=float, default=0.1)
    p.add_argument("--w_end_answer", type=float, default=0.5)
    p.add_argument("--w_end_digits", type=float, default=1.0)
    p.add_argument("--start_weights_steps", type=int, default=500)
    p.add_argument("--goes_up_weights_steps", type=int, default=1500)
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
    p.add_argument("--vllm_cuda_visible_devices", type=str, default="1")

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
    lora_moe_substrings = tuple(
        s.strip() for s in str(args.lora_moe_param_substrings).split(",") if s.strip()
    )
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
        max_length=args.max_length,
        torch_device=args.torch_device,
        attn_implementation=args.attn_implementation,
        dequantize_mxfp4=bool(args.dequantize_mxfp4),
        force_bfloat16=bool(args.force_bfloat16),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_task_type=args.lora_task_type,
        lora_target_modules=args.lora_target_modules,
        lora_enable_moe_target_parameters=bool(args.lora_enable_moe_target_parameters),
        lora_moe_param_substrings=lora_moe_substrings,
        z_label_smoothing=args.z_label_smoothing,
        w_z=args.w_z,
        w_start_answer=args.w_start_answer,
        w_start_digits=args.w_start_digits,
        w_end_answer=args.w_end_answer,
        w_end_digits=args.w_end_digits,
        start_weights_steps=args.start_weights_steps,
        goes_up_weights_steps=args.goes_up_weights_steps,
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
        vllm_cuda_visible_devices=args.vllm_cuda_visible_devices,
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
