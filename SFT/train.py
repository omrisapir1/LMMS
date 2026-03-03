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
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SFTConfig
from .dataset import ANSWER_TOKEN, SFTCollator, SFTDataset
from .eval_vllm import evaluate_with_vllm
from .losses import compute_weighted_loss


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
        cfg.base_model_or_checkpoint,
        torch_dtype=_dtype_from_str("bfloat16") if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)

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

    step = 0
    micro = 0
    best_pass = -math.inf
    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    _log(f"run_dir={run_dir}", log_path)
    _log(f"train_size={len(train_records)} eval_size={len(eval_records)}", log_path)

    warmup_hooks = _apply_warmup_freeze(model=model, z_token_ids=z_token_ids, warmup_active=cfg.warmup_steps > 0)

    while step < int(cfg.max_steps):
        for batch in train_loader:
            if step >= int(cfg.max_steps):
                break

            model.train()
            for k in ("input_ids", "attention_mask", "labels", "target_class"):
                batch[k] = batch[k].to(device)

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
                )
                loss = loss_out.total / max(1, int(cfg.gradient_accumulation_steps))

            loss.backward()
            micro += 1

            if micro % max(1, int(cfg.gradient_accumulation_steps)) != 0:
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
                    "L_total": float(loss_out.total.detach().item()),
                    "L_z": float(loss_out.l_z.detach().item()),
                    "L_answer": float(loss_out.l_answer.detach().item()),
                    "L_digits": float(loss_out.l_digits.detach().item()),
                    "z_acc": float(loss_out.z_acc),
                    "digit_exact_match": float(loss_out.digit_exact_match),
                    "avg_z_len": float(mean_z_len),
                    "no_answer_before_kmax": 0.0,
                }
                _append_metrics_csv(metrics_csv, row)
                _log(
                    "step={} L={:.4f} Lz={:.4f} La={:.4f} Ld={:.4f} z_acc={:.3f} d_em={:.3f} z_len={:.2f}".format(
                        step,
                        row["L_total"],
                        row["L_z"],
                        row["L_answer"],
                        row["L_digits"],
                        row["z_acc"],
                        row["digit_exact_match"],
                        row["avg_z_len"],
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
            eval_on_warmup_end = warmup_steps > 0 and step == warmup_steps
            if eval_ready and (eval_on_interval or eval_on_warmup_end):
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
