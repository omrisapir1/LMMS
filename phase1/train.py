from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Phase1Config
from .dataset import ANSWER_TOKEN, LATENT_TOKEN, Phase1Collator, Phase1Dataset, format_answer
from .eval import evaluate, make_stage_k_filter, make_stage_num_latent_fn
from .loss import AnswerLoss, permutation_sensitivity_loss
from .model import PERM_TRUNCATE_RATIO, Phase1CoconutModel
from .stage_manager import StageManager


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(name: str) -> torch.dtype:
    normalized = str(name).lower()
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in table:
        raise ValueError(f"Unsupported torch dtype string: {name}")
    return table[normalized]


def _log(msg: str, log_path: Optional[str] = None) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts} | {msg}"
    print(line)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _ensure_special_tokens(tokenizer, model) -> int:
    to_add: List[str] = []
    vocab = tokenizer.get_vocab()
    if LATENT_TOKEN not in vocab:
        to_add.append(LATENT_TOKEN)
    if ANSWER_TOKEN not in vocab:
        to_add.append(ANSWER_TOKEN)

    added = 0
    if to_add:
        added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added += int(tokenizer.add_special_tokens({"pad_token": "<|pad|>"}))

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return int(added)


def _verify_digit_tokens(tokenizer) -> List[int]:
    digit_ids: List[int] = []
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                "Digit tokenization requirement failed: "
                f"tokenizer.encode('{d}', add_special_tokens=False) -> {ids}"
            )
        digit_ids.append(int(ids[0]))
    return digit_ids


def _build_loader(
    *,
    records: Iterable[Dict],
    tokenizer,
    config: Phase1Config,
    stage: int,
    shuffle: bool,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, Phase1Dataset, Phase1Collator]:
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
    loader = DataLoader(
        ds,
        batch_size=int(batch_size or config.batch_size),
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
    )
    return loader, ds, collator


def _save_checkpoint(
    *,
    model: Phase1CoconutModel,
    tokenizer,
    config: Phase1Config,
    stage_manager: StageManager,
    microbatch: int,
    optimizer_steps: int,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    os.makedirs(config.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(config.log_dir, f"step_{optimizer_steps:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.base_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    payload = {
        "config": asdict(config),
        "microbatch": int(microbatch),
        "optimizer_steps": int(optimizer_steps),
        "stage": int(stage_manager.current_stage),
        "best_val_acc": stage_manager.best_val_acc,
        "no_improve_count": int(stage_manager.no_improve_count),
        "perm_truncate_ratio": float(PERM_TRUNCATE_RATIO),
        "metrics": metrics or {},
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return ckpt_dir


def _preview_text(text: str) -> str:
    return text



def _print_stage_enter_example(
    *,
    stage: int,
    loader: DataLoader,
    dataset: Phase1Dataset,
    tokenizer,
    answer_loss: AnswerLoss,
    downsample_seed: int,
    log_path: str,
) -> None:
    if len(dataset) == 0:
        _log(
            f"[Stage {stage} Entry Example] ERROR: stage dataset is empty; cannot print example.",
            log_path,
        )
        return

    max_tries = 10
    tries = 0
    candidate = None
    loader_iter = iter(loader)

    while tries < max_tries:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        if batch["input_ids"].numel() == 0:
            continue
        # Deterministic choice: always inspect item index 0 from each batch.
        row = 0
        tries += 1
        input_ids_row_all = batch["input_ids"][row]
        attn_row = batch["attention_mask"][row]
        seq_len = int(attn_row.sum().item())
        input_ids_row = input_ids_row_all[:seq_len].tolist()
        answer_positions = [
            i for i, tid in enumerate(input_ids_row) if int(tid) == int(dataset.answer_token_id)
        ]
        if len(answer_positions) != 1:
            _log(
                f"[Stage {stage} Entry Example] WARNING: sample try {tries} has "
                f"{len(answer_positions)} <ANSWER> tokens; trying another sample.",
                log_path,
            )
            continue
        answer_pos = int(answer_positions[0])
        if answer_pos + 5 >= seq_len:
            _log(
                f"[Stage {stage} Entry Example] WARNING: sample try {tries} truncated around <ANSWER> "
                f"(need 5 digits after it); trying another sample.",
                log_path,
            )
            continue
        candidate = (batch, row, answer_pos, seq_len)
        if candidate is not None:
            break

    if candidate is None:
        _log(
            f"[Stage {stage} Entry Example] ERROR: failed to find valid sample with "
            f"<ANSWER>+5 digits after {max_tries} tries.",
            log_path,
        )
        return

    batch, row, answer_pos, seq_len = candidate
    sample_idx = int(batch["sample_idx"][row].item())
    raw = dataset.samples[sample_idx]
    K = int(batch["K"][row].item())
    num_latent = int(max(0, min(dataset.num_latent_fn(K), K)))
    answer_text = format_answer(
        thoughts=raw.thoughts,
        K=raw.K,
        num_latent=num_latent,
        answer_token=dataset.answer_token,
    )

    input_ids_row = batch["input_ids"][row, :seq_len].tolist()
    labels_row = batch["labels"][row]
    digit_positions = batch["digit_position_indices"][row].tolist()
    digit_values = batch["digit_values"][row]
    digit_str = "".join(str(int(x)) for x in digit_values.tolist())

    prompt_answer_ids = input_ids_row[: answer_pos + 1]
    prompt_answer_decoded = tokenizer.decode(prompt_answer_ids, skip_special_tokens=False)
    around_start = max(0, answer_pos - 12)
    around_end = min(seq_len, answer_pos + 6)
    around_ids = input_ids_row[around_start:around_end]
    around_decoded = tokenizer.decode(around_ids, skip_special_tokens=False)

    label_digit_ids = [int(labels_row[int(p)].item()) for p in digit_positions]
    keep_mask_5 = answer_loss.sample_digit_keep_mask(
        digit_values=digit_values.unsqueeze(0),
        downsample_zeros=True,
        seed=int(downsample_seed),
    )[0]
    effective_loss_mask_values = [int(x) for x in keep_mask_5.to(torch.int).tolist()]

    keep_drop_notes: List[str] = []
    for i, d in enumerate(digit_values.tolist()):
        if int(d) == 0:
            keep_drop_notes.append(
                f"pos{i}: {'kept' if bool(keep_mask_5[i].item()) else 'dropped'} by keep_prob={answer_loss.keep_prob[i]:.2f}"
            )
        else:
            keep_drop_notes.append(f"pos{i}: kept (non-zero digit {int(d)})")

    print("")
    print(f"===== STAGE ENTER EXAMPLE (Stage {stage}) =====")
    print(f"stage number: {stage}")
    print(f"K and num_latent used for this sample: K={K}, num_latent={num_latent}")
    full_decoded = tokenizer.decode(input_ids_row, skip_special_tokens=False)

    print("FULL DECODED SEQUENCE (prompt + answer_text + digits):")
    print(full_decoded)
    print("===== END STAGE ENTER EXAMPLE =====")
    print("")


def train(config: Phase1Config, *, max_optimizer_steps: int = 0) -> None:
    _set_seed(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)
    log_path = os.path.join(config.log_dir, "train.log")

    _log(f"Loading dataset '{config.dataset_name}' ...", log_path)
    hf_ds = load_dataset(config.dataset_name)
    train_records = hf_ds[config.dataset_train_split]
    eval_records = hf_ds[config.dataset_eval_split]
    _log(
        f"Dataset loaded: train={len(train_records)} eval={len(eval_records)}",
        log_path,
    )

    dtype = _dtype_from_str(config.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
    )
    _ensure_special_tokens(tokenizer, base_model)
    digit_token_ids = _verify_digit_tokens(tokenizer)

    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    if latent_token_id is None or int(latent_token_id) < 0:
        raise RuntimeError(f"Failed to resolve token id for {LATENT_TOKEN}")
    answer_token_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    if answer_token_id is None or int(answer_token_id) < 0:
        raise RuntimeError(f"Failed to resolve token id for {ANSWER_TOKEN}")

    model = Phase1CoconutModel(
        base_model=base_model,
        latent_token_id=int(latent_token_id),
        perm_truncate_ratio=PERM_TRUNCATE_RATIO,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    answer_loss = AnswerLoss(keep_prob=config.keep_prob)
    stage_manager = StageManager(
        min_delta=config.min_delta,
        stage_patience=config.stage_patience,
        max_steps_first_stage=config.max_steps_first_stage,
    )
    printed_stage_examples: set[int] = set()

    train_loader: DataLoader
    train_ds: Phase1Dataset
    train_collator: Phase1Collator
    train_iter = None

    def _enter_stage(stage: int, reason: str, *, seed_offset: int) -> None:
        nonlocal train_loader, train_ds, train_collator, train_iter
        train_loader, train_ds, train_collator = _build_loader(
            records=train_records,
            tokenizer=tokenizer,
            config=config,
            stage=stage,
            shuffle=True,
        )
        train_iter = iter(train_loader)
        _log(f"Stage {stage} loader: samples={len(train_ds)} (reason={reason})", log_path)
        if stage not in printed_stage_examples:
            _print_stage_enter_example(
                stage=stage,
                loader=train_loader,
                dataset=train_ds,
                tokenizer=tokenizer,
                answer_loss=answer_loss,
                downsample_seed=int(config.seed * 1_000_003 + seed_offset),
                log_path=log_path,
            )
            printed_stage_examples.add(stage)

    _enter_stage(stage=int(stage_manager.current_stage), reason="initial", seed_offset=0)

    microbatch = 0
    optimizer_steps = 0
    running_total = 0.0
    running_answer = 0.0
    running_perm = 0.0
    running_count = 0

    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)

    while True:
        if max_optimizer_steps > 0 and optimizer_steps >= max_optimizer_steps:
            _log(f"Reached max_optimizer_steps={max_optimizer_steps}; stopping.", log_path)
            break

        try:
            assert train_iter is not None
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch["input_ids"].numel() == 0:
            continue

        microbatch += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        digit_mask = batch["digit_mask"].to(device)
        digit_pos = batch["digit_position_indices"].to(device)
        digit_values = batch["digit_values"].to(device)
        latent_count = batch["latent_count"].to(device)

        compute_perm = (
            config.permutation_loss_interval_batches > 0
            and microbatch % config.permutation_loss_interval_batches == 0
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            compute_aux=compute_perm,
            aux_seed=int(config.seed * 10_000_019 + microbatch),
        )

        answer_out = answer_loss.compute(
            logits=out.logits_orig,
            labels=labels,
            digit_mask=digit_mask,
            digit_position_indices=digit_pos,
            digit_values=digit_values,
            downsample_zeros=True,
            seed=int(config.seed * 1_000_003 + microbatch),
        )

        perm_loss = out.logits_orig.new_zeros(())
        if compute_perm and out.logits_aux is not None:
            eligible = (latent_count >= 2) & out.aux_enabled_mask.to(device)
            perm_loss = permutation_sensitivity_loss(
                logits_orig=out.logits_orig,
                logits_aux=out.logits_aux,
                digit_position_indices=digit_pos,
                digit_token_ids=digit_token_ids_t,
                eligible_mask=eligible,
            )

        total_loss = answer_out.loss + perm_loss
        scaled_loss = total_loss / float(max(1, config.gradient_accumulation_steps))
        scaled_loss.backward()

        running_total += float(total_loss.detach().item())
        running_answer += float(answer_out.loss.detach().item())
        running_perm += float(perm_loss.detach().item())
        running_count += 1

        if microbatch % int(max(1, config.gradient_accumulation_steps)) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            forced = stage_manager.force_stage1_exit_if_needed(optimizer_steps)
            if forced:
                current_stage = stage_manager.current_stage
                _log(
                    f"Stage advanced to {current_stage} (forced by max_steps_first_stage={config.max_steps_first_stage}).",
                    log_path,
                )
                _enter_stage(
                    stage=current_stage,
                    reason="forced-stage1-exit",
                    seed_offset=optimizer_steps,
                )

        if config.logg_loss_interval_batches > 0 and microbatch % config.logg_loss_interval_batches == 0:
            denom = max(1, running_count)
            _log(
                f"stage={stage_manager.current_stage} microbatch={microbatch} opt_steps={optimizer_steps} "
                f"loss={running_total/denom:.6f} answer_loss={running_answer/denom:.6f} "
                f"perm_loss={running_perm/denom:.6f}",
                log_path,
            )
            running_total = 0.0
            running_answer = 0.0
            running_perm = 0.0
            running_count = 0

        if config.eval_interval_batches > 0 and microbatch % config.eval_interval_batches == 0:
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                records=eval_records,
                config=config,
                stage=stage_manager.current_stage,
                device=device,
                seed_base=int(config.seed * 100_003 + microbatch),
                batch_size=config.batch_size,
            )
            _log(
                f"eval stage={stage_manager.current_stage} microbatch={microbatch} "
                f"acc={metrics.acc:.6f} acc_perm={metrics.acc_perm:.6f} "
                f"n={metrics.total} n_perm={metrics.total_perm}",
                log_path,
            )

            update = stage_manager.update(
                val_acc=metrics.acc,
                optimizer_steps=optimizer_steps,
            )
            if update.advanced:
                reason = "forced-step" if update.forced_stage1_exit else "patience"
                _log(
                    f"Stage advanced to {update.stage} ({reason}); "
                    f"best={update.best_val_acc} patience_count={update.patience_counter}",
                    log_path,
                )
                _enter_stage(
                    stage=update.stage,
                    reason=f"eval-{reason}",
                    seed_offset=optimizer_steps + microbatch,
                )
            else:
                _log(
                    f"Stage {update.stage} status: improved={update.improved} "
                    f"patience_count={update.patience_counter} best={update.best_val_acc}",
                    log_path,
                )

            ckpt = _save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                config=config,
                stage_manager=stage_manager,
                microbatch=microbatch,
                optimizer_steps=optimizer_steps,
                metrics={"acc": metrics.acc, "acc_perm": metrics.acc_perm},
            )
            _log(f"Checkpoint saved: {ckpt}", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 training")
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="Optional hard stop. 0 means run continuously.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(Phase1Config(), max_optimizer_steps=int(args.max_optimizer_steps))


if __name__ == "__main__":
    main()
