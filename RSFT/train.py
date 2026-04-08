from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.hf_rollout import HFRolloutEngine
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single, validate_single_token
from PPO.vllm_rollout import VLLMRolloutEngine
from RSFT.config import Config, DEFAULT_SET_ALLOWED_PREFIXES
from RSFT.dataset import PromptExample, load_hf_records, make_digit_id_to_value_map, prepare_prompt_examples, sample_unique_prompt_batch
from RSFT.eval_vllm import evaluate_with_rollout_engine
from RSFT.logic import (
    RoundTrace,
    build_training_example,
    collate_training_examples,
    compute_rsft_losses,
    decode_digit_tokens,
    extract_z_before_answer_from_row,
    mean_or_zero,
)

METRICS_FIELDS: List[str] = [
    "step",
    "update_step",
    "prompts_sampled",
    "total_sequences",
    "accepted_prompts",
    "accepted_rate",
    "avg_rounds_per_accepted",
    "avg_failed_rounds_before_success",
    "mean_accepted_z_len_per_round",
    "l_z_ans",
    "l_digits",
    "l_verify",
    "total_loss",
    "grad_norm",
    "rollout_time",
    "train_time",
    "sync_time",
    "train_mode",
    "warmup_steps",
    "trainable_params",
    "skipped_optimizer",
    "rollout_log_path",
    "evaluated_questions",
    "greedy_exact",
    "pass_at_n",
    "mean_z_length",
    "no_answer_before_kmax_rate",
    "eval_time",
]

def _log(msg: str, log_path: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts} | {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_cfg(cfg: Config) -> None:
    if int(cfg.rollout.max_rounds) < 1:
        raise ValueError("rollout.max_rounds must be >= 1")
    if float(cfg.loss.w_z_ans) < 0.0:
        raise ValueError("loss.w_z_ans must be >= 0")
    if float(cfg.loss.w_digits) < 0.0:
        raise ValueError("loss.w_digits must be >= 0")
    if float(cfg.loss.w_verify) < 0.0:
        raise ValueError("loss.w_verify must be >= 0")
    if int(cfg.train.warmup_steps) < 0:
        raise ValueError("train.warmup_steps must be >= 0")


def _apply_override(cfg: Config, key: str, raw_value: str) -> None:
    if not any(key.startswith(prefix) for prefix in DEFAULT_SET_ALLOWED_PREFIXES):
        raise ValueError(f"Unsupported override key '{key}'")

    try:
        value = ast.literal_eval(raw_value)
    except Exception:
        value = raw_value

    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise ValueError(f"Unknown override path '{key}'")
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"Unknown override path '{key}'")
    setattr(obj, parts[-1], value)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-4 RSFT trainer")
    p.add_argument("--set", action="append", default=[], help="Override config, e.g. train.lr=3e-5")
    return p


def _make_run_dir(output_root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, ts)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def _save_model_dir(model, tokenizer, out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _retain_periodic(ckpt_root: str, keep_last_k: int) -> None:
    rows: List[Tuple[int, str]] = []
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
        rows.append((step, path))
    rows.sort(key=lambda x: x[0], reverse=True)
    for _, path in rows[int(keep_last_k) :]:
        shutil.rmtree(path)


def _save_periodic(*, run_dir: str, model, tokenizer, step: int, keep_last: int) -> str:
    out_dir = os.path.join(run_dir, "checkpoints", f"step_{step:05d}")
    _save_model_dir(model, tokenizer, out_dir)
    _retain_periodic(os.path.join(run_dir, "checkpoints"), keep_last_k=keep_last)
    return out_dir


def _save_last(*, run_dir: str, model, tokenizer, cfg: Config, step: int) -> str:
    out_dir = os.path.join(run_dir, "last")
    _save_model_dir(model, tokenizer, out_dir)
    payload = {"step": int(step), "config": asdict(cfg)}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_dir


def _build_optimizer(*, model, cfg: Config):
    if bool(cfg.train.optimizer_8bit):
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "8-bit optimizer requested but bitsandbytes is not available. "
                "Set train.optimizer_8bit=False or install bitsandbytes."
            ) from exc
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=float(cfg.train.lr),
            betas=tuple(float(x) for x in cfg.train.betas),
            eps=float(cfg.train.eps),
            weight_decay=float(cfg.train.weight_decay),
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        betas=tuple(float(x) for x in cfg.train.betas),
        eps=float(cfg.train.eps),
        weight_decay=float(cfg.train.weight_decay),
    )


def _set_optimizer_weight_decay(optimizer, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        group["weight_decay"] = float(weight_decay)


def _assert_optimizer_weight_decay(optimizer, expected: float) -> None:
    exp = float(expected)
    for i, group in enumerate(optimizer.param_groups):
        got = float(group.get("weight_decay", 0.0))
        if abs(got - exp) > 1e-12:
            raise RuntimeError(f"Optimizer weight_decay mismatch in group {i}: got {got}, expected {exp}")


def _prepare_tokenizer_and_model(cfg: Config):
    if not str(cfg.model.init_ckpt).strip():
        raise ValueError("model.init_ckpt must be set (no auto-guessing)")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.init_ckpt), trust_remote_code=bool(cfg.model.trust_remote_code))
    model_kwargs = {
        "trust_remote_code": bool(cfg.model.trust_remote_code),
    }
    if bool(cfg.train.use_bf16):
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.init_ckpt),
        **model_kwargs,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    special_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [str(cfg.model.finalize_token), str(cfg.model.retry_token)]}
    )
    if int(special_added) > 0:
        model.resize_token_embeddings(len(tokenizer))

    answer_token_id = int(resolve_answer_token_id(tokenizer, answer_token=str(cfg.model.answer_token)))
    validate_answer_token_single(tokenizer, str(cfg.model.answer_token), answer_token_id)

    finalize_ids = tokenizer.encode(str(cfg.model.finalize_token), add_special_tokens=False)
    retry_ids = tokenizer.encode(str(cfg.model.retry_token), add_special_tokens=False)
    if len(finalize_ids) != 1 or len(retry_ids) != 1:
        raise RuntimeError("Verify tokens must tokenize to exactly one token each")
    finalize_token_id = int(finalize_ids[0])
    retry_token_id = int(retry_ids[0])
    validate_single_token(tokenizer, str(cfg.model.finalize_token), finalize_token_id, label="Verify")
    validate_single_token(tokenizer, str(cfg.model.retry_token), retry_token_id, label="Verify")
    if finalize_token_id == retry_token_id:
        raise RuntimeError("<FINALIZE> and <RETRY> must map to distinct token ids")

    z_token_ids, _style = introspect_z_token_ids_and_style(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found in tokenizer vocab")

    digit_token_ids = resolve_digit_token_ids(tokenizer)
    digit_id_to_val = make_digit_id_to_value_map(digit_token_ids)

    return (
        tokenizer,
        model,
        answer_token_id,
        z_token_ids,
        digit_token_ids,
        digit_id_to_val,
        finalize_token_id,
        retry_token_id,
    )


def _make_rollout_engine(
    *,
    cfg: Config,
    tokenizer,
    init_ckpt_ref: str,
    answer_token_id: int,
    z_token_ids: Sequence[int],
    digit_token_ids: Sequence[int],
    verify_token_ids: Sequence[int],
    finalize_token_id: int,
    retry_token_id: int,
    run_dir: str,
    logger,
):
    backend = str(cfg.rollout.backend).strip().lower()
    if backend == "vllm":
        engine_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
        engine_kwargs.setdefault("tensor_parallel_size", int(cfg.rollout.vllm_tp_size))
        engine_kwargs.setdefault("gpu_memory_utilization", float(cfg.rollout.gpu_memory_utilization))
        engine_kwargs.setdefault("cuda_visible_devices", str(cfg.rollout.vllm_cuda_visible_devices))
        return VLLMRolloutEngine(
            init_ckpt=str(init_ckpt_ref),
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=list(z_token_ids),
            digit_allowed_token_ids=list(digit_token_ids),
            verify_allowed_token_ids=list(verify_token_ids),
            finalize_token_id=int(finalize_token_id),
            retry_token_id=int(retry_token_id),
            trust_remote_code=bool(cfg.model.trust_remote_code),
            engine_kwargs=engine_kwargs,
            output_dir=run_dir,
            tmp_ckpt_dir=os.path.join(run_dir, "tmp_vllm_ckpt"),
            sync_every=int(cfg.rollout.sync_every_n_steps),
            seed=int(cfg.rollout.vllm_seed if cfg.rollout.vllm_seed is not None else cfg.train.seed),
            logger=logger,
        )
    if backend == "hf":
        return HFRolloutEngine(
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=list(z_token_ids),
            digit_allowed_token_ids=list(digit_token_ids),
            verify_allowed_token_ids=list(verify_token_ids),
            finalize_token_id=int(finalize_token_id),
            retry_token_id=int(retry_token_id),
            sync_every=int(cfg.rollout.sync_every_n_steps),
            logger=logger,
        )
    raise ValueError(f"Unsupported rollout.backend={cfg.rollout.backend!r}; expected 'vllm' or 'hf'")


def _append_metrics_csv(path: str, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_metrics_jsonl(path: str, row: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _chunk_examples(rows: Sequence[Dict[str, object]], chunk_size: int) -> List[List[Dict[str, object]]]:
    if len(rows) <= 0:
        return []
    k = max(1, int(chunk_size))
    return [list(rows[i : i + k]) for i in range(0, len(rows), k)]


def _next_unique_batch(
    *,
    examples: Sequence[PromptExample],
    ordered_indices: List[int],
    cursor: int,
    batch_size: int,
    step_seen_questions: set[str],
    rng: random.Random,
) -> tuple[List[PromptExample], int]:
    if len(examples) == 0:
        return [], cursor

    selected: List[PromptExample] = []
    cur = int(cursor)
    wraps = 0
    while len(selected) < int(batch_size) and len(step_seen_questions) < len(examples):
        if cur >= len(ordered_indices):
            rng.shuffle(ordered_indices)
            cur = 0
            wraps += 1
            if wraps > 1:
                break
        got, cur = sample_unique_prompt_batch(
            examples=examples,
            ordered_indices=ordered_indices,
            cursor=cur,
            batch_size=int(batch_size) - len(selected),
            seen_questions=step_seen_questions,
        )
        if not got and wraps > 0:
            break
        selected.extend(got)

    return selected, cur


def _build_round_distribution(accepted_rounds: Sequence[int], max_rounds: int) -> Dict[str, float]:
    if len(accepted_rounds) == 0:
        return {f"accepted_round_frac_{r}": 0.0 for r in range(1, int(max_rounds) + 1)}
    ctr = Counter(int(x) for x in accepted_rounds)
    denom = float(len(accepted_rounds))
    return {f"accepted_round_frac_{r}": float(ctr.get(r, 0)) / denom for r in range(1, int(max_rounds) + 1)}


def _resolve_lm_head_weight(model) -> torch.nn.Parameter:
    out_emb = model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        return out_emb.weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    raise RuntimeError("Could not resolve lm_head/output embedding weight")


def _count_trainable_params(model) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad)))


def _count_effective_warmup_trainable_params(
    *,
    warmup_runtime: Dict[str, object],
    verify_token_ids: Sequence[int],
) -> int:
    inp_p = warmup_runtime.get("inp_param", None)
    if not isinstance(inp_p, torch.nn.Parameter):
        return 0
    row_count = int(len([int(x) for x in verify_token_ids]))
    inp_dim = int(inp_p.shape[1]) if inp_p.ndim >= 2 else 0
    total = row_count * inp_dim
    tied = bool(warmup_runtime.get("tied_weights", False))
    if not tied:
        lm_p = warmup_runtime.get("lm_param", None)
        if isinstance(lm_p, torch.nn.Parameter) and lm_p.ndim >= 2:
            total += row_count * int(lm_p.shape[1])
    return int(total)


def _mask_non_verify_rows(grad: torch.Tensor, verify_token_ids: Sequence[int]) -> torch.Tensor:
    if grad is None:
        return grad
    if grad.ndim < 2:
        raise RuntimeError("Expected row-wise parameter grad tensor with ndim >= 2")
    out = grad.clone()
    out.zero_()
    row_ids = [int(x) for x in verify_token_ids]
    out[row_ids] = grad[row_ids]
    return out


def _set_full_train_mode(model) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def _assert_full_train_mode_restored(model) -> None:
    for name, p in model.named_parameters():
        if not bool(p.requires_grad):
            raise RuntimeError(f"Full RSFT mode restore failed: parameter remained frozen: {name}")


def _set_verify_warmup_mode(
    *,
    model,
    verify_token_ids: Sequence[int],
) -> Dict[str, object]:
    for p in model.parameters():
        p.requires_grad_(False)

    inp_w = model.get_input_embeddings().weight
    lm_w = _resolve_lm_head_weight(model)
    tied_weights = bool(inp_w.data_ptr() == lm_w.data_ptr())

    inp_w.requires_grad_(True)
    hooks: List[object] = []
    hooks.append(inp_w.register_hook(lambda g: _mask_non_verify_rows(g, verify_token_ids)))
    if not tied_weights:
        lm_w.requires_grad_(True)
        hooks.append(lm_w.register_hook(lambda g: _mask_non_verify_rows(g, verify_token_ids)))

    return {
        "tied_weights": bool(tied_weights),
        "hooks": hooks,
        "inp_param": inp_w,
        "lm_param": lm_w,
    }


def _remove_hooks(hooks: Sequence[object]) -> None:
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


def _assert_verify_row_grads_only(*, param: torch.nn.Parameter, verify_token_ids: Sequence[int], name: str) -> None:
    g = param.grad
    if g is None:
        return
    if g.ndim < 2:
        raise RuntimeError(f"{name} grad tensor expected ndim>=2, got {g.ndim}")
    ids = [int(x) for x in verify_token_ids]
    with torch.no_grad():
        row_abs = g.abs().sum(dim=tuple(range(1, g.ndim)))
        allowed = torch.zeros_like(row_abs, dtype=torch.bool)
        allowed[ids] = True
        bad = row_abs[(~allowed)] > 0
        if bool(bad.any()):
            raise RuntimeError(f"Warmup safety check failed: non-verify rows received gradient in {name}")


def train(cfg: Optional[Config] = None) -> str:
    if cfg is None:
        cfg = Config()
    _validate_cfg(cfg)

    _set_seed(int(cfg.train.seed))
    run_dir = _make_run_dir(str(cfg.logging.output_dir))
    log_path = os.path.join(run_dir, "logs", "run.log")
    metrics_csv = os.path.join(run_dir, "logs", "metrics.csv")
    metrics_jsonl = os.path.join(run_dir, "logs", "metrics.jsonl")
    rollout_logger = RolloutLogger(os.path.join(run_dir, "logs"))
    metrics_fields = list(METRICS_FIELDS) + [f"accepted_round_frac_{r}" for r in range(1, int(cfg.rollout.max_rounds) + 1)]

    _log(f"Starting RSFT run at {run_dir}", log_path)
    _log(f"Config: {json.dumps(cfg.as_dict(), ensure_ascii=True)}", log_path)

    (
        tokenizer,
        model,
        answer_token_id,
        z_token_ids,
        digit_token_ids,
        digit_id_to_val,
        finalize_token_id,
        retry_token_id,
    ) = _prepare_tokenizer_and_model(cfg)
    verify_token_ids = [int(finalize_token_id), int(retry_token_id)]

    vllm_init_ckpt_ref = str(cfg.model.init_ckpt)
    if str(cfg.rollout.backend).strip().lower() == "vllm":
        # vLLM must see the same vocab size as the torch model after adding special tokens.
        vllm_bootstrap_dir = os.path.join(run_dir, "tmp_vllm_bootstrap_vocab")
        os.makedirs(vllm_bootstrap_dir, exist_ok=True)
        tokenizer.save_pretrained(vllm_bootstrap_dir)
        model.config.save_pretrained(vllm_bootstrap_dir)
        vllm_init_ckpt_ref = str(vllm_bootstrap_dir)
        _log(f"Prepared vLLM bootstrap config/tokenizer at {vllm_bootstrap_dir}", log_path)

    device = torch.device(str(cfg.rollout.torch_device))
    model.to(device)
    model.train()

    warmup_steps = int(cfg.train.warmup_steps)
    current_train_mode = "full_rsft"
    warmup_runtime: Dict[str, object] = {
        "hooks": [],
        "tied_weights": False,
        "inp_param": None,
        "lm_param": None,
    }
    if warmup_steps > 0:
        warmup_runtime = _set_verify_warmup_mode(model=model, verify_token_ids=verify_token_ids)
        current_train_mode = "verify_warmup"
        effective_trainable_params = _count_effective_warmup_trainable_params(
            warmup_runtime=warmup_runtime,
            verify_token_ids=verify_token_ids,
        )
        _log(
            " | ".join(
                [
                    f"train_mode={current_train_mode}",
                    f"warmup_steps={warmup_steps}",
                    f"verify_token_ids={verify_token_ids}",
                    f"emb_lm_head_tied={bool(warmup_runtime.get('tied_weights', False))}",
                    f"trainable_params={effective_trainable_params}",
                    "trainable_rows=input_embeddings+lm_head verify rows only",
                ]
            ),
            log_path,
        )
    else:
        _set_full_train_mode(model)
        _assert_full_train_mode_restored(model)
        _log(
            " | ".join(
                [
                    "train_mode=full_rsft",
                    f"warmup_steps={warmup_steps}",
                    f"verify_token_ids={verify_token_ids}",
                    f"trainable_params={_count_trainable_params(model)}",
                ]
            ),
            log_path,
        )

    optimizer = _build_optimizer(model=model, cfg=cfg)
    if current_train_mode == "verify_warmup":
        _set_optimizer_weight_decay(optimizer, 0.0)
    else:
        _set_optimizer_weight_decay(optimizer, float(cfg.train.weight_decay))
    optimizer.zero_grad(set_to_none=True)

    _log("Loading train records", log_path)
    train_records = load_hf_records(str(cfg.data.dataset_name), str(cfg.data.train_split))
    train_examples = prepare_prompt_examples(
        records=train_records,
        tokenizer=tokenizer,
        question_field=str(cfg.data.question_field),
        answer_digits_field=str(cfg.data.answer_digits_field),
        answer_field=str(cfg.data.answer_field),
    )
    if len(train_examples) == 0:
        raise RuntimeError("No usable training examples after parsing")

    _log("Loading eval records", log_path)
    eval_records = load_hf_records(str(cfg.data.dataset_name), str(cfg.data.eval_split))
    eval_examples = prepare_prompt_examples(
        records=eval_records,
        tokenizer=tokenizer,
        question_field=str(cfg.data.question_field),
        answer_digits_field=str(cfg.data.answer_digits_field),
        answer_field=str(cfg.data.answer_field),
    )

    rng = random.Random(int(cfg.train.seed))
    ordered_indices = list(range(len(train_examples)))
    rng.shuffle(ordered_indices)
    order_cursor = 0

    engine = _make_rollout_engine(
        cfg=cfg,
        tokenizer=tokenizer,
        init_ckpt_ref=vllm_init_ckpt_ref,
        answer_token_id=answer_token_id,
        z_token_ids=z_token_ids,
        digit_token_ids=digit_token_ids,
        verify_token_ids=verify_token_ids,
        finalize_token_id=finalize_token_id,
        retry_token_id=retry_token_id,
        run_dir=run_dir,
        logger=lambda msg: _log(msg, log_path),
    )

    update_step = 0
    with torch.no_grad():
        engine.maybe_sync_from_torch(model, tokenizer, update_idx=1)

    if bool(cfg.eval.eval_at_start):
        t_eval0 = time.perf_counter()
        eval0 = evaluate_with_rollout_engine(
            engine=engine,
            examples=eval_examples,
            cfg=cfg,
            answer_token_id=answer_token_id,
            digit_id_to_val=digit_id_to_val,
        )
        eval_time0 = float(time.perf_counter() - t_eval0)
        row0 = {
            "step": 0,
            "update_step": int(update_step),
            "prompts_sampled": 0,
            "total_sequences": 0,
            "accepted_prompts": 0,
            "accepted_rate": 0.0,
            "avg_rounds_per_accepted": 0.0,
            "avg_failed_rounds_before_success": 0.0,
            "mean_accepted_z_len_per_round": 0.0,
            "l_z_ans": 0.0,
            "l_digits": 0.0,
            "l_verify": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
            "rollout_time": 0.0,
            "train_time": 0.0,
            "sync_time": 0.0,
            "train_mode": ("verify_warmup" if warmup_steps > 0 else "full_rsft"),
            "warmup_steps": int(cfg.train.warmup_steps),
            "trainable_params": float(
                _count_effective_warmup_trainable_params(
                    warmup_runtime=warmup_runtime,
                    verify_token_ids=verify_token_ids,
                )
                if warmup_steps > 0
                else _count_trainable_params(model)
            ),
            "skipped_optimizer": True,
            "rollout_log_path": "",
            "evaluated_questions": eval0.get("evaluated_questions", 0.0),
            "greedy_exact": eval0.get("greedy_exact", 0.0),
            "pass_at_n": eval0.get("pass_at_n", 0.0),
            "mean_z_length": eval0.get("mean_z_length", 0.0),
            "no_answer_before_kmax_rate": eval0.get("no_answer_before_kmax_rate", 0.0),
            "eval_time": eval_time0,
        }
        row0.update(_build_round_distribution([], int(cfg.rollout.max_rounds)))
        _append_metrics_csv(metrics_csv, row0, metrics_fields)
        _append_metrics_jsonl(metrics_jsonl, row0)
        _log(
            " | ".join(
                [
                    "step=0",
                    f"greedy_exact={float(row0['greedy_exact']):.4f}",
                    f"pass@N={float(row0['pass_at_n']):.4f}",
                    f"no_answer_rate={float(row0['no_answer_before_kmax_rate']):.4f}",
                ]
            ),
            log_path,
        )

    try:
        for step in range(1, int(cfg.train.max_steps) + 1):
            should_be_warmup = bool(step <= warmup_steps and warmup_steps > 0)
            desired_mode = "verify_warmup" if should_be_warmup else "full_rsft"
            if desired_mode != current_train_mode:
                if current_train_mode == "verify_warmup":
                    _remove_hooks(warmup_runtime.get("hooks", []))  # type: ignore[arg-type]
                if desired_mode == "verify_warmup":
                    warmup_runtime = _set_verify_warmup_mode(model=model, verify_token_ids=verify_token_ids)
                    _set_optimizer_weight_decay(optimizer, 0.0)
                else:
                    _set_full_train_mode(model)
                    _assert_full_train_mode_restored(model)
                    _set_optimizer_weight_decay(optimizer, float(cfg.train.weight_decay))
                current_train_mode = desired_mode
                optimizer.zero_grad(set_to_none=True)
                effective_trainable_params = (
                    _count_effective_warmup_trainable_params(
                        warmup_runtime=warmup_runtime,
                        verify_token_ids=verify_token_ids,
                    )
                    if current_train_mode == "verify_warmup"
                    else _count_trainable_params(model)
                )
                _log(
                    " | ".join(
                        [
                            f"step={step}",
                            f"train_mode={current_train_mode}",
                            f"warmup_steps={warmup_steps}",
                            f"verify_token_ids={verify_token_ids}",
                            f"emb_lm_head_tied={bool(warmup_runtime.get('tied_weights', False))}",
                            f"trainable_params={effective_trainable_params}",
                        ]
                    ),
                    log_path,
                )

            t_step_start = time.perf_counter()
            accepted_rows: List[Dict[str, object]] = []
            step_rollout_logs: List[Dict[str, object]] = []
            accepted_round_counts: List[int] = []
            accepted_failed_round_counts: List[int] = []
            accepted_round_z_lens: List[float] = []
            accepted_prompt_indices: set[int] = set()
            step_seen_questions: set[str] = set()
            prompts_sampled = 0
            total_sequences = 0

            prompt_batch, order_cursor = _next_unique_batch(
                examples=train_examples,
                ordered_indices=ordered_indices,
                cursor=order_cursor,
                batch_size=int(cfg.rollout.vllm_batch_size),
                step_seen_questions=step_seen_questions,
                rng=rng,
            )
            if prompt_batch:
                prompts_sampled += len(prompt_batch)
                rpp = int(cfg.rollout.rollouts_per_prompt)
                max_rounds = int(cfg.rollout.max_rounds)
                flat_prompt_ids: List[List[int]] = []
                seq_prompt_idx: List[int] = []
                seq_rollout_idx: List[int] = []
                for pidx, prompt_ex in enumerate(prompt_batch):
                    for rollout_idx in range(rpp):
                        flat_prompt_ids.append(list(prompt_ex.prompt_ids))
                        seq_prompt_idx.append(int(pidx))
                        seq_rollout_idx.append(int(rollout_idx))
                total_sequences = len(flat_prompt_ids)

                rounds_by_seq: List[List[RoundTrace]] = [[] for _ in range(total_sequences)]
                status_by_seq: List[str] = ["active"] * total_sequences
                failure_reason_by_seq: List[Optional[str]] = [None] * total_sequences
                current_prompts: List[List[int]] = [list(x) for x in flat_prompt_ids]
                generated_rollout_token_ids_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
                sequence_logs_by_seq_idx: List[Dict[str, object]] = [
                    {
                        "prompt_idx": int(seq_prompt_idx[i]),
                        "rollout_idx": int(seq_rollout_idx[i]),
                        "question": str(prompt_batch[seq_prompt_idx[i]].question),
                        "accepted": False,
                        "terminal_status": "active",
                        "failure_reason": None,
                        "round_count_observed": 0,
                        "full_rollout_token_ids": [],
                        "full_sequence_token_ids": [],
                        "rounds": [],
                    }
                    for i in range(total_sequences)
                ]

                for round_idx in range(1, max_rounds + 1):
                    active_indices = [i for i, st in enumerate(status_by_seq) if st == "active"]
                    if not active_indices:
                        break

                    z_rows = engine.generate_z(
                        prompt_token_ids=[current_prompts[i] for i in active_indices],
                        num_samples_per_prompt=1,
                        max_new_tokens=int(cfg.rollout.max_new_tokens),
                        temperature=float(cfg.rollout.temperature),
                        top_p=float(cfg.rollout.top_p),
                        min_p=float(cfg.rollout.min_p),
                        repetition_penalty=float(cfg.rollout.repetition_penalty),
                    )
                    if len(z_rows) != len(active_indices):
                        raise RuntimeError("Z generation row count mismatch for active trajectories")

                    valid_active_indices: List[int] = []
                    z_ids_by_active: List[List[int]] = []
                    digit_prompts: List[List[int]] = []
                    for j, seq_idx in enumerate(active_indices):
                        z_ids = extract_z_before_answer_from_row(z_rows[j], answer_token_id=answer_token_id)
                        if z_ids is None:
                            status_by_seq[seq_idx] = "failed"
                            failure_reason_by_seq[seq_idx] = "no_answer_before_max_tokens"
                            true_digits = list(prompt_batch[seq_prompt_idx[seq_idx]].true_digits)
                            sequence_logs_by_seq_idx[seq_idx]["rounds"].append(  # type: ignore[index]
                                {
                                    "round_idx": int(round_idx),
                                    "z_len": 0,
                                    "z_token_ids": [],
                                    "digit_token_ids": [],
                                    "pred_digits": "",
                                    "true_digits": "".join(str(int(x)) for x in true_digits),
                                    "is_correct": False,
                                    "verify_token_id": -1,
                                    "verify_action": "NONE",
                                    "round_generated_token_ids": [],
                                    "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[seq_idx]),
                                    "round_event": "no_answer_before_max_tokens",
                                }
                            )
                            continue
                        valid_active_indices.append(seq_idx)
                        z_ids_by_active.append([int(x) for x in z_ids])
                        digit_prompts.append(list(current_prompts[seq_idx]) + list(z_ids) + [int(answer_token_id)])

                    digit_rows = engine.generate_digits(
                        prompt_token_ids=digit_prompts,
                        temperature=float(cfg.rollout.temperature),
                        top_p=float(cfg.rollout.top_p),
                        greedy=bool(cfg.rollout.digit_greedy),
                        min_p=float(cfg.rollout.min_p),
                        repetition_penalty=float(cfg.rollout.repetition_penalty),
                    ) if digit_prompts else []
                    if len(digit_rows) != len(valid_active_indices):
                        raise RuntimeError("Digit generation row count mismatch against valid trajectories")

                    for j, seq_idx in enumerate(valid_active_indices):
                        z_ids = z_ids_by_active[j]
                        dig_tokens = [int(x) for x in digit_rows[j]]
                        if len(dig_tokens) != 5:
                            raise RuntimeError(f"Digits phase must emit exactly 5 digits per round, got {len(dig_tokens)}")
                        pred_digits = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
                        if pred_digits is None:
                            raise RuntimeError("Digits decode failed despite restricted digit allowed-token set")
                        true_digits = list(prompt_batch[seq_prompt_idx[seq_idx]].true_digits)
                        is_correct = bool(pred_digits == true_digits)
                        verify_token_id = int(finalize_token_id if is_correct else retry_token_id)
                        if verify_token_id not in verify_token_ids:
                            raise RuntimeError("Verify token is outside allowed verify set")

                        rounds_by_seq[seq_idx].append(
                            RoundTrace(
                                z_token_ids=list(z_ids),
                                digit_token_ids=list(dig_tokens),
                                pred_digits=list(pred_digits),
                                true_digits=list(true_digits),
                                verify_token_id=int(verify_token_id),
                                is_correct=bool(is_correct),
                            )
                        )

                        current_prompts[seq_idx].extend(list(z_ids))
                        current_prompts[seq_idx].append(int(answer_token_id))
                        current_prompts[seq_idx].extend(list(dig_tokens))
                        current_prompts[seq_idx].append(int(verify_token_id))
                        round_generated_ids = list(z_ids) + [int(answer_token_id)] + list(dig_tokens) + [int(verify_token_id)]
                        generated_rollout_token_ids_by_seq[seq_idx].extend(round_generated_ids)

                        sequence_logs_by_seq_idx[seq_idx]["rounds"].append(  # type: ignore[index]
                            {
                                "round_idx": int(round_idx),
                                "z_len": int(len(z_ids)),
                                "z_token_ids": list(z_ids),
                                "digit_token_ids": list(dig_tokens),
                                "pred_digits": "".join(str(int(x)) for x in pred_digits),
                                "true_digits": "".join(str(int(x)) for x in true_digits),
                                "is_correct": bool(is_correct),
                                "verify_token_id": int(verify_token_id),
                                "verify_action": "FINALIZE" if is_correct else "RETRY",
                                "round_generated_token_ids": list(round_generated_ids),
                                "full_rollout_token_ids_so_far": list(generated_rollout_token_ids_by_seq[seq_idx]),
                            }
                        )

                        if is_correct:
                            status_by_seq[seq_idx] = "success"
                        elif round_idx >= max_rounds:
                            status_by_seq[seq_idx] = "failed"
                            failure_reason_by_seq[seq_idx] = "max_rounds_reached_without_success"

                for seq_idx in range(total_sequences):
                    rounds = rounds_by_seq[seq_idx]
                    status = status_by_seq[seq_idx]
                    reason = failure_reason_by_seq[seq_idx]
                    candidate_for_train = bool(status == "success") or bool(
                        status == "failed" and reason == "max_rounds_reached_without_success"
                    )
                    accepted_for_train = False
                    example_type: Optional[str] = None
                    built_example: Optional[Dict[str, object]] = None
                    if candidate_for_train:
                        built = build_training_example(
                            prompt_ids=flat_prompt_ids[seq_idx],
                            rounds=rounds,
                            answer_token_id=answer_token_id,
                            finalize_token_id=finalize_token_id,
                            retry_token_id=retry_token_id,
                            max_length=int(cfg.train.max_length),
                        )
                        if built is None:
                            status = "failed"
                            reason = "max_length_exceeded"
                            failure_reason_by_seq[seq_idx] = str(reason)
                        else:
                            built_example = built
                            if str(status) == "success":
                                example_type = "correct_answer"
                            elif str(status) == "failed" and str(reason) == "max_rounds_reached_without_success":
                                example_type = "full_failure"

                    if status != "success" and reason is None:
                        reason = "unknown_failure"
                    seq_row = sequence_logs_by_seq_idx[seq_idx]
                    seq_row["accepted"] = bool(accepted_for_train)
                    seq_row["terminal_status"] = str(status)
                    seq_row["failure_reason"] = (None if str(status) == "success" else str(reason))
                    seq_row["round_count_observed"] = int(len(seq_row["rounds"]))  # type: ignore[arg-type]
                    seq_row["full_rollout_token_ids"] = list(generated_rollout_token_ids_by_seq[seq_idx])
                    seq_row["full_sequence_token_ids"] = list(flat_prompt_ids[seq_idx]) + list(
                        generated_rollout_token_ids_by_seq[seq_idx]
                    )
                    seq_row["example_type"] = example_type

                    # Store candidate artifacts for two-stage filtering.
                    seq_row["_candidate_built_example"] = built_example

                # Stage A: exclude only prompts where all rollouts are successful on round 1.
                excluded_prompt_idxs: set[int] = set()
                by_prompt_seq_indices: Dict[int, List[int]] = {}
                for seq_idx in range(total_sequences):
                    pidx = int(seq_prompt_idx[seq_idx])
                    by_prompt_seq_indices.setdefault(pidx, []).append(seq_idx)
                for pidx, seq_ids in by_prompt_seq_indices.items():
                    if len(seq_ids) == 0:
                        continue
                    all_success_round1 = all(
                        str(status_by_seq[i]) == "success" and int(len(rounds_by_seq[i])) == 1 for i in seq_ids
                    )
                    if all_success_round1:
                        excluded_prompt_idxs.add(int(pidx))

                # Stage B: among eligible examples, keep all correct and sample failures up to half.
                eligible_correct_seq_idxs: List[int] = []
                eligible_full_failure_seq_idxs: List[int] = []
                for seq_idx in range(total_sequences):
                    pidx = int(seq_prompt_idx[seq_idx])
                    if pidx in excluded_prompt_idxs:
                        continue
                    seq_row = sequence_logs_by_seq_idx[seq_idx]
                    built_example = seq_row.get("_candidate_built_example", None)
                    if not isinstance(built_example, dict):
                        continue
                    et = str(seq_row.get("example_type", ""))
                    if et == "correct_answer":
                        eligible_correct_seq_idxs.append(seq_idx)
                    elif et == "full_failure":
                        eligible_full_failure_seq_idxs.append(seq_idx)

                correct_before = int(len(eligible_correct_seq_idxs))
                full_failure_before = int(len(eligible_full_failure_seq_idxs))
                full_failure_cap = int(correct_before * 0.5)
                if full_failure_before <= full_failure_cap:
                    kept_full_failure_seq_idxs = list(eligible_full_failure_seq_idxs)
                else:
                    kept_full_failure_seq_idxs = list(rng.sample(eligible_full_failure_seq_idxs, full_failure_cap))
                kept_seq_idxs: set[int] = set(eligible_correct_seq_idxs) | set(kept_full_failure_seq_idxs)

                # Finalize accepted_rows and per-sequence accepted flags.
                for seq_idx in range(total_sequences):
                    seq_row = sequence_logs_by_seq_idx[seq_idx]
                    built_example = seq_row.get("_candidate_built_example", None)
                    accepted_for_train = bool(seq_idx in kept_seq_idxs and isinstance(built_example, dict))
                    seq_row["accepted"] = bool(accepted_for_train)
                    if accepted_for_train:
                        built = dict(built_example)
                        built["source_prompt_idx"] = int(seq_prompt_idx[seq_idx])
                        accepted_rows.append(built)
                        accepted_prompt_indices.add(int(seq_prompt_idx[seq_idx]))
                        accepted_round_counts.append(int(built["round_count"]))  # type: ignore[index]
                        accepted_failed_round_counts.append(int(built["failed_rounds"]))  # type: ignore[index]
                        for zlen in list(built["round_z_lens"]):  # type: ignore[index]
                            accepted_round_z_lens.append(float(zlen))
                    seq_row.pop("_candidate_built_example", None)

                _log(
                    " | ".join(
                        [
                            f"step={step}",
                            f"excluded_prompts_all_r1_correct={int(len(excluded_prompt_idxs))}",
                            f"correct_examples_before={correct_before}",
                            f"correct_examples_after={int(len(eligible_correct_seq_idxs))}",
                            f"full_failures_before_sampling={full_failure_before}",
                            f"full_failures_kept={int(len(kept_full_failure_seq_idxs))}",
                        ]
                    ),
                    log_path,
                )

                step_rollout_logs.extend(sequence_logs_by_seq_idx)

                if accepted_rows:
                    by_prompt: Dict[int, int] = {}
                    for ex in accepted_rows:
                        pidx = int(ex.get("source_prompt_idx", -1))
                        by_prompt[pidx] = by_prompt.get(pidx, 0) + 1
                    for ex in accepted_rows:
                        pidx = int(ex.get("source_prompt_idx", -1))
                        denom = max(by_prompt.get(pidx, 1), 1)
                        ex["example_weight"] = 1.0 / float(denom)

            rollout_path = rollout_logger.write_step(step, step_rollout_logs)

            l_z_ans_val = 0.0
            l_digits_val = 0.0
            l_verify_val = 0.0
            total_loss_val = 0.0
            grad_norm_val = 0.0
            train_time = 0.0
            sync_time = 0.0
            skipped_optimizer = False

            if len(accepted_rows) == 0:
                skipped_optimizer = True
                _log(f"step={step} no accepted examples; skipping optimizer step", log_path)
            else:
                t_train = time.perf_counter()
                micro_batches = _chunk_examples(accepted_rows, int(cfg.train.train_batch_size))
                total_example_weight = float(
                    sum(float(ex.get("example_weight", 1.0)) for ex in accepted_rows)  # type: ignore[call-overload]
                )
                if total_example_weight <= 0.0:
                    total_example_weight = float(len(accepted_rows))

                optimizer.zero_grad(set_to_none=True)

                lz_weighted_sum = 0.0
                ld_weighted_sum = 0.0
                lv_weighted_sum = 0.0
                tl_weighted_sum = 0.0
                w_seen_sum = 0.0

                for micro in micro_batches:
                    batch = collate_training_examples(micro, pad_token_id=int(tokenizer.pad_token_id))
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    target_class = batch["target_class"].to(device)
                    example_weights = torch.tensor(
                        [float(ex.get("example_weight", 1.0)) for ex in micro],  # type: ignore[call-overload]
                        dtype=torch.float32,
                        device=device,
                    )

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if (device.type == "cuda" and bool(cfg.train.use_bf16))
                        else nullcontext()
                    )
                    with amp_ctx:
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        per_lz: List[torch.Tensor] = []
                        per_ld: List[torch.Tensor] = []
                        per_lv: List[torch.Tensor] = []
                        per_tl: List[torch.Tensor] = []
                        for i in range(out.logits.shape[0]):
                            losses_i = compute_rsft_losses(
                                logits=out.logits[i : i + 1],
                                labels=labels[i : i + 1],
                                target_class=target_class[i : i + 1],
                                z_token_ids=z_token_ids,
                                answer_token_id=answer_token_id,
                                digit_token_ids=digit_token_ids,
                                verify_token_ids=verify_token_ids,
                                w_z_ans=float(cfg.loss.w_z_ans),
                                w_digits=float(cfg.loss.w_digits),
                                w_verify=float(cfg.loss.w_verify),
                            )
                            per_lv.append(losses_i["l_verify"])
                            if current_train_mode == "verify_warmup":
                                per_lz.append(losses_i["l_verify"].new_zeros(()))
                                per_ld.append(losses_i["l_verify"].new_zeros(()))
                                per_tl.append(float(cfg.loss.w_verify) * losses_i["l_verify"])
                            else:
                                per_lz.append(losses_i["l_z_ans"])
                                per_ld.append(losses_i["l_digits"])
                                per_tl.append(losses_i["loss"])
                        lz_t = torch.stack(per_lz)
                        ld_t = torch.stack(per_ld)
                        lv_t = torch.stack(per_lv)
                        tl_t = torch.stack(per_tl)
                        w_t = example_weights.to(tl_t.dtype)
                        # Normalize by total step question-weight so each question contributes equally.
                        loss = (tl_t * w_t).sum() / float(total_example_weight)

                    loss.backward()
                    if current_train_mode == "verify_warmup":
                        inp_p = warmup_runtime.get("inp_param", None)
                        if isinstance(inp_p, torch.nn.Parameter):
                            _assert_verify_row_grads_only(
                                param=inp_p,
                                verify_token_ids=verify_token_ids,
                                name="input_embeddings.weight",
                            )
                        lm_p = warmup_runtime.get("lm_param", None)
                        if isinstance(lm_p, torch.nn.Parameter) and (lm_p is not inp_p):
                            _assert_verify_row_grads_only(
                                param=lm_p,
                                verify_token_ids=verify_token_ids,
                                name="lm_head.weight",
                            )

                    lz_weighted_sum += float((lz_t.detach().float() * example_weights).sum().item())
                    ld_weighted_sum += float((ld_t.detach().float() * example_weights).sum().item())
                    lv_weighted_sum += float((lv_t.detach().float() * example_weights).sum().item())
                    tl_weighted_sum += float((tl_t.detach().float() * example_weights).sum().item())
                    w_seen_sum += float(example_weights.sum().item())

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
                if current_train_mode == "verify_warmup":
                    _assert_optimizer_weight_decay(optimizer, 0.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                denom = max(w_seen_sum, 1e-12)
                l_z_ans_val = float(lz_weighted_sum / denom)
                l_digits_val = float(ld_weighted_sum / denom)
                l_verify_val = float(lv_weighted_sum / denom)
                total_loss_val = float(tl_weighted_sum / denom)
                grad_norm_val = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                train_time = time.perf_counter() - t_train

                update_step += 1
                t_sync = time.perf_counter()
                with torch.no_grad():
                    engine.maybe_sync_from_torch(model, tokenizer, update_idx=update_step)
                sync_time = time.perf_counter() - t_sync

            rollout_time = (time.perf_counter() - t_step_start) - train_time - sync_time

            accepted_prompts = int(len(accepted_prompt_indices))
            accepted_rate = float(accepted_prompts) / float(prompts_sampled) if prompts_sampled > 0 else 0.0
            row = {
                "step": int(step),
                "update_step": int(update_step),
                "prompts_sampled": int(prompts_sampled),
                "total_sequences": int(total_sequences),
                "accepted_prompts": int(accepted_prompts),
                "accepted_rate": float(accepted_rate),
                "avg_rounds_per_accepted": float(mean_or_zero(accepted_round_counts)),
                "avg_failed_rounds_before_success": float(mean_or_zero(accepted_failed_round_counts)),
                "mean_accepted_z_len_per_round": float(mean_or_zero(accepted_round_z_lens)),
                "l_z_ans": float(l_z_ans_val),
                "l_digits": float(l_digits_val),
                "l_verify": float(l_verify_val),
                "total_loss": float(total_loss_val),
                "grad_norm": float(grad_norm_val),
                "rollout_time": float(max(rollout_time, 0.0)),
                "train_time": float(train_time),
                "sync_time": float(sync_time),
                "train_mode": str(current_train_mode),
                "warmup_steps": int(warmup_steps),
                "trainable_params": float(
                    _count_effective_warmup_trainable_params(
                        warmup_runtime=warmup_runtime,
                        verify_token_ids=verify_token_ids,
                    )
                    if current_train_mode == "verify_warmup"
                    else _count_trainable_params(model)
                ),
                "skipped_optimizer": bool(skipped_optimizer),
                "rollout_log_path": str(rollout_path),
                "evaluated_questions": None,
                "greedy_exact": None,
                "pass_at_n": None,
                "mean_z_length": None,
                "no_answer_before_kmax_rate": None,
                "eval_time": None,
            }
            row.update(_build_round_distribution(accepted_round_counts, int(cfg.rollout.max_rounds)))

            if step % int(cfg.eval.eval_every_steps) == 0:
                t_eval = time.perf_counter()
                eval_metrics = evaluate_with_rollout_engine(
                    engine=engine,
                    examples=eval_examples,
                    cfg=cfg,
                    answer_token_id=answer_token_id,
                    digit_id_to_val=digit_id_to_val,
                )
                row.update(eval_metrics)
                row["eval_time"] = float(time.perf_counter() - t_eval)

            _append_metrics_csv(metrics_csv, row, metrics_fields)
            _append_metrics_jsonl(metrics_jsonl, row)

            if (step % int(cfg.logging.log_every)) == 0:
                _log(
                    " | ".join(
                        [
                            f"step={row['step']}",
                            f"mode={row['train_mode']}",
                            f"accepted={row['accepted_prompts']}",
                            f"accepted_rate={row['accepted_rate']:.4f}",
                            f"avg_rounds={row['avg_rounds_per_accepted']:.3f}",
                            f"Lz={row['l_z_ans']:.4f}",
                            f"Ld={row['l_digits']:.4f}",
                            f"Lv={row['l_verify']:.4f}",
                            f"L={row['total_loss']:.4f}",
                            f"rollout_t={row['rollout_time']:.2f}s",
                            f"train_t={row['train_time']:.2f}s",
                            f"sync_t={row['sync_time']:.2f}s",
                        ]
                    ),
                    log_path,
                )

            if (step % int(cfg.logging.save_every)) == 0:
                ckpt = _save_periodic(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    step=step,
                    keep_last=int(cfg.logging.keep_last),
                )
                _save_last(run_dir=run_dir, model=model, tokenizer=tokenizer, cfg=cfg, step=step)
                _log(f"Saved checkpoint: {ckpt}", log_path)

        _save_last(run_dir=run_dir, model=model, tokenizer=tokenizer, cfg=cfg, step=int(cfg.train.max_steps))
    finally:
        _remove_hooks(warmup_runtime.get("hooks", []))  # type: ignore[arg-type]
        _set_full_train_mode(model)
        try:
            engine.close()
        except Exception:
            pass
    return run_dir


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = Config()
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item!r}; expected key=value")
        k, v = item.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
    train(cfg)


if __name__ == "__main__":
    main()
