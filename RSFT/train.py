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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.hf_rollout import HFRolloutEngine
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single
from PPO.vllm_rollout import VLLMRolloutEngine
from RSFT.config import Config, DEFAULT_SET_ALLOWED_PREFIXES
from RSFT.dataset import PromptExample, load_hf_records, make_digit_id_to_value_map, prepare_prompt_examples, sample_unique_prompt_batch
from RSFT.eval_vllm import evaluate_with_rollout_engine
from RSFT.logic import (
    RolloutCandidate,
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
    "total_rollouts",
    "exact_rollouts",
    "selected_examples",
    "exact_rollout_rate",
    "selected_example_rate",
    "k_distribution",
    "avg_applied_prompt_weight",
    "mean_accepted_z_len",
    "l_z_ans",
    "l_digits",
    "total_loss",
    "grad_norm",
    "rollout_time",
    "train_time",
    "sync_time",
    "skipped_optimizer",
    "rollout_log_path",
    "evaluated_questions",
    "greedy_exact",
    "pass_at_n",
    "mean_z_length",
    "no_answer_before_kmax_rate",
    "eval_time",
]

PROMPT_WEIGHT_BY_K: Dict[int, float] = {
    1: 1.0,
    2: 0.95,
    3: 0.90,
    4: 0.8,
    5: 0.6,
    6: 0.40,
    7: 0.20,
}


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

    answer_token_id = int(resolve_answer_token_id(tokenizer, answer_token=str(cfg.model.answer_token)))
    validate_answer_token_single(tokenizer, str(cfg.model.answer_token), answer_token_id)

    z_token_ids, _style = introspect_z_token_ids_and_style(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found in tokenizer vocab")

    digit_token_ids = resolve_digit_token_ids(tokenizer)
    digit_id_to_val = make_digit_id_to_value_map(digit_token_ids)

    return tokenizer, model, answer_token_id, z_token_ids, digit_token_ids, digit_id_to_val


def _make_rollout_engine(
    *,
    cfg: Config,
    tokenizer,
    answer_token_id: int,
    z_token_ids: Sequence[int],
    digit_token_ids: Sequence[int],
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
            init_ckpt=str(cfg.model.init_ckpt),
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=list(z_token_ids),
            digit_allowed_token_ids=list(digit_token_ids),
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
            sync_every=int(cfg.rollout.sync_every_n_steps),
            logger=logger,
        )
    raise ValueError(f"Unsupported rollout.backend={cfg.rollout.backend!r}; expected 'vllm' or 'hf'")


def _append_metrics_csv(path: str, row: Dict[str, object]) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _chunk_examples(rows: Sequence[Dict[str, List[int]]], chunk_size: int) -> List[List[Dict[str, List[int]]]]:
    if len(rows) <= 0:
        return []
    k = max(1, int(chunk_size))
    return [list(rows[i : i + k]) for i in range(0, len(rows), k)]


def _prompt_weight_multiplier(k_correct: int, *, use_prompt_weighting: bool) -> float:
    if int(k_correct) <= 0:
        return 0.0
    if not bool(use_prompt_weighting):
        return 1.0
    return float(PROMPT_WEIGHT_BY_K.get(int(k_correct), 1.0))


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


def _mean_prefix_logprob(row: Dict[str, object], prefix_len: int) -> Optional[float]:
    if prefix_len <= 0:
        return None
    vals_raw = row.get("token_logprobs", None)
    if vals_raw is None:
        return None
    vals = list(vals_raw)
    if len(vals) < int(prefix_len):
        return None
    use = vals[: int(prefix_len)]
    if any(v is None for v in use):
        return None
    try:
        nums = [float(v) for v in use]
    except Exception:
        return None
    if len(nums) == 0:
        return None
    return float(sum(nums) / len(nums))


def _build_rollout_candidates(
    *,
    prompts: Sequence[PromptExample],
    z_rows: Sequence[Dict[str, object]],
    answer_token_id: int,
    digit_rows: Sequence[List[int]],
    digit_id_to_val: Dict[int, int],
    rollouts_per_prompt: int,
) -> tuple[Dict[int, List[RolloutCandidate]], List[Dict[str, object]], int]:
    by_prompt: Dict[int, List[RolloutCandidate]] = {i: [] for i in range(len(prompts))}
    logs: List[Dict[str, object]] = []
    exact_rollouts = 0

    valid_indices: List[int] = []
    valid_z_ids: List[List[int]] = []
    for flat_idx, row in enumerate(z_rows):
        prompt_idx = flat_idx // int(rollouts_per_prompt)
        rollout_idx = flat_idx % int(rollouts_per_prompt)
        z_ids = extract_z_before_answer_from_row(row, answer_token_id=answer_token_id)
        if z_ids is None:
            logs.append(
                {
                    "prompt_idx": int(prompt_idx),
                    "rollout_idx": int(rollout_idx),
                    "has_answer": False,
                    "z_len": None,
                    "exact_match": False,
                    "selected": False,
                }
            )
            continue
        valid_indices.append(flat_idx)
        valid_z_ids.append(z_ids)

    dig_rows = list(digit_rows)
    if len(dig_rows) != len(valid_indices):
        raise RuntimeError("digit_rows length mismatch against valid rollout rows")

    for local_idx, flat_idx in enumerate(valid_indices):
        prompt_idx = flat_idx // int(rollouts_per_prompt)
        rollout_idx = flat_idx % int(rollouts_per_prompt)
        dig_tokens = [int(x) for x in dig_rows[local_idx]]
        pred_digits = decode_digit_tokens(dig_tokens, digit_id_to_val=digit_id_to_val)
        if pred_digits is None:
            logs.append(
                {
                    "prompt_idx": int(prompt_idx),
                    "rollout_idx": int(rollout_idx),
                    "has_answer": True,
                    "z_len": int(len(valid_z_ids[local_idx])),
                    "exact_match": False,
                    "selected": False,
                }
            )
            continue

        cand = RolloutCandidate(
            prompt_idx=int(prompt_idx),
            rollout_idx=int(rollout_idx),
            z_token_ids=list(valid_z_ids[local_idx]),
            z_avg_logprob=_mean_prefix_logprob(z_rows[flat_idx], len(valid_z_ids[local_idx])),
            digit_token_ids=list(dig_tokens),
            pred_digits=list(pred_digits),
            true_digits=list(prompts[prompt_idx].true_digits),
        )
        by_prompt[prompt_idx].append(cand)
        is_exact = bool(cand.pred_digits == cand.true_digits)
        if is_exact:
            exact_rollouts += 1
        logs.append(
            {
                "prompt_idx": int(prompt_idx),
                "rollout_idx": int(rollout_idx),
                "has_answer": True,
                "z_len": int(len(cand.z_token_ids)),
                "z_avg_logprob": cand.z_avg_logprob,
                "exact_match": is_exact,
                "selected": False,
            }
        )

    return by_prompt, logs, int(exact_rollouts)


def train(cfg: Optional[Config] = None) -> str:
    if cfg is None:
        cfg = Config()

    _set_seed(int(cfg.train.seed))
    run_dir = _make_run_dir(str(cfg.logging.output_dir))
    log_path = os.path.join(run_dir, "logs", "run.log")
    metrics_csv = os.path.join(run_dir, "logs", "metrics.csv")
    rollout_logger = RolloutLogger(os.path.join(run_dir, "logs"))

    _log(f"Starting RSFT run at {run_dir}", log_path)
    _log(f"Config: {json.dumps(cfg.as_dict(), ensure_ascii=True)}", log_path)

    tokenizer, model, answer_token_id, z_token_ids, digit_token_ids, digit_id_to_val = _prepare_tokenizer_and_model(cfg)

    device = torch.device(str(cfg.rollout.torch_device))
    model.to(device)
    model.train()

    optimizer = _build_optimizer(model=model, cfg=cfg)
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
        answer_token_id=answer_token_id,
        z_token_ids=z_token_ids,
        digit_token_ids=digit_token_ids,
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
            "total_rollouts": 0,
            "exact_rollouts": 0,
            "selected_examples": 0,
            "exact_rollout_rate": 0.0,
            "selected_example_rate": 0.0,
            "k_distribution": "{}",
            "avg_applied_prompt_weight": 0.0,
            "mean_accepted_z_len": 0.0,
            "l_z_ans": 0.0,
            "l_digits": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
            "rollout_time": 0.0,
            "train_time": 0.0,
            "sync_time": 0.0,
            "skipped_optimizer": True,
            "rollout_log_path": "",
            "evaluated_questions": eval0.get("evaluated_questions", 0.0),
            "greedy_exact": eval0.get("greedy_exact", 0.0),
            "pass_at_n": eval0.get("pass_at_n", 0.0),
            "mean_z_length": eval0.get("mean_z_length", 0.0),
            "no_answer_before_kmax_rate": eval0.get("no_answer_before_kmax_rate", 0.0),
            "eval_time": eval_time0,
        }
        _append_metrics_csv(metrics_csv, row0)
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
            t_step_start = time.perf_counter()
            accepted_rows: List[Dict[str, List[int]]] = []
            step_rollout_logs: List[Dict[str, object]] = []
            accepted_z_lens: List[float] = []
            step_seen_questions: set[str] = set()
            k_distribution: Dict[int, int] = {}
            applied_prompt_weights: List[float] = []
            prompts_sampled = 0
            total_rollouts = 0
            exact_rollouts = 0

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
                z_rows = engine.generate_z(
                    prompt_token_ids=[list(ex.prompt_ids) for ex in prompt_batch],
                    num_samples_per_prompt=int(cfg.rollout.rollouts_per_prompt),
                    max_new_tokens=int(cfg.rollout.max_new_tokens),
                    temperature=float(cfg.rollout.temperature),
                    top_p=float(cfg.rollout.top_p),
                    min_p=float(cfg.rollout.min_p),
                    repetition_penalty=float(cfg.rollout.repetition_penalty),
                )
                total_rollouts += len(z_rows)

                valid_prompts_for_digits: List[List[int]] = []
                for flat_idx, row in enumerate(z_rows):
                    pidx = flat_idx // int(cfg.rollout.rollouts_per_prompt)
                    z_ids = extract_z_before_answer_from_row(row, answer_token_id=answer_token_id)
                    if z_ids is None:
                        continue
                    valid_prompts_for_digits.append(list(prompt_batch[pidx].prompt_ids) + list(z_ids) + [int(answer_token_id)])

                if valid_prompts_for_digits:
                    digit_rows = engine.generate_digits(
                        prompt_token_ids=valid_prompts_for_digits,
                        temperature=float(cfg.rollout.temperature),
                        top_p=float(cfg.rollout.top_p),
                        greedy=bool(cfg.rollout.digit_greedy),
                        min_p=float(cfg.rollout.min_p),
                        repetition_penalty=float(cfg.rollout.repetition_penalty),
                    )
                else:
                    digit_rows = []

                cand_by_prompt, rollout_log_rows, exact_count_chunk = _build_rollout_candidates(
                    prompts=prompt_batch,
                    z_rows=z_rows,
                    answer_token_id=answer_token_id,
                    digit_rows=digit_rows,
                    digit_id_to_val=digit_id_to_val,
                    rollouts_per_prompt=int(cfg.rollout.rollouts_per_prompt),
                )
                exact_rollouts += int(exact_count_chunk)
                prompt_has_wrong: Dict[int, bool] = {i: False for i in range(len(prompt_batch))}
                for row in rollout_log_rows:
                    pidx_row = int(row.get("prompt_idx", -1))
                    if pidx_row < 0 or pidx_row >= len(prompt_batch):
                        continue
                    if not bool(row.get("exact_match", False)):
                        prompt_has_wrong[pidx_row] = True
                for pidx, prompt_ex in enumerate(prompt_batch):
                    # Keep only prompts with mixed outcomes (at least one wrong rollout).
                    if not bool(prompt_has_wrong.get(pidx, True)):
                        continue
                    cands = list(cand_by_prompt.get(pidx, []))
                    if not cands:
                        continue
                    correct_cands = [c for c in cands if c.pred_digits == c.true_digits]
                    k_correct = int(len(correct_cands))
                    if k_correct == 0:
                        continue
                    k_distribution[k_correct] = int(k_distribution.get(k_correct, 0) + 1)
                    built_rows: List[Tuple[int, Dict[str, List[int]]]] = []
                    for cand in correct_cands:
                        built = build_training_example(
                            prompt_ids=prompt_ex.prompt_ids,
                            z_token_ids=cand.z_token_ids,
                            answer_token_id=answer_token_id,
                            digit_token_ids=cand.digit_token_ids,
                            max_length=int(cfg.train.max_length),
                        )
                        if built is None:
                            continue
                        built_rows.append((int(cand.rollout_idx), built))
                    if len(built_rows) == 0:
                        continue
                    k_trained = int(len(built_rows))
                    prompt_weight = _prompt_weight_multiplier(
                        k_trained,
                        use_prompt_weighting=bool(cfg.loss.use_prompt_weighting),
                    )
                    per_example_weight = prompt_weight * (1.0 / float(k_trained))
                    applied_prompt_weights.append(float(prompt_weight))
                    for rollout_idx, built in built_rows:
                        built["example_weight"] = per_example_weight  # type: ignore[index]
                        accepted_rows.append(built)
                        accepted_z_lens.append(float(built["z_len"]))
                        # Mark selected rows in current rollout chunk.
                        for row in rollout_log_rows:
                            if (
                                int(row.get("prompt_idx", -1)) == int(pidx)
                                and int(row.get("rollout_idx", -1)) == int(rollout_idx)
                            ):
                                row["selected"] = True
                                break

                step_rollout_logs.extend(rollout_log_rows)

            rollout_path = rollout_logger.write_step(step, step_rollout_logs)

            l_z_ans_val = 0.0
            l_digits_val = 0.0
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
                        per_tl: List[torch.Tensor] = []
                        for i in range(out.logits.shape[0]):
                            losses_i = compute_rsft_losses(
                                logits=out.logits[i : i + 1],
                                labels=labels[i : i + 1],
                                target_class=target_class[i : i + 1],
                                z_token_ids=z_token_ids,
                                answer_token_id=answer_token_id,
                                digit_token_ids=digit_token_ids,
                                w_z_ans=float(cfg.loss.w_z_ans),
                                w_digits=float(cfg.loss.w_digits),
                            )
                            per_lz.append(losses_i["l_z_ans"])
                            per_ld.append(losses_i["l_digits"])
                            per_tl.append(losses_i["loss"])
                        lz_t = torch.stack(per_lz)
                        ld_t = torch.stack(per_ld)
                        tl_t = torch.stack(per_tl)
                        w_t = example_weights.to(tl_t.dtype)
                        # Normalize by total step question-weight so each question contributes equally.
                        loss = (tl_t * w_t).sum() / float(total_example_weight)

                    loss.backward()

                    lz_weighted_sum += float((lz_t.detach().float() * example_weights).sum().item())
                    ld_weighted_sum += float((ld_t.detach().float() * example_weights).sum().item())
                    tl_weighted_sum += float((tl_t.detach().float() * example_weights).sum().item())
                    w_seen_sum += float(example_weights.sum().item())

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                denom = max(w_seen_sum, 1e-12)
                l_z_ans_val = float(lz_weighted_sum / denom)
                l_digits_val = float(ld_weighted_sum / denom)
                total_loss_val = float(tl_weighted_sum / denom)
                grad_norm_val = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                train_time = time.perf_counter() - t_train

                update_step += 1
                t_sync = time.perf_counter()
                with torch.no_grad():
                    engine.maybe_sync_from_torch(model, tokenizer, update_idx=update_step)
                sync_time = time.perf_counter() - t_sync

            rollout_time = (time.perf_counter() - t_step_start) - train_time - sync_time

            selected_examples = int(len(accepted_rows))
            exact_rollout_rate = float(exact_rollouts) / float(total_rollouts) if total_rollouts > 0 else 0.0
            selected_example_rate = float(selected_examples) / float(total_rollouts) if total_rollouts > 0 else 0.0
            k_distribution_str = json.dumps(
                {str(k): int(v) for k, v in sorted(k_distribution.items(), key=lambda kv: kv[0])},
                ensure_ascii=True,
                sort_keys=True,
            )
            avg_applied_prompt_weight = (
                float(sum(applied_prompt_weights) / float(len(applied_prompt_weights)))
                if len(applied_prompt_weights) > 0
                else 0.0
            )
            row = {
                "step": int(step),
                "update_step": int(update_step),
                "prompts_sampled": int(prompts_sampled),
                "total_rollouts": int(total_rollouts),
                "exact_rollouts": int(exact_rollouts),
                "selected_examples": selected_examples,
                "exact_rollout_rate": float(exact_rollout_rate),
                "selected_example_rate": float(selected_example_rate),
                "k_distribution": k_distribution_str,
                "avg_applied_prompt_weight": float(avg_applied_prompt_weight),
                "mean_accepted_z_len": float(mean_or_zero(accepted_z_lens)),
                "l_z_ans": float(l_z_ans_val),
                "l_digits": float(l_digits_val),
                "total_loss": float(total_loss_val),
                "grad_norm": float(grad_norm_val),
                "rollout_time": float(max(rollout_time, 0.0)),
                "train_time": float(train_time),
                "sync_time": float(sync_time),
                "skipped_optimizer": bool(skipped_optimizer),
                "rollout_log_path": str(rollout_path),
                "evaluated_questions": None,
                "greedy_exact": None,
                "pass_at_n": None,
                "mean_z_length": None,
                "no_answer_before_kmax_rate": None,
                "eval_time": None,
            }

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

            _append_metrics_csv(metrics_csv, row)

            if (step % int(cfg.logging.log_every)) == 0:
                _log(
                    " | ".join(
                        [
                            f"step={row['step']}",
                            f"selected={row['selected_examples']}",
                            f"exact_rate={row['exact_rollout_rate']:.4f}",
                            f"selected_rate={row['selected_example_rate']:.4f}",
                            f"k_dist={row['k_distribution']}",
                            f"avg_prompt_w={row['avg_applied_prompt_weight']:.4f}",
                            f"Lz={row['l_z_ans']:.4f}",
                            f"Ld={row['l_digits']:.4f}",
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
