from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bitsandbytes as bnb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.hf_rollout import HFRolloutEngine
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single, validate_single_token
from PPO.train import (
    _build_minibatch_order,
    _build_prompt_text,
    _build_trajectory_device_cache,
    _extract_true_digits,
    _get_token_stats_kernel,
    _load_rsft_trained_questions,
    _question_text,
)
from PPO.vllm_rollout import VLLMRolloutEngine
from TREE_GRPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from TREE_GRPO.rollout import collect_tree_grpo_v1_batch

_RUN_LOG_PATH: Optional[str] = None


def _set_run_log_path(path: str) -> None:
    global _RUN_LOG_PATH
    _RUN_LOG_PATH = str(path)


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"{ts} | {msg}"
    print(line)
    if _RUN_LOG_PATH:
        with open(_RUN_LOG_PATH, "a", encoding="utf-8") as f:
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
    p = argparse.ArgumentParser(description="Tree-GRPO shallow tree v1 trainer")
    p.add_argument("--set", action="append", default=[], help="Override config, e.g. train.lr=3e-5")
    return p


def _resolve_strict_vocab_token_id(tokenizer, token_text: str, *, label: str) -> int:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if token_text not in vocab:
        raise RuntimeError(f"{label} token {token_text!r} is missing from tokenizer vocabulary")
    tok_id = int(vocab[token_text])
    validate_single_token(tokenizer, token_text, tok_id, label=label)
    return int(tok_id)


def _save_checkpoint(
    *,
    output_dir: str,
    step: int,
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(os.path.join(ckpt_dir, "model"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(
        {
            "step": int(step),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(ckpt_dir, "tree_grpo_state.pt"),
    )


def _action_stats_tensors_batched_policy_only(
    *,
    model,
    ref_model,
    trajs: Sequence,
    traj_cache: Sequence,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    v_id_to_local: torch.Tensor,
    temperature: float,
    pad_token_id: int,
    token_stats_kernel,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = next(model.parameters()).device
    if not trajs:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty, empty_l

    cache = list(traj_cache)
    if len(cache) != len(trajs):
        raise RuntimeError("traj_cache length must match trajs length")

    max_len = max(c.seq_len for c in cache)
    bsz = len(cache)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, c in enumerate(cache):
        L = c.seq_len
        input_ids[i, :L] = c.seq_ids
        attention_mask[i, :L] = c.attention_mask

    base_model = model.get_submodule(model.base_model_prefix)
    out = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_all = out.last_hidden_state

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model output embeddings (LM head) are unavailable")
    weight = lm_head.weight
    z_w = weight.index_select(0, z_allowed_t)
    d_w = weight.index_select(0, digit_allowed_t)
    v_w = weight.index_select(0, verify_allowed_t)
    bias = getattr(lm_head, "bias", None)
    z_b = bias.index_select(0, z_allowed_t) if bias is not None else None
    d_b = bias.index_select(0, digit_allowed_t) if bias is not None else None
    v_b = bias.index_select(0, verify_allowed_t) if bias is not None else None

    ref_hidden_all: Optional[torch.Tensor] = None
    ref_z_w: Optional[torch.Tensor] = None
    ref_d_w: Optional[torch.Tensor] = None
    ref_v_w: Optional[torch.Tensor] = None
    ref_z_b: Optional[torch.Tensor] = None
    ref_d_b: Optional[torch.Tensor] = None
    ref_v_b: Optional[torch.Tensor] = None
    ref_device: Optional[torch.device] = None
    if ref_model is not None:
        ref_device = next(ref_model.parameters()).device
        ref_input_ids = input_ids if ref_device == device else input_ids.to(ref_device)
        ref_attention_mask = attention_mask if ref_device == device else attention_mask.to(ref_device)
        z_allowed_ref = z_allowed_t if ref_device == device else z_allowed_t.to(ref_device)
        digit_allowed_ref = digit_allowed_t if ref_device == device else digit_allowed_t.to(ref_device)
        verify_allowed_ref = verify_allowed_t if ref_device == device else verify_allowed_t.to(ref_device)
        ref_base_model = ref_model.get_submodule(ref_model.base_model_prefix)
        with torch.no_grad():
            ref_out = ref_base_model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        ref_hidden_all = ref_out.last_hidden_state
        ref_lm_head = ref_model.get_output_embeddings()
        if ref_lm_head is None:
            raise RuntimeError("Reference model output embeddings (LM head) are unavailable")
        ref_weight = ref_lm_head.weight
        ref_z_w = ref_weight.index_select(0, z_allowed_ref)
        ref_d_w = ref_weight.index_select(0, digit_allowed_ref)
        ref_v_w = ref_weight.index_select(0, verify_allowed_ref)
        ref_bias = getattr(ref_lm_head, "bias", None)
        ref_z_b = ref_bias.index_select(0, z_allowed_ref) if ref_bias is not None else None
        ref_d_b = ref_bias.index_select(0, digit_allowed_ref) if ref_bias is not None else None
        ref_v_b = ref_bias.index_select(0, verify_allowed_ref) if ref_bias is not None else None

    lengths_all = torch.tensor([c.action_len for c in cache], dtype=torch.long, device=device)
    nonzero_rows = torch.nonzero(lengths_all > 0, as_tuple=False).squeeze(-1)
    if nonzero_rows.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty, empty_l

    lengths = lengths_all.index_select(0, nonzero_rows)
    hidden_nz = hidden_all.index_select(0, nonzero_rows)

    batch_ids = torch.repeat_interleave(
        torch.arange(nonzero_rows.numel(), device=device, dtype=torch.long),
        lengths,
    )
    state_positions = torch.cat([cache[int(i)].state_positions for i in nonzero_rows.tolist()], dim=0)
    action_ids = torch.cat([cache[int(i)].action_ids for i in nonzero_rows.tolist()], dim=0)
    action_phase = torch.cat([cache[int(i)].action_phase for i in nonzero_rows.tolist()], dim=0)
    logp_old = torch.cat([cache[int(i)].logp_old for i in nonzero_rows.tolist()], dim=0)
    advantages = torch.cat([cache[int(i)].advantages_norm for i in nonzero_rows.tolist()], dim=0)

    h_states = hidden_nz[batch_ids, state_positions]
    logp_new, entropy_new, invalid = token_stats_kernel(
        h_states,
        action_ids,
        action_phase,
        z_w,
        z_b,
        d_w,
        d_b,
        v_w,
        v_b,
        z_id_to_local,
        d_id_to_local,
        v_id_to_local,
        float(temperature),
    )
    if bool(invalid.any()):
        raise RuntimeError("Found actions not in allowed set")

    if ref_model is not None:
        assert ref_hidden_all is not None and ref_z_w is not None and ref_d_w is not None and ref_v_w is not None
        assert ref_device is not None
        if ref_device == device:
            nonzero_rows_ref = nonzero_rows
            batch_ids_ref = batch_ids
            state_positions_ref = state_positions
            action_ids_ref = action_ids
            action_phase_ref = action_phase
            z_id_to_local_ref = z_id_to_local
            d_id_to_local_ref = d_id_to_local
            v_id_to_local_ref = v_id_to_local
        else:
            nonzero_rows_ref = nonzero_rows.to(ref_device)
            batch_ids_ref = batch_ids.to(ref_device)
            state_positions_ref = state_positions.to(ref_device)
            action_ids_ref = action_ids.to(ref_device)
            action_phase_ref = action_phase.to(ref_device)
            z_id_to_local_ref = z_id_to_local.to(ref_device)
            d_id_to_local_ref = d_id_to_local.to(ref_device)
            v_id_to_local_ref = v_id_to_local.to(ref_device)
        ref_hidden_nz = ref_hidden_all.index_select(0, nonzero_rows_ref)
        ref_h_states = ref_hidden_nz[batch_ids_ref, state_positions_ref]
        with torch.no_grad():
            logp_ref, _ref_entropy, ref_invalid = token_stats_kernel(
                ref_h_states,
                action_ids_ref,
                action_phase_ref,
                ref_z_w,
                ref_z_b,
                ref_d_w,
                ref_d_b,
                ref_v_w,
                ref_v_b,
                z_id_to_local_ref,
                d_id_to_local_ref,
                v_id_to_local_ref,
                float(temperature),
            )
        if bool(ref_invalid.any()):
            raise RuntimeError("Reference model found actions not in allowed set")
        if logp_ref.device != device:
            logp_ref = logp_ref.to(device)
    else:
        logp_ref = logp_new.detach()

    return logp_new, logp_ref, entropy_new, logp_old, advantages, lengths


def train(cfg: Config) -> None:
    _set_seed(cfg.train.seed)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "rollouts"), exist_ok=True)
    _set_run_log_path(os.path.join(cfg.train.output_dir, "train.log"))

    torch_device_cfg = str(getattr(cfg.rollout, "torch_device", "cuda:0")).strip()
    device = torch.device(torch_device_cfg if torch.cuda.is_available() else "cpu")
    _log(f"Device={device}")

    ref_device_cfg = str(getattr(cfg.rollout, "ref_model_device", "cuda:1")).strip()
    ref_device = torch.device(ref_device_cfg if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.init_ckpt,
        use_fast=True,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.init_ckpt,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    use_ref_model = float(cfg.ppo.kl_coef) > 0.0
    ref_model: Optional[Any] = None
    if use_ref_model:
        ref_model = copy.deepcopy(model)
        ref_model.to(ref_device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        _log(f"Reference model enabled on {ref_device}")
    else:
        _log("Reference model disabled")

    z_token_ids, z_style = introspect_z_token_ids_and_style(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found in tokenizer")
    if z_style == "upper":
        _log("WARNING: using uppercase <Z_i> fallback")

    answer_token_id = resolve_answer_token_id(tokenizer, answer_token=cfg.model.answer_token)
    validate_answer_token_single(tokenizer, cfg.model.answer_token, answer_token_id)
    finalize_token_id = _resolve_strict_vocab_token_id(tokenizer, str(cfg.model.finalize_token), label="Verify")
    retry_token_id = _resolve_strict_vocab_token_id(tokenizer, str(cfg.model.retry_token), label="Verify")
    if int(finalize_token_id) == int(retry_token_id):
        raise RuntimeError("<FINALIZE> and <RETRY> must have distinct token ids")

    digit_token_ids = resolve_digit_token_ids(tokenizer)

    z_allowed_t = torch.tensor(list(z_token_ids) + [int(answer_token_id)], dtype=torch.long, device=device)
    digit_allowed_t = torch.tensor(list(digit_token_ids), dtype=torch.long, device=device)
    verify_allowed_t = torch.tensor([int(finalize_token_id), int(retry_token_id)], dtype=torch.long, device=device)
    if float(cfg.rollout.verify_temperature) <= 0.0:
        raise ValueError("rollout.verify_temperature must be > 0")
    if float(cfg.rollout.verify_p) <= 0.0 or float(cfg.rollout.verify_p) > 1.0:
        raise ValueError("rollout.verify_p must be in (0, 1]")
    p4 = list(cfg.tree.tree_p4_by_depth)
    p2 = list(cfg.tree.tree_p2_by_depth)
    p1 = list(cfg.tree.tree_p1_by_depth)
    if len(p4) == 0 or len(p2) == 0 or len(p1) == 0:
        raise ValueError("tree_p*_by_depth lists must be non-empty")
    for i in range(max(len(p4), len(p2), len(p1))):
        a = float(p4[min(i, len(p4) - 1)])
        b = float(p2[min(i, len(p2) - 1)])
        c = float(p1[min(i, len(p1) - 1)])
        if a < 0.0 or b < 0.0 or c < 0.0:
            raise ValueError(f"Branching probabilities must be >=0 at depth={i}")
        if abs((a + b + c) - 1.0) > 1e-6:
            raise ValueError(f"Branching probabilities must sum to 1 at depth={i}: got {a+b+c}")
    if int(cfg.tree.max_total_nodes_per_prompt) <= 0:
        raise ValueError("tree.max_total_nodes_per_prompt must be > 0")
    if int(cfg.tree.max_leaves_per_prompt) <= 0:
        raise ValueError("tree.max_leaves_per_prompt must be > 0")
    if int(cfg.tree.max_active_nodes_per_wave) <= 0:
        raise ValueError("tree.max_active_nodes_per_wave must be > 0")
    if int(cfg.tree.max_expanded_retry_nodes_per_level) <= 0:
        raise ValueError("tree.max_expanded_retry_nodes_per_level must be > 0")

    _log(
        f"Tree-GRPO v1 | root_siblings={cfg.tree.root_siblings} | "
        f"tree_p4_by_depth={cfg.tree.tree_p4_by_depth} | "
        f"tree_p2_by_depth={cfg.tree.tree_p2_by_depth} | "
        f"tree_p1_by_depth={cfg.tree.tree_p1_by_depth} | "
        f"max_total_nodes_per_prompt={cfg.tree.max_total_nodes_per_prompt} | "
        f"max_leaves_per_prompt={cfg.tree.max_leaves_per_prompt} | "
        f"max_active_nodes_per_wave={cfg.tree.max_active_nodes_per_wave} | "
        f"max_expanded_retry_nodes_per_level={cfg.tree.max_expanded_retry_nodes_per_level} | "
        f"max_retry_depth={cfg.tree.max_retry_depth} | "
        f"verify_temperature={cfg.rollout.verify_temperature:.4f} | "
        f"verify_p={cfg.rollout.verify_p:.4f} | "
        f"c_retry={cfg.tree.c_retry:.4f} gamma={cfg.tree.gamma:.4f}"
    )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model output embeddings (LM head) are unavailable")
    vocab_size = int(lm_head.weight.size(0))
    z_id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    d_id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    v_id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    z_id_to_local[z_allowed_t] = torch.arange(z_allowed_t.numel(), device=device, dtype=torch.long)
    d_id_to_local[digit_allowed_t] = torch.arange(digit_allowed_t.numel(), device=device, dtype=torch.long)
    v_id_to_local[verify_allowed_t] = torch.arange(verify_allowed_t.numel(), device=device, dtype=torch.long)

    ppo_params = list(model.parameters())
    optimizer = bnb.optim.AdamW8bit(
        ppo_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )

    rollout_backend = str(getattr(cfg.rollout, "backend", "vllm")).strip().lower()
    if rollout_backend not in ("vllm", "hf"):
        raise ValueError(f"Unsupported rollout.backend={cfg.rollout.backend!r}")

    vllm_engine: Optional[Any] = None
    if rollout_backend == "hf":
        vllm_engine = HFRolloutEngine(
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=z_allowed_t.tolist(),
            digit_allowed_token_ids=digit_allowed_t.tolist(),
            verify_allowed_token_ids=verify_allowed_t.tolist(),
            finalize_token_id=int(finalize_token_id),
            retry_token_id=int(retry_token_id),
            sync_every=int(cfg.rollout.vllm_sync_every),
            logger=_log,
        )
    else:
        if not bool(cfg.rollout.vllm_enabled):
            raise ValueError("rollout.vllm_enabled must be True for vllm backend")
        vllm_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
        vllm_kwargs.setdefault("tensor_parallel_size", int(cfg.rollout.vllm_tp_size))
        vllm_kwargs.setdefault("gpu_memory_utilization", float(cfg.rollout.gpu_memory_utilization))
        vllm_kwargs.setdefault("weight_transfer_device", str(device))
        if int(cfg.rollout.vllm_tp_size) == 1:
            vllm_cvd = str(getattr(cfg.rollout, "vllm_cuda_visible_devices", "")).strip()
            if vllm_cvd:
                vllm_kwargs.setdefault("cuda_visible_devices", vllm_cvd)
                _log(f"vLLM CUDA_VISIBLE_DEVICES={vllm_cvd}")

        vllm_seed = int(cfg.rollout.vllm_seed) if cfg.rollout.vllm_seed is not None else int(cfg.train.seed)
        vllm_engine = VLLMRolloutEngine(
            init_ckpt=cfg.model.init_ckpt,
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=z_allowed_t.tolist(),
            digit_allowed_token_ids=digit_allowed_t.tolist(),
            verify_allowed_token_ids=verify_allowed_t.tolist(),
            finalize_token_id=int(finalize_token_id),
            retry_token_id=int(retry_token_id),
            trust_remote_code=bool(cfg.model.trust_remote_code),
            engine_kwargs=vllm_kwargs,
            output_dir=cfg.train.output_dir,
            tmp_ckpt_dir=(
                cfg.rollout.vllm_tmp_ckpt_dir
                if cfg.rollout.vllm_tmp_ckpt_dir
                else os.path.join(cfg.train.output_dir, "vllm_ckpt_latest")
            ),
            sync_every=int(cfg.rollout.vllm_sync_every),
            seed=vllm_seed,
            logger=_log,
        )

    ds = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)
    if len(ds) == 0:
        raise RuntimeError("Training dataset is empty")

    excluded_questions, excluded_questions_path = _load_rsft_trained_questions(
        str(getattr(cfg.data, "rsft_trained_questions_path", ""))
    )
    if excluded_questions_path:
        _log(f"Loaded {len(excluded_questions)} exclusion questions from {excluded_questions_path}")

    train_row_indices: List[int] = []
    q_field = str(cfg.data.question_field)
    removed_count = 0
    for row_idx in range(len(ds)):
        sample = ds[int(row_idx)]
        q_text = _question_text(sample.get(q_field, ""))
        if q_text in excluded_questions:
            removed_count += 1
            continue
        train_row_indices.append(int(row_idx))
    if len(train_row_indices) == 0:
        raise RuntimeError("No training rows remain after filtering")
    if excluded_questions_path:
        _log(f"Dataset filtered: before={len(ds)} after={len(train_row_indices)} removed={removed_count}")

    ds_index = 0
    rollout_logger = RolloutLogger(os.path.join(cfg.train.output_dir, "rollouts"))

    try:
        for update in range(1, int(cfg.train.updates) + 1):
            _t_update0 = time.perf_counter()

            if vllm_engine is not None:
                _ = vllm_engine.maybe_sync_from_torch(model=model, tokenizer=tokenizer, update_idx=update)

            prompts_per_update = max(1, int(cfg.rollout.tree_prompts_per_update))
            prepared: List[Dict[str, object]] = []
            prompt_counter = 0
            while len(prepared) < prompts_per_update:
                sample = ds[int(train_row_indices[ds_index % len(train_row_indices)])]
                ds_index += 1

                question = str(sample[cfg.data.question_field])
                true_digits = _extract_true_digits(
                    sample=sample,
                    answer_digits_field=cfg.data.answer_digits_field,
                    answer_field=cfg.data.answer_field,
                )
                if true_digits is None:
                    continue

                prompt_text = _build_prompt_text(tokenizer, question)
                prompt_pack = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=True)
                prompt_ids = list(prompt_pack["input_ids"])
                prompt_attn = list(prompt_pack.get("attention_mask") or [1] * len(prompt_ids))

                prepared.append(
                    {
                        "sample_id_base": f"u{update}_p{prompt_counter}",
                        "prompt_id": int(prompt_counter),
                        "question": question,
                        "true_digits": [int(x) for x in true_digits],
                        "prompt_text": prompt_text,
                        "prompt_ids": prompt_ids,
                        "prompt_attention_mask": prompt_attn,
                    }
                )
                prompt_counter += 1

            trajectories, tree_stats = collect_tree_grpo_v1_batch(
                model=model,
                tokenizer=tokenizer,
                vllm_engine=vllm_engine,
                prepared=prepared,
                cfg=cfg,
                z_allowed_t=z_allowed_t,
                digit_allowed_t=digit_allowed_t,
                verify_allowed_t=verify_allowed_t,
                answer_token_id=int(answer_token_id),
                finalize_token_id=int(finalize_token_id),
                retry_token_id=int(retry_token_id),
                digit_token_ids=digit_token_ids,
            )

            if not trajectories:
                _log(f"update={update} | no trajectories; skipping")
                continue

            # Tree mode: no global/per-prompt std normalization.
            for t in trajectories:
                t.advantages_norm = list(t.advantages)

            roll_rows: List[Dict[str, object]] = []
            for traj in trajectories:
                row = {
                    "schema_version": 3,
                    "id": traj.sample_id,
                    "prompt_id": int(traj.prompt_id),
                    "question": traj.question,
                    "input_ids": traj.prompt_ids,
                    "generated_z_ids": traj.generated_z_ids,
                    "generated_digit_ids": traj.generated_digit_ids,
                    "generated_verify_ids": traj.generated_verify_ids,
                    "terminated_by": traj.terminated_by,
                    "termination_reason": traj.termination_reason,
                    "num_generated": traj.num_generated_total,
                    "num_digits_generated": traj.num_digits_generated,
                    "digit_pred": traj.digit_pred,
                    "digit_true": traj.digit_true,
                    "full_generated_ids": traj.full_generated_ids,
                    "rounds_meta": traj.rounds_meta,
                    "reward_final": traj.reward_info.get("reward_final", 0.0),
                    "q": traj.reward_info.get("q", None),
                    "Q_F": traj.reward_info.get("Q_F", None),
                    "Q_R": traj.reward_info.get("Q_R", None),
                    "U": traj.reward_info.get("U", None),
                    "V": traj.reward_info.get("V", None),
                    "A_Z": traj.reward_info.get("A_Z", None),
                    "A_V": traj.reward_info.get("A_V", None),
                    "group_id": traj.reward_info.get("group_id", None),
                    "group_type": traj.reward_info.get("group_type", None),
                    "retry_depth": traj.reward_info.get("retry_depth", None),
                    "parent_node_id": traj.reward_info.get("parent_node_id", None),
                    "child_node_ids": traj.reward_info.get("child_node_ids", None),
                    "retry_block_reason": traj.reward_info.get("retry_block_reason", None),
                    "leaf_end_type": traj.reward_info.get("leaf_end_type", None),
                    "was_forced_finalize": traj.reward_info.get("was_forced_finalize", None),
                    "retry_depth_at_leaf": traj.reward_info.get("retry_depth_at_leaf", None),
                    "verify_action_present": traj.reward_info.get("verify_action_present", None),
                    "k_used": traj.reward_info.get("k_used", None),
                    "branching_decision": traj.reward_info.get("branching_decision", None),
                    "has_forced_retry_probe": traj.reward_info.get("has_forced_retry_probe", None),
                    "probe_terminal_value": traj.reward_info.get("probe_terminal_value", None),
                    "probe_terminal_node_id": traj.reward_info.get("probe_terminal_node_id", None),
                    "probe_length_rounds": traj.reward_info.get("probe_length_rounds", None),
                    "probe_leaf_end_type": traj.reward_info.get("probe_leaf_end_type", None),
                    "probe_start_retry_depth": traj.reward_info.get("probe_start_retry_depth", None),
                    "probe_nodes": traj.reward_info.get("probe_nodes", None),
                    "actions": traj.actions,
                    "action_types": traj.action_types,
                    "logp_old": traj.logp_old,
                    "values": traj.values_old,
                    "entropy": traj.entropy_old,
                    "advantages": traj.advantages,
                    "returns": traj.returns,
                }
                if cfg.logging.log_action_tokens:
                    row["action_tokens"] = tokenizer.convert_ids_to_tokens(traj.actions)
                roll_rows.append(row)
            _ = rollout_logger.write_step(step=update, rows=roll_rows)

            optimizer.zero_grad(set_to_none=True)
            trajectory_cache = _build_trajectory_device_cache(trajectories=trajectories, device=device)
            seq_lens = [c.seq_len for c in trajectory_cache]
            token_stats_kernel = _get_token_stats_kernel(
                compile_update_stats=bool(getattr(cfg.runtime, "compile_update_stats", False))
            )

            minibatch_count = 0
            pol_acc = 0.0
            ent_acc = 0.0
            clip_acc = 0.0
            kl_acc = 0.0

            for _epoch in range(int(cfg.ppo.ppo_epochs)):
                order = _build_minibatch_order(
                    seq_lens=seq_lens,
                    use_length_bucketing=bool(getattr(cfg.runtime, "use_length_bucketing", True)),
                    bucket_width=int(getattr(cfg.runtime, "length_bucket_width", 64)),
                )
                for start in range(0, len(order), int(cfg.ppo.minibatch_size)):
                    batch_idx = order[start: start + int(cfg.ppo.minibatch_size)]
                    batch_trajs = [trajectories[idx] for idx in batch_idx]
                    batch_cache = [trajectory_cache[idx] for idx in batch_idx]

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and bool(cfg.runtime.use_bf16)
                        else nullcontext()
                    )
                    with amp_ctx:
                        (
                            logp_new,
                            logp_ref,
                            entropy_new,
                            logp_old,
                            advantages,
                            lengths,
                        ) = _action_stats_tensors_batched_policy_only(
                            model=model,
                            ref_model=ref_model,
                            trajs=batch_trajs,
                            traj_cache=batch_cache,
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            verify_allowed_t=verify_allowed_t,
                            z_id_to_local=z_id_to_local,
                            d_id_to_local=d_id_to_local,
                            v_id_to_local=v_id_to_local,
                            temperature=float(cfg.rollout.temperature),
                            pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0,
                            token_stats_kernel=token_stats_kernel,
                        )

                        if int(lengths.numel()) == 0:
                            continue

                        logp_new_f = logp_new.float()
                        logp_old_f = logp_old.float()
                        logp_ref_f = logp_ref.float()
                        advantages_f = advantages.float()
                        entropy_new_f = entropy_new.float()

                        log_ratio = logp_new_f - logp_old_f
                        ratio = torch.exp(log_ratio)
                        ratio_clipped = torch.clamp(
                            ratio,
                            1.0 - float(cfg.ppo.clip_range),
                            1.0 + float(cfg.ppo.clip_range),
                        )
                        pg1 = ratio * advantages_f
                        pg2 = ratio_clipped * advantages_f
                        policy_loss_tok = -torch.min(pg1, pg2)
                        kl_tok = logp_new_f - logp_ref_f
                        lo = 1.0 - float(cfg.ppo.clip_range)
                        hi = 1.0 + float(cfg.ppo.clip_range)
                        clipped_tok = ((ratio < lo) | (ratio > hi)).float()

                        def _segment_means(values: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
                            if int(lens.numel()) == 0:
                                return torch.empty((0,), dtype=values.dtype, device=values.device)
                            seg_ids = torch.repeat_interleave(
                                torch.arange(lens.numel(), device=values.device, dtype=torch.long),
                                lens,
                            )
                            sums = torch.zeros((lens.numel(),), dtype=values.dtype, device=values.device)
                            sums.scatter_add_(0, seg_ids, values)
                            denom = lens.to(device=values.device, dtype=values.dtype).clamp_min(1.0)
                            return sums / denom

                        policy_loss = _segment_means(policy_loss_tok, lengths).mean()
                        clipfrac = _segment_means(clipped_tok, lengths).mean()
                        entropy_mean = _segment_means(entropy_new_f, lengths).mean()
                        kl_mean = _segment_means(kl_tok, lengths).mean()
                        entropy_loss = -entropy_mean
                        kl_penalty = float(cfg.ppo.kl_coef) * kl_mean

                        loss = (
                            policy_loss
                            + kl_penalty
                            + float(cfg.ppo.c_ent) * entropy_loss
                        ) / float(cfg.train.grad_accum_steps)

                    loss.backward()
                    minibatch_count += 1

                    pol_acc += float(policy_loss.detach().item())
                    ent_acc += float(entropy_mean.detach().item())
                    clip_acc += float(clipfrac.detach().item())
                    kl_acc += float(kl_mean.detach().item())

                    if minibatch_count % int(cfg.train.grad_accum_steps) == 0:
                        torch.nn.utils.clip_grad_norm_(ppo_params, float(cfg.ppo.max_grad_norm))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            if minibatch_count % int(cfg.train.grad_accum_steps) != 0:
                torch.nn.utils.clip_grad_norm_(ppo_params, float(cfg.ppo.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            ref_refresh_every = max(int(cfg.ppo.update_ref_model_each_steps), 1)
            if ref_model is not None and update % ref_refresh_every == 0:
                ref_model.load_state_dict(model.state_dict())
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)

            denom = float(max(minibatch_count, 1))
            t_total = time.perf_counter() - _t_update0
            _log(
                " | ".join(
                    [
                        f"update={update}",
                        f"nodes={int(tree_stats.get('num_nodes', 0.0))}",
                        f"retry_nodes={int(tree_stats.get('num_retry_nodes', 0.0))}",
                        f"mean_q={tree_stats.get('mean_q', 0.0):.4f}",
                        f"mean_u={tree_stats.get('mean_u', 0.0):.4f}",
                        f"mean_v={tree_stats.get('mean_v', 0.0):.4f}",
                        f"mean_az={tree_stats.get('mean_az', 0.0):.4f}",
                        f"mean_av={tree_stats.get('mean_av', 0.0):.4f}",
                        f"policy_loss={pol_acc / denom:.4f}",
                        f"entropy={ent_acc / denom:.4f}",
                        f"clipfrac={clip_acc / denom:.4f}",
                        f"kl={kl_acc / denom:.4f}",
                        f"t_update={t_total:.3f}s",
                    ]
                )
            )

            if update % int(cfg.train.save_every) == 0:
                _save_checkpoint(
                    output_dir=cfg.train.output_dir,
                    step=update,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                )
    finally:
        if vllm_engine is not None:
            vllm_engine.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = Config()
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set expression: {item}")
        key, value = item.split("=", 1)
        _apply_override(cfg, key.strip(), value.strip())

    train(cfg)


if __name__ == "__main__":
    main()
