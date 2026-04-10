from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from PPO.hf_rollout import HFRolloutEngine
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.ppo_math import explained_variance
from PPO.reward import compute_multi_round_reward, compute_reward, parse_answer_digits, parse_final_answer_to_digits
from PPO.rollout_contract import is_ppo_action, validate_action_scope
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single, validate_single_token
from PPO.vllm_rollout import VLLMRolloutEngine
from phase1.dataset import SYSTEM_PROMPT

_REWARD_TIME_ACC_SEC: float = 0.0
_RUN_LOG_PATH: Optional[str] = None
_COMPILED_TOKEN_STATS_KERNEL: Optional[Any] = None


def _set_run_log_path(path: str) -> None:
    global _RUN_LOG_PATH
    _RUN_LOG_PATH = path


def _reset_reward_timing_acc() -> None:
    global _REWARD_TIME_ACC_SEC
    _REWARD_TIME_ACC_SEC = 0.0


def _add_reward_timing_acc(delta_sec: float) -> None:
    global _REWARD_TIME_ACC_SEC
    _REWARD_TIME_ACC_SEC += float(delta_sec)


def _get_reward_timing_acc() -> float:
    return float(_REWARD_TIME_ACC_SEC)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class Trajectory:
    def __init__(
        self,
        *,
        prompt_id: int,
        sample_id: str,
        question: str,
        prompt_ids: List[int],
        prompt_attention_mask: List[int],
        actions: List[int],
        action_types: List[str],
        logp_old: List[float],
        values_old: List[float],
        entropy_old: List[float],
        terminated_by: str,
        generated_z_ids: List[int],
        generated_digit_ids: List[int],
        digit_logits: Optional[List[List[float]]],
        digit_probs: Optional[List[List[float]]],
        digit_pred: Optional[List[int]],
        digit_true: List[int],
        reward_info: Dict[str, object],
        num_generated_total: int,
        num_digits_generated: int,
        generated_verify_ids: Optional[List[int]] = None,
        rounds_meta: Optional[List[Dict[str, object]]] = None,
        full_generated_ids: Optional[List[int]] = None,
        termination_reason: Optional[str] = None,
    ) -> None:
        self.prompt_id = int(prompt_id)
        self.sample_id = sample_id
        self.question = question
        self.prompt_ids = prompt_ids
        self.prompt_attention_mask = prompt_attention_mask
        self.actions = actions
        self.action_types = action_types
        self.logp_old = logp_old
        self.values_old = values_old
        self.entropy_old = entropy_old
        self.terminated_by = terminated_by
        self.generated_z_ids = generated_z_ids
        self.generated_digit_ids = generated_digit_ids
        self.digit_logits = digit_logits
        self.digit_probs = digit_probs
        self.digit_pred = digit_pred
        self.digit_true = digit_true
        self.reward_info = reward_info
        self.num_generated_total = int(num_generated_total)
        self.num_digits_generated = int(num_digits_generated)
        self.generated_verify_ids = list(generated_verify_ids or [])
        self.rounds_meta = list(rounds_meta or [])
        self.full_generated_ids = list(full_generated_ids or [])
        self.termination_reason = str(termination_reason) if termination_reason is not None else str(terminated_by)

        self.returns = [float(self.reward_info["reward_final"])] * len(actions)
        self.advantages = [float(self.reward_info["reward_final"]) - float(v) for v in values_old]
        self.advantages_norm_global: List[float] = []
        self.advantages_norm_prompt: List[float] = []
        self.advantages_norm: List[float] = []


@dataclass
class TrajectoryDeviceCache:
    seq_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_ids: torch.Tensor
    action_phase: torch.Tensor
    state_positions: torch.Tensor
    prompt_len: int
    seq_len: int
    action_len: int
    logp_old: torch.Tensor
    advantages_norm: torch.Tensor
    returns: torch.Tensor


def _action_type_to_phase_id(action_type: str) -> int:
    t = str(action_type)
    if t in ("z", "answer"):
        return 0
    if t == "digit":
        return 1
    if t == "verify":
        return 2
    raise RuntimeError(f"Unsupported action_type {action_type!r}")


def _build_trajectory_device_cache(
    *,
    trajectories: Sequence[Trajectory],
    device: torch.device,
) -> List[TrajectoryDeviceCache]:
    cached: List[TrajectoryDeviceCache] = []
    for traj in trajectories:
        prompt_len = int(len(traj.prompt_ids))
        action_len = int(len(traj.actions))
        if action_len != len(traj.action_types):
            raise RuntimeError("Trajectory actions/action_types length mismatch")
        if len(traj.advantages_norm) != action_len:
            raise AssertionError("advantages_norm length must match action length before cache build")

        seq_ids = torch.tensor(list(traj.prompt_ids) + list(traj.actions), dtype=torch.long, device=device)
        attn = torch.tensor(
            list(traj.prompt_attention_mask) + [1] * action_len,
            dtype=torch.long,
            device=device,
        )
        if action_len > 0:
            state_positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + action_len,
                device=device,
                dtype=torch.long,
            )
        else:
            state_positions = torch.empty((0,), dtype=torch.long, device=device)

        cached.append(
            TrajectoryDeviceCache(
                seq_ids=seq_ids,
                attention_mask=attn,
                action_ids=torch.tensor(traj.actions, dtype=torch.long, device=device),
                action_phase=torch.tensor(
                    [_action_type_to_phase_id(t) for t in traj.action_types],
                    dtype=torch.long,
                    device=device,
                ),
                state_positions=state_positions,
                prompt_len=prompt_len,
                seq_len=int(seq_ids.numel()),
                action_len=action_len,
                logp_old=torch.tensor(traj.logp_old, dtype=torch.float32, device=device),
                advantages_norm=torch.tensor(traj.advantages_norm, dtype=torch.float32, device=device),
                returns=torch.tensor(traj.returns, dtype=torch.float32, device=device),
            )
        )
    return cached


def _build_minibatch_order(
    *,
    seq_lens: Sequence[int],
    use_length_bucketing: bool,
    bucket_width: int,
) -> List[int]:
    indices = list(range(len(seq_lens)))
    if not use_length_bucketing or len(indices) <= 1:
        random.shuffle(indices)
        return indices

    width = max(int(bucket_width), 1)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        key = int(seq_lens[idx]) // width
        buckets[key].append(idx)

    ordered: List[int] = []
    bucket_keys = list(buckets.keys())
    random.shuffle(bucket_keys)
    for key in bucket_keys:
        rows = buckets[key]
        random.shuffle(rows)
        ordered.extend(rows)
    return ordered


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


def _set_value_warmup_trainability(model, value_head: ValueHead, enabled: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(not enabled)
    for p in value_head.parameters():
        p.requires_grad_(True)


def _assert_value_warmup_trainability(model, value_head: ValueHead, enabled: bool) -> None:
    backbone_trainable = [bool(p.requires_grad) for p in model.parameters()]
    value_head_trainable = [bool(p.requires_grad) for p in value_head.parameters()]
    if enabled:
        if any(backbone_trainable):
            raise AssertionError("Warmup active but backbone params require grad")
    if not value_head_trainable or not all(value_head_trainable):
        raise AssertionError("Value head params must be trainable")


def _optimizer_param_id_set(optimizer: torch.optim.Optimizer) -> set[int]:
    ids: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            ids.add(id(p))
    return ids


def _assert_optimizer_matches_params(optimizer: torch.optim.Optimizer, expected_params: Sequence[torch.nn.Parameter]) -> None:
    expected = {id(p) for p in expected_params}
    actual = _optimizer_param_id_set(optimizer)
    if actual != expected:
        raise AssertionError("Optimizer parameter groups do not match intended phase parameters")


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
    p = argparse.ArgumentParser(description="Phase-4 PPO trainer")
    p.add_argument("--set", action="append", default=[], help="Override config, e.g. train.lr=3e-5")
    return p


def _make_rng(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _build_prompt_text(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _nucleus_sample_from_probs(probs: torch.Tensor, top_p: float) -> int:
    if top_p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    keep = cum_probs <= top_p
    keep[0] = True
    kept_probs = sorted_probs * keep.to(dtype=sorted_probs.dtype)
    kept_sum = kept_probs.sum()
    if float(kept_sum.item()) <= 0.0:
        kept_probs = sorted_probs
        kept_sum = kept_probs.sum()
    kept_probs = kept_probs / kept_sum
    sampled_in_sorted = int(torch.multinomial(kept_probs, num_samples=1).item())
    return int(sorted_idx[sampled_in_sorted].item())


def _sample_action_from_allowed_logits(
    allowed_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, float]:
    if temperature <= 0:
        raise ValueError("rollout.temperature must be > 0")
    if top_p <= 0 or top_p > 1:
        raise ValueError("rollout.top_p must be in (0, 1]")

    logits = allowed_logits / float(temperature)
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = float((-(probs * logp).sum()).item())
    if greedy:
        local_idx = int(torch.argmax(logits, dim=-1).item())
    else:
        local_idx = _nucleus_sample_from_probs(probs, top_p=top_p)
    return local_idx, logp, probs, entropy


def _forward_last_with_cache(core, input_ids, attention_mask, past_key_values):
    return core(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        output_hidden_states=True,
        return_dict=True,
    )


def _extract_true_digits(sample: Dict[str, object], answer_digits_field: str, answer_field: str) -> Optional[List[int]]:
    if answer_digits_field in sample:
        parsed = parse_answer_digits(sample.get(answer_digits_field))
        if parsed is not None:
            return parsed
    return parse_final_answer_to_digits(sample.get(answer_field))


def _question_text(raw: object) -> str:
    return str(raw).strip()


def _extract_questions_from_json_payload(payload: object) -> List[str]:
    rows: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                txt = _question_text(item)
                if txt:
                    rows.append(txt)
            elif isinstance(item, dict):
                if "question" in item:
                    txt = _question_text(item.get("question"))
                    if txt:
                        rows.append(txt)
                elif "problem" in item:
                    txt = _question_text(item.get("problem"))
                    if txt:
                        rows.append(txt)
    elif isinstance(payload, dict):
        if "questions" in payload:
            rows.extend(_extract_questions_from_json_payload(payload.get("questions")))
        elif "trained_questions" in payload:
            rows.extend(_extract_questions_from_json_payload(payload.get("trained_questions")))
        elif "items" in payload:
            rows.extend(_extract_questions_from_json_payload(payload.get("items")))
        elif "data" in payload:
            rows.extend(_extract_questions_from_json_payload(payload.get("data")))
        elif "question" in payload:
            txt = _question_text(payload.get("question"))
            if txt:
                rows.append(txt)
        elif "problem" in payload:
            txt = _question_text(payload.get("problem"))
            if txt:
                rows.append(txt)
    return rows


def _load_rsft_trained_questions(path: str) -> Tuple[set[str], str]:
    p = str(path).strip()
    if not p:
        return set(), ""
    abs_path = os.path.abspath(os.path.expanduser(p))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Configured data.rsft_trained_questions_path does not exist or is not a file: {abs_path}"
        )
    with open(abs_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = [q for q in _extract_questions_from_json_payload(payload) if q]
    return set(questions), abs_path


def _resolve_strict_vocab_token_id(tokenizer, token_text: str, *, label: str) -> int:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if token_text not in vocab:
        raise RuntimeError(f"{label} token {token_text!r} is missing from tokenizer vocabulary")
    tok_id = int(vocab[token_text])
    validate_single_token(tokenizer, token_text, tok_id, label=label)
    return int(tok_id)


def _should_run_debug_restricted_logits_check(cfg: Config) -> bool:
    env = os.getenv("PPO_DEBUG_RESTRICTED_LOGITS_CHECK", "").strip().lower()
    env_on = env in ("1", "true", "yes", "y", "on")
    return bool(cfg.runtime.debug_restricted_logits_check) or env_on


def _debug_restricted_logits_check_once(
    *,
    model,
    tokenizer,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    z_w: torch.Tensor,
    d_w: torch.Tensor,
    z_b: Optional[torch.Tensor],
    d_b: Optional[torch.Tensor],
    use_bf16: bool = True,
) -> None:
    """
    Debug invariant we actually care about:
      restricted_proj(h) == lm_head(h) sliced to the same allowed ids

    NOTE:
      Some models (incl. Qwen2.5 variants) can produce `model(...).logits`
      via a path that doesn't exactly equal `lm_head(base.last_hidden_state)`.
      So we log full-vs-lm_head for awareness, but we DO NOT fail on it.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        # Build a tiny prompt
        prompt = _build_prompt_text(tokenizer, "Compute 1+1.")
        pack = tokenizer(prompt, add_special_tokens=False, return_attention_mask=True)
        ids = list(pack.get("input_ids", []))
        if not ids:
            fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            ids = [int(fallback_id)]
        att = list(pack.get("attention_mask") or [1] * len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(att, dtype=torch.long, device=device).unsqueeze(0)
        pos = int((attention_mask[0].sum() - 1).clamp(min=0).item())

        base_model = model.get_submodule(model.base_model_prefix)
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Model output embeddings (LM head) are unavailable")

        # Use same compute dtype as lm_head weights (usually bf16 on GPU)
        compute_dtype = lm_head.weight.dtype
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and bool(use_bf16)
            else nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            # Get post-final-norm hidden states from base model (what lm_head usually consumes)
            base = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            h = base.last_hidden_state[0, pos, :].to(compute_dtype)  # [H]

            # lm_head logits at this position (ground truth for our restricted projection)
            lm_logits = lm_head(h)  # [V]

            # Our restricted projection logits
            z_logits_dbg = h @ z_w.to(compute_dtype).t()
            d_logits_dbg = h @ d_w.to(compute_dtype).t()
            if z_b is not None:
                z_logits_dbg = z_logits_dbg + z_b.to(compute_dtype)
            if d_b is not None:
                d_logits_dbg = d_logits_dbg + d_b.to(compute_dtype)

            # Slice lm_head to allowed sets
            lm_z = lm_logits.index_select(0, z_allowed_t)
            lm_d = lm_logits.index_select(0, digit_allowed_t)

            # Compare in fp32 for stable diff reporting
            lmhead_vs_dbg_z = float((lm_z.float() - z_logits_dbg.float()).abs().max().item())
            lmhead_vs_dbg_d = float((lm_d.float() - d_logits_dbg.float()).abs().max().item())

            # Optional: log how model(...).logits compares to lm_head(h) (awareness only)
            full_vs_lmhead = None
            full_vs_dbg_z = None
            full_vs_dbg_d = None
            try:
                full = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                full_logits = full.logits[0, pos, :]  # [V] (compute dtype)
                full_vs_lmhead = float((full_logits.float() - lm_logits.float()).abs().max().item())
                full_z = full_logits.index_select(0, z_allowed_t)
                full_d = full_logits.index_select(0, digit_allowed_t)
                full_vs_dbg_z = float((full_z.float() - z_logits_dbg.float()).abs().max().item())
                full_vs_dbg_d = float((full_d.float() - d_logits_dbg.float()).abs().max().item())
            except Exception:
                pass

        # Tolerances
        tol = 2e-3 if (device.type == "cuda" and compute_dtype == torch.bfloat16) else 1e-4

        # Log everything
        if full_vs_lmhead is None:
            _log(
                "Restricted-logits debug check | "
                f"dtype={str(compute_dtype)} | "
                f"lmhead_vs_dbg_z={lmhead_vs_dbg_z:.6f} | lmhead_vs_dbg_d={lmhead_vs_dbg_d:.6f} | "
                f"tol={tol:.6f}"
            )
        else:
            _log(
                "Restricted-logits debug check | "
                f"dtype={str(compute_dtype)} | "
                f"full_vs_lmhead={full_vs_lmhead:.6f} | "
                f"full_vs_dbg_z={float(full_vs_dbg_z):.6f} | full_vs_dbg_d={float(full_vs_dbg_d):.6f} | "
                f"lmhead_vs_dbg_z={lmhead_vs_dbg_z:.6f} | lmhead_vs_dbg_d={lmhead_vs_dbg_d:.6f} | "
                f"tol={tol:.6f}"
            )

        # The ONLY correctness condition we enforce:
        if lmhead_vs_dbg_z >= tol or lmhead_vs_dbg_d >= tol:
            raise RuntimeError(
                "Restricted projection mismatch vs lm_head(h) too large "
                f"(lmhead_vs_dbg_z={lmhead_vs_dbg_z:.6f}, lmhead_vs_dbg_d={lmhead_vs_dbg_d:.6f}, tol={tol:.6f}). "
                "This indicates your restricted projection path is not equivalent to lm_head on the same hidden state."
            )

    finally:
        model.train(was_training)
def _action_stats_tensors(
    *,
    model,
    value_head: ValueHead,
    prompt_ids: Sequence[int],
    prompt_attention_mask: Sequence[int],
    actions: Sequence[int],
    action_types: Sequence[str],
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    if len(actions) == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return empty, empty, empty

    seq_ids = torch.tensor(list(prompt_ids) + list(actions), dtype=torch.long, device=device).unsqueeze(0)
    full_attn_list = list(prompt_attention_mask) + [1] * len(actions)
    attn = torch.tensor(full_attn_list, dtype=torch.long, device=device).unsqueeze(0)

    out = model(
        input_ids=seq_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    p_len = len(prompt_ids)
    t_steps = len(actions)
    state_positions = torch.arange(
        p_len - 1,
        p_len - 1 + t_steps,
        device=device,
        dtype=torch.long,
    )

    logits_all = out.logits[0]
    hidden_all = out.hidden_states[-1][0]

    logp_list: List[torch.Tensor] = []
    entropy_list: List[torch.Tensor] = []

    for i in range(t_steps):
        pos = int(state_positions[i].item())
        action_id = int(actions[i])
        action_type = str(action_types[i])
        if action_type == "digit":
            allowed_t = digit_allowed_t
        elif action_type == "verify":
            allowed_t = verify_allowed_t
        else:
            allowed_t = z_allowed_t

        allowed_logits = logits_all[pos].index_select(0, allowed_t) / float(temperature)
        log_probs_allowed = torch.log_softmax(allowed_logits, dim=-1)
        probs_allowed = log_probs_allowed.exp()

        local_matches = torch.nonzero(allowed_t == action_id, as_tuple=False)
        if local_matches.numel() == 0:
            raise RuntimeError(f"Action id {action_id} not in allowed set for type={action_type}")
        local_idx = int(local_matches[0].item())

        logp_list.append(log_probs_allowed[local_idx])
        entropy_list.append((-(probs_allowed * log_probs_allowed).sum()))

    h_states = hidden_all.index_select(0, state_positions)
    values = value_head(h_states.float()).squeeze(-1)
    logp = torch.stack(logp_list, dim=0)
    entropy = torch.stack(entropy_list, dim=0)
    return logp, values, entropy


def _token_stats_from_hidden(
    hidden_states: torch.Tensor,
    action_ids: torch.Tensor,
    action_phase: torch.Tensor,
    z_w: torch.Tensor,
    z_b: Optional[torch.Tensor],
    d_w: torch.Tensor,
    d_b: Optional[torch.Tensor],
    v_w: torch.Tensor,
    v_b: Optional[torch.Tensor],
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    v_id_to_local: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_z = z_id_to_local[action_ids]
    local_d = d_id_to_local[action_ids]
    local_v = v_id_to_local[action_ids]
    is_z = action_phase == 0
    is_d = action_phase == 1
    is_v = action_phase == 2
    invalid = (is_z & (local_z < 0)) | (is_d & (local_d < 0)) | (is_v & (local_v < 0))

    z_logits = (hidden_states @ z_w.t()) / float(temperature)
    d_logits = (hidden_states @ d_w.t()) / float(temperature)
    v_logits = (hidden_states @ v_w.t()) / float(temperature)
    if z_b is not None:
        z_logits = z_logits + z_b
    if d_b is not None:
        d_logits = d_logits + d_b
    if v_b is not None:
        v_logits = v_logits + v_b

    z_logp = torch.log_softmax(z_logits, dim=-1)
    d_logp = torch.log_softmax(d_logits, dim=-1)
    v_logp = torch.log_softmax(v_logits, dim=-1)
    z_probs = z_logp.exp()
    d_probs = d_logp.exp()
    v_probs = v_logp.exp()

    local_z_safe = local_z.clamp_min(0)
    local_d_safe = local_d.clamp_min(0)
    local_v_safe = local_v.clamp_min(0)
    z_chosen = z_logp.gather(1, local_z_safe.view(-1, 1)).squeeze(1)
    d_chosen = d_logp.gather(1, local_d_safe.view(-1, 1)).squeeze(1)
    v_chosen = v_logp.gather(1, local_v_safe.view(-1, 1)).squeeze(1)
    z_ent = -(z_probs * z_logp).sum(dim=-1)
    d_ent = -(d_probs * d_logp).sum(dim=-1)
    v_ent = -(v_probs * v_logp).sum(dim=-1)

    logp_vec = torch.where(is_z, z_chosen, torch.where(is_d, d_chosen, v_chosen))
    entropy_vec = torch.where(is_z, z_ent, torch.where(is_d, d_ent, v_ent))
    return logp_vec, entropy_vec, invalid


def _get_token_stats_kernel(*, compile_update_stats: bool):
    global _COMPILED_TOKEN_STATS_KERNEL
    if not compile_update_stats:
        return _token_stats_from_hidden
    if not hasattr(torch, "compile"):
        return _token_stats_from_hidden
    if _COMPILED_TOKEN_STATS_KERNEL is not None:
        return _COMPILED_TOKEN_STATS_KERNEL
    try:
        _COMPILED_TOKEN_STATS_KERNEL = torch.compile(_token_stats_from_hidden, dynamic=True)
    except Exception:
        _COMPILED_TOKEN_STATS_KERNEL = _token_stats_from_hidden
    return _COMPILED_TOKEN_STATS_KERNEL


def _segment_means(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if int(lengths.numel()) == 0:
        return torch.empty((0,), dtype=values.dtype, device=values.device)
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=values.device, dtype=torch.long),
        lengths,
    )
    sums = torch.zeros((lengths.numel(),), dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, segment_ids, values)
    denom = lengths.to(device=values.device, dtype=values.dtype).clamp_min(1.0)
    return sums / denom


def _action_stats_tensors_batched(
    *,
    model,
    ref_model,
    value_head: ValueHead,
    trajs: Sequence[Trajectory],
    traj_cache: Optional[Sequence[TrajectoryDeviceCache]],
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
    torch.Tensor,
    torch.Tensor,
]:
    device = next(model.parameters()).device
    if not trajs:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty, empty, empty, empty_l

    if traj_cache is None:
        cache = _build_trajectory_device_cache(trajectories=trajs, device=device)
    else:
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

    hidden_all = out.last_hidden_state  # [B,L,H]
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model output embeddings (LM head) are unavailable")
    weight = lm_head.weight
    z_w = weight.index_select(0, z_allowed_t)  # [|Z|,H]
    d_w = weight.index_select(0, digit_allowed_t)  # [|D|,H]
    v_w = weight.index_select(0, verify_allowed_t)  # [|V|,H]
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
    if ref_model is not None:
        ref_base_model = ref_model.get_submodule(ref_model.base_model_prefix)
        with torch.no_grad():
            ref_out = ref_base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        ref_hidden_all = ref_out.last_hidden_state  # [B,L,H]
        ref_lm_head = ref_model.get_output_embeddings()
        if ref_lm_head is None:
            raise RuntimeError("Reference model output embeddings (LM head) are unavailable")
        ref_weight = ref_lm_head.weight
        ref_z_w = ref_weight.index_select(0, z_allowed_t)  # [|Z|,H]
        ref_d_w = ref_weight.index_select(0, digit_allowed_t)  # [|D|,H]
        ref_v_w = ref_weight.index_select(0, verify_allowed_t)  # [|V|,H]
        ref_bias = getattr(ref_lm_head, "bias", None)
        ref_z_b = ref_bias.index_select(0, z_allowed_t) if ref_bias is not None else None
        ref_d_b = ref_bias.index_select(0, digit_allowed_t) if ref_bias is not None else None
        ref_v_b = ref_bias.index_select(0, verify_allowed_t) if ref_bias is not None else None

    lengths_all = torch.tensor([c.action_len for c in cache], dtype=torch.long, device=device)
    nonzero_rows = torch.nonzero(lengths_all > 0, as_tuple=False).squeeze(-1)
    if nonzero_rows.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty, empty, empty, empty_l

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
    returns = torch.cat([cache[int(i)].returns for i in nonzero_rows.tolist()], dim=0)

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
        bad_idx = torch.nonzero(invalid, as_tuple=False).squeeze(-1)[:8]
        bad_ids = action_ids.index_select(0, bad_idx).tolist()
        bad_types = []
        for x in action_phase.index_select(0, bad_idx).tolist():
            if int(x) == 1:
                bad_types.append("digit")
            elif int(x) == 2:
                bad_types.append("verify")
            else:
                bad_types.append("z_or_answer")
        raise RuntimeError(f"Found actions not in allowed set: ids={bad_ids}, types={bad_types}")

    if ref_model is not None:
        assert ref_hidden_all is not None and ref_z_w is not None and ref_d_w is not None and ref_v_w is not None
        ref_hidden_nz = ref_hidden_all.index_select(0, nonzero_rows)
        ref_h_states = ref_hidden_nz[batch_ids, state_positions]
        with torch.no_grad():
            logp_ref, _ref_entropy, ref_invalid = token_stats_kernel(
                ref_h_states,
                action_ids,
                action_phase,
                ref_z_w,
                ref_z_b,
                ref_d_w,
                ref_d_b,
                ref_v_w,
                ref_v_b,
                z_id_to_local,
                d_id_to_local,
                v_id_to_local,
                float(temperature),
            )
        if bool(ref_invalid.any()):
            raise RuntimeError("Reference model found actions not in allowed set")
    else:
        logp_ref = logp_new.detach()

    values_new = value_head(h_states.float()).squeeze(-1)

    return (
        logp_new,
        logp_ref,
        values_new,
        entropy_new,
        logp_old,
        advantages,
        returns,
        lengths,
    )


def _validate_actions_in_allowed(
    *,
    actions: Sequence[int],
    action_types: Sequence[str],
    z_allowed_set: set[int],
    digit_allowed_set: set[int],
    verify_allowed_set: set[int],
) -> None:
    if len(actions) != len(action_types):
        raise RuntimeError("actions/action_types length mismatch")
    for a, t in zip(actions, action_types):
        aid = int(a)
        if t == "digit":
            if aid not in digit_allowed_set:
                raise RuntimeError(f"Digit action id {aid} not in digit allowed set")
        elif t == "verify":
            if aid not in verify_allowed_set:
                raise RuntimeError(f"Verify action id {aid} not in verify allowed set")
        elif t in ("z", "answer"):
            if aid not in z_allowed_set:
                raise RuntimeError(f"Z/answer action id {aid} not in Z allowed set")
        else:
            raise RuntimeError(f"Unsupported action type {t!r}")


def _rollout_one_torch(
    *,
    model,
    value_head: ValueHead,
    tokenizer,
    prompt_id: int,
    question: str,
    true_digits: Sequence[int],
    z_token_ids: Sequence[int],
    digit_token_ids: Sequence[int],
    answer_token_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    action_scope: str,
    digit_greedy: bool,
    reward_cfg,
    reward_rng: torch.Generator,
    sample_id: str,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
) -> Trajectory:
    prompt_text = _build_prompt_text(tokenizer, question)
    prompt_pack = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=True)
    prompt_ids = prompt_pack["input_ids"]
    prompt_attn = prompt_pack.get("attention_mask") or [1] * len(prompt_ids)

    device = next(model.parameters()).device
    seq = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_attn = torch.tensor(prompt_attn, dtype=torch.long, device=device).unsqueeze(0)

    actions: List[int] = []
    action_types: List[str] = []
    generated_z_ids: List[int] = []
    generated_digit_ids: List[int] = []
    digit_logits_rows: List[List[float]] = []
    digit_probs_rows: List[List[float]] = []
    phase = "z"
    terminated_by = "max_new_tokens"

    with torch.no_grad():
        out0 = model(
            input_ids=seq,
            attention_mask=seq_attn,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out0.past_key_values
        last_logits = out0.logits[:, -1, :].squeeze(0)

        for _ in range(max_new_tokens):
            if phase == "z":
                local_idx, _logp_allowed, _probs_allowed, _entropy = _sample_action_from_allowed_logits(
                    last_logits.index_select(0, z_allowed_t),
                    temperature=temperature,
                    top_p=top_p,
                    greedy=False,
                )
                action = int(z_allowed_t[local_idx].item())
                actions.append(action)
                action_types.append("answer" if action == answer_token_id else "z")
                if action == answer_token_id:
                    phase = "digits"
                else:
                    generated_z_ids.append(action)
            else:
                local_idx, _logp_allowed, probs_allowed, _entropy = _sample_action_from_allowed_logits(
                    last_logits.index_select(0, digit_allowed_t),
                    temperature=temperature,
                    top_p=top_p,
                    greedy=bool(digit_greedy),
                )
                action = int(digit_allowed_t[local_idx].item())
                generated_digit_ids.append(action)
                digit_logits_rows.append(last_logits.index_select(0, digit_allowed_t).float().cpu().tolist())
                digit_probs_rows.append(probs_allowed.float().cpu().tolist())

                if is_ppo_action(action_scope, "digits"):
                    actions.append(action)
                    action_types.append("digit")

                if len(generated_digit_ids) == 5:
                    terminated_by = "answer_with_5_digits"
                    break

            action_t = torch.tensor([[action]], dtype=torch.long, device=device)
            seq = torch.cat([seq, action_t], dim=1)
            seq_attn = torch.cat([seq_attn, torch.ones((1, 1), dtype=seq_attn.dtype, device=device)], dim=1)

            out1 = _forward_last_with_cache(
                core=model,
                input_ids=action_t,
                attention_mask=seq_attn,
                past_key_values=past,
            )
            past = out1.past_key_values
            last_logits = out1.logits[:, -1, :].squeeze(0)
        else:
            if phase == "digits" and len(generated_digit_ids) < 5:
                terminated_by = "max_new_tokens_during_digits"

    if phase == "z" and terminated_by != "answer_with_5_digits":
        terminated_by = "max_new_tokens"

    pred_digits: Optional[List[int]] = None
    if terminated_by == "answer_with_5_digits":
        id2d = {int(tok): i for i, tok in enumerate(digit_token_ids)}
        pred_digits = [int(id2d[x]) for x in generated_digit_ids]

    z_allowed_set = set(int(x) for x in z_token_ids + [answer_token_id])
    digit_allowed_set = set(int(x) for x in digit_token_ids)
    verify_allowed_set = set(int(x) for x in verify_allowed_t.tolist())
    _validate_actions_in_allowed(
        actions=actions,
        action_types=action_types,
        z_allowed_set=z_allowed_set,
        digit_allowed_set=digit_allowed_set,
        verify_allowed_set=verify_allowed_set,
    )

    logp_t, values_t, entropy_t = _action_stats_tensors(
        model=model,
        value_head=value_head,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        actions=actions,
        action_types=action_types,
        z_allowed_t=z_allowed_t,
        digit_allowed_t=digit_allowed_t,
        verify_allowed_t=verify_allowed_t,
        temperature=temperature,
    )

    _t_reward0 = time.perf_counter()
    reward_info = compute_reward(
        pred_digits=pred_digits,
        true_digits=true_digits,
        terminated_reason=terminated_by,
        partial_scale=reward_cfg.partial_scale,
        keep_prob=reward_cfg.keep_prob,
        length_penalty=reward_cfg.length_penalty,
        correct_length_discount=reward_cfg.correct_length_discount,
        reward_if_max_len=reward_cfg.reward_if_max_len,
        num_generated_tokens=int(seq.size(1) - len(prompt_ids)),
        generator=reward_rng,
    )
    _add_reward_timing_acc(time.perf_counter() - _t_reward0)

    return Trajectory(
        prompt_id=prompt_id,
        sample_id=sample_id,
        question=question,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        actions=actions,
        action_types=action_types,
        logp_old=logp_t.float().cpu().tolist(),
        values_old=values_t.float().cpu().tolist(),
        entropy_old=entropy_t.float().cpu().tolist(),
        terminated_by=terminated_by,
        generated_z_ids=generated_z_ids,
        generated_digit_ids=generated_digit_ids,
        digit_logits=digit_logits_rows if pred_digits is not None else None,
        digit_probs=digit_probs_rows if pred_digits is not None else None,
        digit_pred=pred_digits,
        digit_true=[int(x) for x in true_digits],
        reward_info=reward_info,
        num_generated_total=int(seq.size(1) - len(prompt_ids)),
        num_digits_generated=len(generated_digit_ids),
    )


def _build_trajectory_from_vllm_tokens(
    *,
    model,
    value_head: ValueHead,
    tokenizer,
    prompt_id: int,
    question: str,
    true_digits: Sequence[int],
    prompt_ids: Sequence[int],
    prompt_attention_mask: Sequence[int],
    z_prefix_ids: Sequence[int],
    has_answer: bool,
    digit_ids: Sequence[int],
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    action_scope: str,
    reward_cfg,
    reward_rng: torch.Generator,
    sample_id: str,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    temperature: float,
    terminated_by: str,
) -> Trajectory:
    generated_z_ids = [int(x) for x in z_prefix_ids]
    generated_digit_ids = [int(x) for x in digit_ids]

    actions: List[int] = list(generated_z_ids)
    action_types: List[str] = ["z"] * len(generated_z_ids)
    if has_answer:
        actions.append(int(answer_token_id))
        action_types.append("answer")

    if is_ppo_action(action_scope, "digits"):
        actions.extend(generated_digit_ids)
        action_types.extend(["digit"] * len(generated_digit_ids))

    z_allowed_set = set(int(x) for x in z_allowed_t.tolist())
    digit_allowed_set = set(int(x) for x in digit_allowed_t.tolist())
    verify_allowed_set = set(int(x) for x in verify_allowed_t.tolist())
    _validate_actions_in_allowed(
        actions=actions,
        action_types=action_types,
        z_allowed_set=z_allowed_set,
        digit_allowed_set=digit_allowed_set,
        verify_allowed_set=verify_allowed_set,
    )

    logp_t, values_t, entropy_t = _action_stats_tensors(
        model=model,
        value_head=value_head,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attention_mask,
        actions=actions,
        action_types=action_types,
        z_allowed_t=z_allowed_t,
        digit_allowed_t=digit_allowed_t,
        verify_allowed_t=verify_allowed_t,
        temperature=temperature,
    )

    pred_digits: Optional[List[int]] = None
    digit_logits: Optional[List[List[float]]] = None
    digit_probs: Optional[List[List[float]]] = None
    if terminated_by == "answer_with_5_digits":
        id2d = {int(tok): i for i, tok in enumerate(digit_token_ids)}
        pred_digits = [int(id2d[x]) for x in generated_digit_ids]

    _t_reward0 = time.perf_counter()
    reward_info = compute_reward(
        pred_digits=pred_digits,
        true_digits=true_digits,
        terminated_reason=terminated_by,
        partial_scale=reward_cfg.partial_scale,
        keep_prob=reward_cfg.keep_prob,
        length_penalty=reward_cfg.length_penalty,
        correct_length_discount=reward_cfg.correct_length_discount,
        reward_if_max_len=reward_cfg.reward_if_max_len,
        num_generated_tokens=len(generated_z_ids) + (1 if has_answer else 0) + len(generated_digit_ids),
        generator=reward_rng,
    )
    _add_reward_timing_acc(time.perf_counter() - _t_reward0)

    return Trajectory(
        prompt_id=prompt_id,
        sample_id=sample_id,
        question=question,
        prompt_ids=list(prompt_ids),
        prompt_attention_mask=list(prompt_attention_mask),
        actions=actions,
        action_types=action_types,
        logp_old=logp_t.float().cpu().tolist(),
        values_old=values_t.float().cpu().tolist(),
        entropy_old=entropy_t.float().cpu().tolist(),
        terminated_by=terminated_by,
        generated_z_ids=generated_z_ids,
        generated_digit_ids=generated_digit_ids,
        digit_logits=digit_logits,
        digit_probs=digit_probs,
        digit_pred=pred_digits,
        digit_true=[int(x) for x in true_digits],
        reward_info=reward_info,
        num_generated_total=len(generated_z_ids) + (1 if has_answer else 0) + len(generated_digit_ids),
        num_digits_generated=len(generated_digit_ids),
    )


def _collect_rollouts_vllm_batch(
    *,
    model,
    value_head: ValueHead,
    tokenizer,
    vllm_engine: Any,
    prepared: Sequence[Dict[str, object]],
    rollouts_per_prompt: int,
    cfg: Config,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    reward_rng: torch.Generator,
    logger,
) -> List[Trajectory]:
    if len(prepared) == 0:
        return []
    num_samples_per_prompt = max(1, int(rollouts_per_prompt))
    supports_token_prompts = vllm_engine.supports_prompt_token_ids()
    prompt_texts = [str(x["prompt_text"]) for x in prepared]
    prompt_ids_batch = [list(map(int, x["prompt_ids"])) for x in prepared]
    z_gen_rows = vllm_engine.generate_z(
        prompts=prompt_texts,
        prompt_token_ids=prompt_ids_batch if supports_token_prompts else None,
        num_samples_per_prompt=num_samples_per_prompt,
        max_new_tokens=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
    )
    expected_rows = len(prepared) * int(num_samples_per_prompt)
    if len(z_gen_rows) != expected_rows:
        raise RuntimeError(
            f"vLLM generate_z returned {len(z_gen_rows)} rows, expected {expected_rows} "
            f"(batch={len(prepared)} n={num_samples_per_prompt})"
        )

    with_answer_idx: List[int] = []
    z_prefix_by_idx: Dict[int, List[int]] = {}
    digit_prompt_ids_batch: List[List[int]] = []
    digit_prompt_texts: List[str] = []

    logged_example = False
    for i, row in enumerate(z_gen_rows):
        base_idx = i // int(num_samples_per_prompt)
        base_item = prepared[base_idx]
        seq_raw = [int(x) for x in list(row.get("token_ids", []))][: int(cfg.rollout.max_new_tokens)]
        stop_reason = row.get("stop_reason")
        finish_reason = row.get("finish_reason")

        answer_in_tokens = int(answer_token_id) in seq_raw
        if answer_in_tokens:
            pos = seq_raw.index(int(answer_token_id))
            seq = seq_raw[: pos + 1]
            assert seq[-1] == int(answer_token_id), "Z-phase sequence must truncate at first <ANSWER>"
            z_prefix = seq[:pos]
            has_answer = True
        else:
            has_answer = False
            if stop_reason is not None:
                try:
                    has_answer = int(stop_reason) == int(answer_token_id)
                except Exception:
                    has_answer = False
            if has_answer:
                pos = len(seq_raw)
                z_prefix = seq_raw
            else:
                z_prefix = seq_raw


        if has_answer:
            z_prefix_by_idx[i] = z_prefix
            with_answer_idx.append(i)
            digit_prompt_ids = list(map(int, base_item["prompt_ids"])) + z_prefix + [int(answer_token_id)]
            assert digit_prompt_ids[-1] == int(answer_token_id)
            prepared_prompt_ids = list(map(int, base_item["prompt_ids"]))
            if int(answer_token_id) not in prepared_prompt_ids:
                assert int(answer_token_id) not in digit_prompt_ids[:-1]
            digit_prompt_ids_batch.append(digit_prompt_ids)
            if not supports_token_prompts:
                digit_prompt_texts.append(
                    tokenizer.decode(
                        digit_prompt_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
        else:
            z_prefix_by_idx[i] = z_prefix

    digit_map: Dict[int, List[int]] = {}
    if with_answer_idx:
        digit_allowed_set = set(int(x) for x in digit_token_ids)
        digit_gens = vllm_engine.generate_digits(
            prompts=digit_prompt_texts if not supports_token_prompts else None,
            prompt_token_ids=digit_prompt_ids_batch if supports_token_prompts else None,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            greedy=bool(cfg.rollout.digit_greedy),
        )
        for j, idx in enumerate(with_answer_idx):
            digits = [int(x) for x in digit_gens[j]]
            if len(digits) != 5:
                raise RuntimeError(f"Digit rollout must return exactly 5 tokens, got {len(digits)}")
            bad = [d for d in digits if int(d) not in digit_allowed_set]
            if bad:
                raise RuntimeError(f"Digit rollout contains tokens outside digit set: {bad}")
            digit_map[idx] = digits

    trajectories: List[Trajectory] = []
    for i in range(len(z_gen_rows)):
        base_idx = i // int(num_samples_per_prompt)
        rollout_idx = i % int(num_samples_per_prompt)
        item = prepared[base_idx]
        prompt_ids = list(item["prompt_ids"])
        prompt_attn = list(item["prompt_attention_mask"])
        z_prefix = z_prefix_by_idx[i]

        has_answer = i in with_answer_idx
        digits = digit_map.get(i, [])

        terminated_by = "max_new_tokens"
        if has_answer:
            if len(digits) == 5:
                terminated_by = "answer_with_5_digits"
            else:
                terminated_by = "max_new_tokens_during_digits"

        traj = _build_trajectory_from_vllm_tokens(
            model=model,
            value_head=value_head,
            tokenizer=tokenizer,
            prompt_id=int(item["prompt_id"]),
            question=str(item["question"]),
            true_digits=list(item["true_digits"]),
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attn,
            z_prefix_ids=z_prefix,
            has_answer=has_answer,
            digit_ids=digits,
            answer_token_id=int(answer_token_id),
            digit_token_ids=digit_token_ids,
            action_scope=cfg.rollout.action_scope,
            reward_cfg=cfg.reward,
            reward_rng=reward_rng,
            sample_id=f"{str(item['sample_id_base'])}_r{rollout_idx}",
            z_allowed_t=z_allowed_t,
            digit_allowed_t=digit_allowed_t,
            verify_allowed_t=verify_allowed_t,
            temperature=cfg.rollout.temperature,
            terminated_by=terminated_by,
        )
        trajectories.append(traj)

    return trajectories


def _extract_z_phase_from_vllm_row_with_budget(
    *,
    row: Dict[str, object],
    answer_token_id: int,
    budget: int,
) -> Tuple[List[int], bool]:
    if int(budget) <= 0:
        return [], False
    token_ids_full = [int(x) for x in list(row.get("token_ids", []))]
    token_ids = token_ids_full[: int(budget)]
    was_truncated_by_budget = len(token_ids_full) > int(budget)
    if int(answer_token_id) in token_ids:
        pos = token_ids.index(int(answer_token_id))
        return token_ids[:pos], True

    has_answer = False
    if (not was_truncated_by_budget) and row.get("stop_reason") is not None:
        try:
            has_answer = int(row.get("stop_reason")) == int(answer_token_id)
        except Exception:
            has_answer = False
    return token_ids, bool(has_answer)


def _collect_rollouts_vllm_batch_multiround(
    *,
    model,
    value_head: ValueHead,
    tokenizer,
    vllm_engine: Any,
    prepared: Sequence[Dict[str, object]],
    rollouts_per_prompt: int,
    cfg: Config,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    answer_token_id: int,
    finalize_token_id: int,
    retry_token_id: int,
    digit_token_ids: Sequence[int],
    reward_rng: torch.Generator,
    logger,
) -> List[Trajectory]:
    del logger
    if len(prepared) == 0:
        return []
    if str(cfg.rollout.action_scope) != "ppo_only_z_tokens_and_verify":
        raise RuntimeError("_collect_rollouts_vllm_batch_multiround requires action_scope=ppo_only_z_tokens_and_verify")

    num_samples_per_prompt = max(1, int(rollouts_per_prompt))
    supports_token_prompts = bool(vllm_engine.supports_prompt_token_ids())
    total_sequences = len(prepared) * int(num_samples_per_prompt)
    max_tokens_global = int(cfg.rollout.max_new_tokens)
    if max_tokens_global <= 0:
        raise RuntimeError("rollout.max_new_tokens must be > 0")
    digit_allowed_set = set(int(x) for x in digit_token_ids)
    id2d = {int(tok): i for i, tok in enumerate(digit_token_ids)}
    verify_logit_bias: Optional[Dict[int, float]] = None
    finalize_bias = float(cfg.rollout.verify_finalize_logit_bias)
    retry_bias = float(cfg.rollout.verify_retry_logit_bias)
    if finalize_bias != 0.0 or retry_bias != 0.0:
        verify_logit_bias = {
            int(finalize_token_id): float(finalize_bias),
            int(retry_token_id): float(retry_bias),
        }

    prompt_ids_by_seq: List[List[int]] = []
    prompt_attn_by_seq: List[List[int]] = []
    prompt_id_by_seq: List[int] = []
    question_by_seq: List[str] = []
    true_digits_by_seq: List[List[int]] = []
    sample_id_by_seq: List[str] = []

    for base_idx, item in enumerate(prepared):
        for rollout_idx in range(num_samples_per_prompt):
            prompt_ids_by_seq.append(list(map(int, item["prompt_ids"])))
            prompt_attn_by_seq.append(list(map(int, item["prompt_attention_mask"])))
            prompt_id_by_seq.append(int(item["prompt_id"]))
            question_by_seq.append(str(item["question"]))
            true_digits_by_seq.append([int(x) for x in list(item["true_digits"])])
            sample_id_by_seq.append(f"{str(item['sample_id_base'])}_r{rollout_idx}")

    current_prompts: List[List[int]] = [list(x) for x in prompt_ids_by_seq]
    terminated_by: List[Optional[str]] = [None for _ in range(total_sequences)]
    generated_all_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
    generated_z_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
    generated_digit_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
    generated_verify_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
    actions_by_seq: List[List[int]] = [[] for _ in range(total_sequences)]
    action_types_by_seq: List[List[str]] = [[] for _ in range(total_sequences)]
    rounds_meta_by_seq: List[List[Dict[str, object]]] = [[] for _ in range(total_sequences)]

    while True:
        active = [i for i in range(total_sequences) if terminated_by[i] is None]
        if not active:
            break

        start_active: List[int] = []
        remaining_before_round: Dict[int, int] = {}
        for idx in active:
            rem = int(max_tokens_global - len(generated_all_by_seq[idx]))
            if rem <= 0:
                terminated_by[idx] = "max_new_tokens"
                continue
            remaining_before_round[idx] = rem
            rounds_meta_by_seq[idx].append(
                {
                    "round_index": int(len(rounds_meta_by_seq[idx]) + 1),
                    "action_start": int(len(actions_by_seq[idx])),
                    "action_end": None,
                    "z_token_ids": [],
                    "digit_token_ids": [],
                    "pred_digits": None,
                    "verify_token_id": None,
                    "completed_answer": False,
                }
            )
            start_active.append(idx)

        if not start_active:
            continue

        max_z_budget = max(remaining_before_round[idx] for idx in start_active)
        z_prompt_ids = [current_prompts[idx] for idx in start_active]
        if supports_token_prompts:
            z_rows = vllm_engine.generate_z(
                prompt_token_ids=z_prompt_ids,
                num_samples_per_prompt=1,
                max_new_tokens=max_z_budget,
                temperature=cfg.rollout.temperature,
                top_p=cfg.rollout.top_p,
            )
        else:
            z_texts = [
                tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for p in z_prompt_ids
            ]
            z_rows = vllm_engine.generate_z(
                prompts=z_texts,
                num_samples_per_prompt=1,
                max_new_tokens=max_z_budget,
                temperature=cfg.rollout.temperature,
                top_p=cfg.rollout.top_p,
            )
        if len(z_rows) != len(start_active):
            raise RuntimeError("vLLM Z-phase row count mismatch in multi-round rollout")

        need_digits: List[int] = []
        for j, idx in enumerate(start_active):
            row = z_rows[j]
            rem = int(remaining_before_round[idx])
            z_prefix, has_answer = _extract_z_phase_from_vllm_row_with_budget(
                row=row,
                answer_token_id=int(answer_token_id),
                budget=rem,
            )
            round_meta = rounds_meta_by_seq[idx][-1]
            round_meta["z_token_ids"] = [int(x) for x in z_prefix]
            if z_prefix:
                actions_by_seq[idx].extend([int(x) for x in z_prefix])
                action_types_by_seq[idx].extend(["z"] * len(z_prefix))
                generated_z_by_seq[idx].extend([int(x) for x in z_prefix])
                generated_all_by_seq[idx].extend([int(x) for x in z_prefix])
                current_prompts[idx].extend([int(x) for x in z_prefix])

            if not has_answer:
                terminated_by[idx] = "max_new_tokens"
                continue

            if len(generated_all_by_seq[idx]) >= max_tokens_global:
                terminated_by[idx] = "max_new_tokens"
                continue

            actions_by_seq[idx].append(int(answer_token_id))
            action_types_by_seq[idx].append("answer")
            generated_all_by_seq[idx].append(int(answer_token_id))
            current_prompts[idx].append(int(answer_token_id))
            round_meta["has_answer"] = True
            need_digits.append(idx)

        if need_digits:
            need_verify: List[int] = []
            digits_group: Dict[int, List[int]] = defaultdict(list)
            for idx in need_digits:
                rem = int(max_tokens_global - len(generated_all_by_seq[idx]))
                if rem <= 0:
                    terminated_by[idx] = "max_new_tokens"
                    continue
                k = min(5, rem)
                digits_group[int(k)].append(idx)

            for k, idxs in digits_group.items():
                prompt_ids_batch = [current_prompts[idx] for idx in idxs]
                if supports_token_prompts:
                    digit_rows = vllm_engine.generate_digits(
                        prompt_token_ids=prompt_ids_batch,
                        num_digits=int(k),
                        temperature=cfg.rollout.temperature,
                        top_p=cfg.rollout.top_p,
                        greedy=bool(cfg.rollout.digit_greedy),
                    )
                else:
                    digit_texts = [
                        tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for p in prompt_ids_batch
                    ]
                    digit_rows = vllm_engine.generate_digits(
                        prompts=digit_texts,
                        num_digits=int(k),
                        temperature=cfg.rollout.temperature,
                        top_p=cfg.rollout.top_p,
                        greedy=bool(cfg.rollout.digit_greedy),
                    )
                if len(digit_rows) != len(idxs):
                    raise RuntimeError("vLLM digit-phase row count mismatch in multi-round rollout")

                for row_i, idx in enumerate(idxs):
                    digits = [int(x) for x in list(digit_rows[row_i])]
                    if len(digits) != int(k):
                        raise RuntimeError(f"Digit phase must return exactly {k} tokens, got {len(digits)}")
                    bad = [d for d in digits if d not in digit_allowed_set]
                    if bad:
                        raise RuntimeError(f"Digit rollout contains tokens outside digit set: {bad}")
                    round_meta = rounds_meta_by_seq[idx][-1]
                    round_meta["digit_token_ids"] = list(digits)
                    generated_digit_by_seq[idx].extend(list(digits))
                    generated_all_by_seq[idx].extend(list(digits))
                    current_prompts[idx].extend(list(digits))
                    if int(k) < 5:
                        terminated_by[idx] = "max_new_tokens"
                        continue
                    pred_digits = [int(id2d[x]) for x in digits]
                    round_meta["pred_digits"] = list(pred_digits)
                    round_meta["completed_answer"] = True
                    need_verify.append(idx)

            if need_verify:
                verify_prompt_ids = []
                verify_owner: List[int] = []
                for idx in need_verify:
                    rem = int(max_tokens_global - len(generated_all_by_seq[idx]))
                    if rem <= 0:
                        terminated_by[idx] = "max_new_tokens"
                        continue
                    verify_prompt_ids.append(current_prompts[idx])
                    verify_owner.append(idx)

                if verify_owner:
                    if supports_token_prompts:
                        verify_rows = vllm_engine.generate_verify(
                            prompt_token_ids=verify_prompt_ids,
                            temperature=cfg.rollout.temperature,
                            top_p=cfg.rollout.top_p,
                            greedy=True,
                            logit_bias=verify_logit_bias,
                        )
                    else:
                        verify_texts = [
                            tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            for p in verify_prompt_ids
                        ]
                        verify_rows = vllm_engine.generate_verify(
                            prompts=verify_texts,
                            temperature=cfg.rollout.temperature,
                            top_p=cfg.rollout.top_p,
                            greedy=True,
                            logit_bias=verify_logit_bias,
                        )
                    if len(verify_rows) != len(verify_owner):
                        raise RuntimeError("vLLM verify-phase row count mismatch in multi-round rollout")
                    for row_i, idx in enumerate(verify_owner):
                        row = [int(x) for x in list(verify_rows[row_i])]
                        if len(row) != 1:
                            raise RuntimeError(f"Verify phase must return exactly 1 token, got {len(row)}")
                        tok = int(row[0])
                        if tok not in (int(finalize_token_id), int(retry_token_id)):
                            raise RuntimeError("Verify phase emitted token outside {<FINALIZE>, <RETRY>}")
                        round_meta = rounds_meta_by_seq[idx][-1]
                        round_meta["verify_token_id"] = int(tok)
                        generated_verify_by_seq[idx].append(int(tok))
                        generated_all_by_seq[idx].append(int(tok))
                        current_prompts[idx].append(int(tok))
                        actions_by_seq[idx].append(int(tok))
                        action_types_by_seq[idx].append("verify")
                        if tok == int(finalize_token_id):
                            terminated_by[idx] = "finalize"
                        elif tok == int(retry_token_id):
                            pass
                        else:
                            raise RuntimeError("Invalid verify token emitted")

        for idx in start_active:
            rounds_meta_by_seq[idx][-1]["action_end"] = int(len(actions_by_seq[idx]))
            if terminated_by[idx] is None and len(generated_all_by_seq[idx]) >= max_tokens_global:
                terminated_by[idx] = "max_new_tokens"

    trajectories: List[Trajectory] = []
    z_allowed_set = set(int(x) for x in z_allowed_t.tolist())
    digit_allowed_set_t = set(int(x) for x in digit_allowed_t.tolist())
    verify_allowed_set = set(int(x) for x in verify_allowed_t.tolist())
    for idx in range(total_sequences):
        term = str(terminated_by[idx] or "max_new_tokens")
        rounds_meta = rounds_meta_by_seq[idx]
        round_pred_digits: List[Optional[List[int]]] = []
        for rnd in rounds_meta:
            pred = rnd.get("pred_digits")
            if isinstance(pred, list) and len(pred) == 5:
                round_pred_digits.append([int(x) for x in pred])
            else:
                round_pred_digits.append(None)

        _validate_actions_in_allowed(
            actions=actions_by_seq[idx],
            action_types=action_types_by_seq[idx],
            z_allowed_set=z_allowed_set,
            digit_allowed_set=digit_allowed_set_t,
            verify_allowed_set=verify_allowed_set,
        )
        logp_t, values_t, entropy_t = _action_stats_tensors(
            model=model,
            value_head=value_head,
            prompt_ids=prompt_ids_by_seq[idx],
            prompt_attention_mask=prompt_attn_by_seq[idx],
            actions=actions_by_seq[idx],
            action_types=action_types_by_seq[idx],
            z_allowed_t=z_allowed_t,
            digit_allowed_t=digit_allowed_t,
            verify_allowed_t=verify_allowed_t,
            temperature=cfg.rollout.temperature,
        )

        _t_reward0 = time.perf_counter()
        reward_info = compute_multi_round_reward(
            round_pred_digits=round_pred_digits,
            true_digits=true_digits_by_seq[idx],
            terminated_reason=term,
            partial_scale=cfg.reward.partial_scale,
            keep_prob=cfg.reward.keep_prob,
            length_penalty=cfg.reward.length_penalty,
            correct_length_discount=cfg.reward.correct_length_discount,
            early_success=cfg.reward.early_success,
            reward_if_max_len=cfg.reward.reward_if_max_len,
            rounds_penalty_coef=cfg.reward.rounds_penalty_coef,
            num_generated_tokens=len(generated_all_by_seq[idx]),
            round_count=len(rounds_meta),
            generator=reward_rng,
        )
        reward_info["verify_tokens_per_round"] = [rnd.get("verify_token_id", None) for rnd in rounds_meta]
        reward_info["termination_reason"] = str(term)
        _add_reward_timing_acc(time.perf_counter() - _t_reward0)

        best_idx = int(reward_info.get("best_round_index", -1))
        digit_pred: Optional[List[int]] = None
        if best_idx >= 0 and best_idx < len(round_pred_digits):
            best_pred = round_pred_digits[best_idx]
            if best_pred is not None:
                digit_pred = [int(x) for x in best_pred]

        trajectories.append(
            Trajectory(
                prompt_id=prompt_id_by_seq[idx],
                sample_id=sample_id_by_seq[idx],
                question=question_by_seq[idx],
                prompt_ids=prompt_ids_by_seq[idx],
                prompt_attention_mask=prompt_attn_by_seq[idx],
                actions=actions_by_seq[idx],
                action_types=action_types_by_seq[idx],
                logp_old=logp_t.float().cpu().tolist(),
                values_old=values_t.float().cpu().tolist(),
                entropy_old=entropy_t.float().cpu().tolist(),
                terminated_by=str(term),
                generated_z_ids=generated_z_by_seq[idx],
                generated_digit_ids=generated_digit_by_seq[idx],
                digit_logits=None,
                digit_probs=None,
                digit_pred=digit_pred,
                digit_true=[int(x) for x in true_digits_by_seq[idx]],
                reward_info=reward_info,
                num_generated_total=len(generated_all_by_seq[idx]),
                num_digits_generated=len(generated_digit_by_seq[idx]),
                generated_verify_ids=generated_verify_by_seq[idx],
                rounds_meta=rounds_meta,
                full_generated_ids=generated_all_by_seq[idx],
                termination_reason=str(term),
            )
        )

    return trajectories


def _group_trajectories_by_prompt_id(trajectories: Sequence[Trajectory]) -> Dict[int, List[Trajectory]]:
    groups: Dict[int, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if not hasattr(t, "prompt_id") or t.prompt_id is None:
            raise AssertionError("Trajectory is missing prompt_id")
        groups[int(t.prompt_id)].append(t)
    for prompt_id, group in groups.items():
        if len(group) < 1:
            raise AssertionError(f"prompt_id={prompt_id} has no trajectories")
    return groups


def _normalize_advantages_global(trajectories: Sequence[Trajectory]) -> None:
    flat: List[float] = []
    for t in trajectories:
        flat.extend(t.advantages)
    if not flat:
        return

    x = torch.tensor(flat, dtype=torch.float32)
    mean = float(x.mean().item())
    std = float(x.std(unbiased=False).item())
    denom = max(std, 1e-8)

    for t in trajectories:
        t.advantages_norm_global = [(a - mean) / denom for a in t.advantages]
        if len(t.advantages_norm_global) != len(t.advantages):
            raise AssertionError("advantages_norm_global length must match advantages length")


def _normalize_advantages_per_prompt(trajectories: Sequence[Trajectory]) -> None:
    groups = _group_trajectories_by_prompt_id(trajectories)
    for group in groups.values():
        flat: List[float] = []
        for t in group:
            flat.extend(t.advantages)
        if not flat:
            continue

        x = torch.tensor(flat, dtype=torch.float32)
        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item())
        denom = max(std, 1e-8)

        for t in group:
            t.advantages_norm_prompt = [(a - mean) / denom for a in t.advantages]
            if len(t.advantages_norm_prompt) != len(t.advantages):
                raise AssertionError("advantages_norm_prompt length must match advantages length")


def _combine_advantages_hybrid(
    trajectories: Sequence[Trajectory],
    alpha: float,
    use_global_for_homogeneous_prompts: bool,
) -> None:
    groups = _group_trajectories_by_prompt_id(trajectories)
    alpha_f = float(alpha)
    for group in groups.values():
        rewards = [float(t.reward_info["reward_final"]) for t in group]
        homogeneous = len(set(rewards)) <= 1
        for t in group:
            if len(t.advantages_norm_global) != len(t.advantages):
                raise AssertionError("Hybrid mode requires advantages_norm_global before combine")
            if len(t.advantages_norm_prompt) != len(t.advantages):
                raise AssertionError("Hybrid mode requires advantages_norm_prompt before combine")

            if homogeneous and bool(use_global_for_homogeneous_prompts):
                t.advantages_norm = list(t.advantages_norm_global)
            else:
                t.advantages_norm = [
                    alpha_f * ag + (1.0 - alpha_f) * ap
                    for ag, ap in zip(t.advantages_norm_global, t.advantages_norm_prompt)
                ]
            if len(t.advantages_norm) != len(t.advantages):
                raise AssertionError("advantages_norm length must match advantages length")


def _prepare_normalized_advantages(trajectories: Sequence[Trajectory], cfg: Config) -> None:
    mode = str(cfg.ppo.adv_norm_mode).strip().lower()
    normalize_enabled = bool(cfg.ppo.normalize_advantages)
    if not normalize_enabled:
        mode = "none"

    for t in trajectories:
        t.advantages_norm_global = []
        t.advantages_norm_prompt = []
        t.advantages_norm = []

    if mode == "none":
        for t in trajectories:
            t.advantages_norm = list(t.advantages)
    elif mode == "global":
        _normalize_advantages_global(trajectories)
        for t in trajectories:
            t.advantages_norm = list(t.advantages_norm_global)
    elif mode == "per_prompt":
        _normalize_advantages_per_prompt(trajectories)
        for t in trajectories:
            t.advantages_norm = list(t.advantages_norm_prompt)
    elif mode == "hybrid":
        _normalize_advantages_global(trajectories)
        _normalize_advantages_per_prompt(trajectories)
        _combine_advantages_hybrid(
            trajectories,
            alpha=float(cfg.ppo.adv_norm_hybrid_alpha),
            use_global_for_homogeneous_prompts=bool(cfg.ppo.adv_norm_use_global_for_homogeneous_prompts),
        )
    else:
        raise ValueError(f"Unsupported ppo.adv_norm_mode={cfg.ppo.adv_norm_mode!r}")

    for t in trajectories:
        if len(t.advantages_norm) != len(t.advantages):
            raise AssertionError("advantages_norm length must match advantages length")


def _prompt_reward_homogeneity_stats(groups: Dict[int, List[Trajectory]]) -> Tuple[float, float]:
    if not groups:
        return 0.0, 0.0
    homogeneous = 0
    mixed = 0
    for group in groups.values():
        rewards = {float(t.reward_info["reward_final"]) for t in group}
        if len(rewards) <= 1:
            homogeneous += 1
        else:
            mixed += 1
    total = max(len(groups), 1)
    return float(homogeneous) / float(total), float(mixed) / float(total)


def _traj_has_valid_ce_target(
    traj: Trajectory,
    *,
    expected_digits: int = 5,
) -> bool:
    if "answer" not in traj.action_types:
        return False
    if len(traj.digit_true) != int(expected_digits):
        return False
    return all(0 <= int(d) <= 9 for d in traj.digit_true)


def _select_ce_trajectory_indices(
    *,
    batch_trajs: Sequence[Trajectory],
    batch_frac_to_apply_ce: float,
    ce_mode: str,
) -> List[int]:
    if not batch_trajs:
        return []

    frac = float(batch_frac_to_apply_ce)
    if frac <= 0.0:
        return []
    frac = min(frac, 1.0)

    valid = [i for i, t in enumerate(batch_trajs) if _traj_has_valid_ce_target(t)]
    if not valid:
        return []

    mode = str(ce_mode).strip().lower()
    if mode == "random":
        # Random mode uses minibatch-size-based cap.
        k = int(math.ceil(frac * len(batch_trajs)))
        if k <= 0:
            return []
        k = min(k, len(valid))
        if k <= 0:
            return []
        picked = random.sample(valid, k=k)
        picked.sort()
        return picked

    if mode != "successful_traces":
        raise ValueError(f"Unsupported ppo.ce_mode={ce_mode!r}; expected 'successful_traces' or 'random'")

    # Successful traces are exact-match only; no fallback to non-exact trajectories.
    success = [i for i in valid if bool(batch_trajs[i].reward_info.get("exact_match", False))]
    if not success:
        return []
    # successful_traces mode scales by number of successful trajectories.
    k = int(math.ceil(frac * len(success)))
    if k <= 0:
        return []
    k = min(k, len(success))
    picked = random.sample(success, k=k)
    picked.sort()
    return picked


def _compute_digit_ce_for_minibatch(
    *,
    model,
    batch_trajs: Sequence[Trajectory],
    selected_indices: Sequence[int],
    digit_token_ids: Sequence[int],
    answer_token_id: int,
    pad_token_id: int,
) -> Tuple[Optional[torch.Tensor], int]:
    if not selected_indices:
        return None, 0

    device = next(model.parameters()).device
    ce_inputs: List[List[int]] = []
    ce_targets_local: List[List[int]] = []

    for idx in selected_indices:
        traj = batch_trajs[int(idx)]
        if not _traj_has_valid_ce_target(traj):
            continue

        try:
            target_token_ids = [int(digit_token_ids[int(d)]) for d in traj.digit_true]
        except Exception:
            continue

        prefix = list(traj.prompt_ids) + list(traj.generated_z_ids) + [int(answer_token_id)]
        if not prefix:
            continue
        ce_inputs.append(prefix + target_token_ids)
        ce_targets_local.append([int(d) for d in traj.digit_true])

    if not ce_inputs:
        return None, 0

    max_len = max(len(x) for x in ce_inputs)
    bsz = len(ce_inputs)

    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    prefix_lens: List[int] = []
    for i, seq in enumerate(ce_inputs):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :L] = 1
        prefix_lens.append(L - len(ce_targets_local[i]))

    base_model = model.get_submodule(model.base_model_prefix)
    out = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = out.last_hidden_state  # [B,L,H]

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model output embeddings (LM head) are unavailable")

    digit_allowed_t = torch.tensor(list(digit_token_ids), dtype=torch.long, device=device)
    d_w = lm_head.weight.index_select(0, digit_allowed_t)
    d_b = None
    if getattr(lm_head, "bias", None) is not None:
        d_b = lm_head.bias.index_select(0, digit_allowed_t)

    per_traj_losses: List[torch.Tensor] = []
    for b in range(bsz):
        prefix_len = int(prefix_lens[b])
        target_local = torch.tensor(ce_targets_local[b], dtype=torch.long, device=device)
        state_positions = torch.arange(
            prefix_len - 1,
            prefix_len - 1 + target_local.numel(),
            device=device,
            dtype=torch.long,
        )
        h = hidden[b].index_select(0, state_positions)
        logits = h @ d_w.t()
        if d_b is not None:
            logits = logits + d_b
        per_traj_losses.append(F.cross_entropy(logits, target_local, reduction="mean"))

    if not per_traj_losses:
        return None, 0

    return torch.stack(per_traj_losses).mean(), len(per_traj_losses)


def _recompute_trajectory(
    model,
    value_head: ValueHead,
    traj: Trajectory,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    verify_allowed_t: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _action_stats_tensors(
        model=model,
        value_head=value_head,
        prompt_ids=traj.prompt_ids,
        prompt_attention_mask=traj.prompt_attention_mask,
        actions=traj.actions,
        action_types=traj.action_types,
        z_allowed_t=z_allowed_t,
        digit_allowed_t=digit_allowed_t,
        verify_allowed_t=verify_allowed_t,
        temperature=temperature,
    )


def _save_checkpoint(
    *,
    output_dir: str,
    step: int,
    model,
    value_head: ValueHead,
    tokenizer,
    ppo_optimizer: torch.optim.Optimizer,
    warmup_optimizer: Optional[torch.optim.Optimizer],
    ds_index: int,
    prev_train_mode: Optional[str],
    reward_rng: torch.Generator,
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
            "checkpoint_format_version": 1,
            "step": int(step),
            "value_head_state_dict": value_head.state_dict(),
            "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
            "warmup_optimizer_state_dict": (warmup_optimizer.state_dict() if warmup_optimizer is not None else None),
            "ds_index": int(ds_index),
            "prev_train_mode": prev_train_mode,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "reward_rng_state": reward_rng.get_state(),
        },
        os.path.join(ckpt_dir, "ppo_state.pt"),
    )


def _checkpoint_step_from_dirname(path: str) -> Optional[int]:
    base = os.path.basename(os.path.normpath(path))
    m = re.fullmatch(r"step_(\d+)", base)
    if m is None:
        return None
    return int(m.group(1))


def _find_latest_checkpoint_in_dir(root_dir: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None
    candidates: List[Tuple[int, str]] = []
    for p in glob(os.path.join(root_dir, "step_*")):
        if not os.path.isdir(p):
            continue
        step = _checkpoint_step_from_dirname(p)
        if step is None:
            continue
        if not os.path.isfile(os.path.join(p, "ppo_state.pt")):
            continue
        if not os.path.isdir(os.path.join(p, "model")):
            continue
        candidates.append((int(step), str(p)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def _resolve_resume_checkpoint_dir(cfg: Config) -> Optional[str]:
    resume_from_raw = str(getattr(cfg.train, "resume_from", "")).strip()
    if resume_from_raw:
        p = os.path.abspath(os.path.expanduser(resume_from_raw))
        if not os.path.isdir(p):
            raise FileNotFoundError(f"train.resume_from is not a directory: {p}")

        # Direct step directory.
        step = _checkpoint_step_from_dirname(p)
        if step is not None:
            return p

        # output_dir or checkpoints dir.
        search_roots: List[str] = [p]
        if os.path.basename(os.path.normpath(p)) != "checkpoints":
            search_roots.append(os.path.join(p, "checkpoints"))
        for root in search_roots:
            latest = _find_latest_checkpoint_in_dir(root)
            if latest is not None:
                return latest
        raise FileNotFoundError(
            f"Could not find any checkpoint step_* directory under resume_from={p}"
        )

    if bool(getattr(cfg.train, "resume_auto_latest", False)):
        latest = _find_latest_checkpoint_in_dir(os.path.join(cfg.train.output_dir, "checkpoints"))
        if latest is not None:
            return latest
    return None


def _rotate_checkpoints(output_dir: str, keep_last: int) -> None:
    ckpts = sorted(glob(os.path.join(output_dir, "checkpoints", "step_*")))
    if len(ckpts) <= keep_last:
        return
    for old in ckpts[: len(ckpts) - keep_last]:
        shutil.rmtree(old, ignore_errors=True)


def train(cfg: Config) -> None:
    _set_seed(cfg.train.seed)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "rollouts"), exist_ok=True)
    _set_run_log_path(os.path.join(cfg.train.output_dir, "train.log"))
    _log(f"Run log file: {os.path.abspath(os.path.join(cfg.train.output_dir, 'train.log'))}")

    torch_device_cfg = str(getattr(cfg.rollout, "torch_device", "cuda:0")).strip()
    device = torch.device(torch_device_cfg if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    resume_ckpt_dir = _resolve_resume_checkpoint_dir(cfg)
    model_init_path = (
        os.path.join(resume_ckpt_dir, "model")
        if resume_ckpt_dir is not None
        else cfg.model.init_ckpt
    )
    tokenizer_init_path = (
        os.path.join(resume_ckpt_dir, "tokenizer")
        if resume_ckpt_dir is not None
        else cfg.model.init_ckpt
    )
    if resume_ckpt_dir is not None:
        _log(f"Resuming from checkpoint: {resume_ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_init_path,
        use_fast=True,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_init_path,
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
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        _log("Reference model enabled (kl_coef > 0)")
    else:
        _log("Reference model disabled (kl_coef <= 0); skipping ref model allocation")

    action_scope = validate_action_scope(cfg.rollout.action_scope)
    if action_scope == "ppo_full" and bool(cfg.rollout.digit_greedy):
        raise ValueError("digit_greedy=True is incompatible with ppo_full")
    rollout_backend = str(getattr(cfg.rollout, "backend", "vllm")).strip().lower()
    if action_scope == "ppo_only_z_tokens_and_verify":
        if rollout_backend != "vllm":
            raise ValueError("ppo_only_z_tokens_and_verify currently requires rollout.backend='vllm'")
        if not bool(cfg.rollout.vllm_enabled):
            raise ValueError("ppo_only_z_tokens_and_verify requires rollout.vllm_enabled=True")
        if not bool(cfg.rollout.digit_greedy):
            raise ValueError("ppo_only_z_tokens_and_verify requires rollout.digit_greedy=True")
    if not math.isfinite(float(cfg.rollout.verify_finalize_logit_bias)):
        raise ValueError("rollout.verify_finalize_logit_bias must be finite")
    if not math.isfinite(float(cfg.rollout.verify_retry_logit_bias)):
        raise ValueError("rollout.verify_retry_logit_bias must be finite")
    if float(cfg.reward.rounds_penalty_coef) < 0.0:
        raise ValueError("reward.rounds_penalty_coef must be >= 0")
    if float(cfg.reward.early_success) < 0.0 or float(cfg.reward.early_success) > 1.0:
        raise ValueError("reward.early_success must be in [0.0, 1.0]")
    ce_mode = str(cfg.ppo.ce_mode).strip().lower()
    if ce_mode not in ("successful_traces", "random"):
        raise ValueError(
            f"Unsupported ppo.ce_mode={cfg.ppo.ce_mode!r}; expected 'successful_traces' or 'random'"
        )
    adv_norm_mode = str(cfg.ppo.adv_norm_mode).strip().lower()
    if adv_norm_mode not in ("global", "per_prompt", "hybrid", "none"):
        raise ValueError(
            f"Unsupported ppo.adv_norm_mode={cfg.ppo.adv_norm_mode!r}; expected "
            "'global', 'per_prompt', 'hybrid', or 'none'"
        )
    if float(cfg.ppo.adv_norm_hybrid_alpha) < 0.0 or float(cfg.ppo.adv_norm_hybrid_alpha) > 1.0:
        raise ValueError("ppo.adv_norm_hybrid_alpha must be in [0, 1]")
    if float(cfg.ppo.batch_frac_to_apply_ce) < 0.0:
        raise ValueError("ppo.batch_frac_to_apply_ce must be >= 0")
    if int(cfg.rollout.rollouts_per_prompt) < 1:
        raise ValueError("rollout.rollouts_per_prompt must be >= 1")
    if int(cfg.ppo.value_warmup_steps) < 0:
        raise ValueError("ppo.value_warmup_steps must be >= 0")
    if float(cfg.ppo.value_warmup_lr) <= 0.0:
        raise ValueError("ppo.value_warmup_lr must be > 0")
    if int(getattr(cfg.runtime, "length_bucket_width", 64)) <= 0:
        raise ValueError("runtime.length_bucket_width must be > 0")

    z_token_ids, z_style = introspect_z_token_ids_and_style(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found in tokenizer (checked lowercase <z_i> then uppercase <Z_i>)")
    if z_style == "upper":
        _log("WARNING: using uppercase <Z_i> tokens fallback; lowercase <z_i> not found")

    answer_token_id = resolve_answer_token_id(tokenizer, answer_token=cfg.model.answer_token)
    validate_answer_token_single(tokenizer, cfg.model.answer_token, answer_token_id)
    finalize_token_id = _resolve_strict_vocab_token_id(tokenizer, str(cfg.model.finalize_token), label="Verify")
    retry_token_id = _resolve_strict_vocab_token_id(tokenizer, str(cfg.model.retry_token), label="Verify")
    if int(finalize_token_id) == int(retry_token_id):
        raise RuntimeError("<FINALIZE> and <RETRY> must map to distinct token ids")
    digit_token_ids = resolve_digit_token_ids(tokenizer)

    z_allowed_t = torch.tensor(list(z_token_ids) + [int(answer_token_id)], dtype=torch.long, device=device)
    digit_allowed_t = torch.tensor(list(digit_token_ids), dtype=torch.long, device=device)
    verify_allowed_t = torch.tensor([int(finalize_token_id), int(retry_token_id)], dtype=torch.long, device=device)

    _log(
        f"Action scope={action_scope} | Z tokens={len(z_token_ids)} ({z_style}) | "
        f"answer_token_id={answer_token_id} | finalize_token_id={finalize_token_id} | retry_token_id={retry_token_id} | "
        f"verify_finalize_logit_bias={float(cfg.rollout.verify_finalize_logit_bias):.4f} | "
        f"verify_retry_logit_bias={float(cfg.rollout.verify_retry_logit_bias):.4f}"
    )

    hidden_size = int(model.config.hidden_size)
    value_head = ValueHead(hidden_size=hidden_size).to(device)
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
    if _should_run_debug_restricted_logits_check(cfg):
        with torch.no_grad():
            weight = lm_head.weight
            bias = getattr(lm_head, "bias", None)
            z_w_dbg = weight.index_select(0, z_allowed_t).detach()
            d_w_dbg = weight.index_select(0, digit_allowed_t).detach()
            z_b_dbg = bias.index_select(0, z_allowed_t).detach() if bias is not None else None
            d_b_dbg = bias.index_select(0, digit_allowed_t).detach() if bias is not None else None
        _debug_restricted_logits_check_once(
            model=model,
            tokenizer=tokenizer,
            z_allowed_t=z_allowed_t,
            digit_allowed_t=digit_allowed_t,
            z_w=z_w_dbg,
            d_w=d_w_dbg,
            z_b=z_b_dbg,
            d_b=d_b_dbg,
        )

    ce_enabled = bool(cfg.ppo.apply_ce)
    if action_scope == "ppo_only_z_tokens_and_verify" and ce_enabled:
        _log("Disabling CE auxiliary for ppo_only_z_tokens_and_verify path (no round-aware CE semantics implemented)")
        ce_enabled = False

    ppo_params = list(model.parameters()) + list(value_head.parameters())
    value_head_params = list(value_head.parameters())
    ppo_optimizer = bnb.optim.AdamW8bit(
        ppo_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )
    warmup_optimizer: Optional[torch.optim.Optimizer] = None
    if int(cfg.ppo.value_warmup_steps) > 0:
        warmup_optimizer = bnb.optim.AdamW8bit(
            value_head_params,
            lr=float(cfg.ppo.value_warmup_lr),
            weight_decay=cfg.train.weight_decay,
            betas=cfg.train.betas,
            eps=cfg.train.eps,
        )
        _assert_optimizer_matches_params(warmup_optimizer, value_head_params)
    _assert_optimizer_matches_params(ppo_optimizer, ppo_params)

    resume_state: Optional[Dict[str, object]] = None
    resume_step = 0
    if resume_ckpt_dir is not None:
        ppo_state_path = os.path.join(resume_ckpt_dir, "ppo_state.pt")
        if not os.path.isfile(ppo_state_path):
            raise FileNotFoundError(f"Resume checkpoint missing ppo_state.pt: {ppo_state_path}")
        resume_state = torch.load(ppo_state_path, map_location="cpu")
        if "value_head_state_dict" not in resume_state:
            raise RuntimeError(f"Resume checkpoint missing value_head_state_dict: {ppo_state_path}")
        value_head.load_state_dict(resume_state["value_head_state_dict"])
        if "ppo_optimizer_state_dict" in resume_state and resume_state["ppo_optimizer_state_dict"] is not None:
            ppo_optimizer.load_state_dict(resume_state["ppo_optimizer_state_dict"])
        if warmup_optimizer is not None and resume_state.get("warmup_optimizer_state_dict") is not None:
            warmup_optimizer.load_state_dict(resume_state["warmup_optimizer_state_dict"])
        if "step" in resume_state:
            resume_step = int(resume_state["step"])
        else:
            parsed = _checkpoint_step_from_dirname(resume_ckpt_dir)
            resume_step = int(parsed) if parsed is not None else 0
        _log(f"Loaded optimizer/value state from checkpoint step={resume_step}")

    rollout_backend = str(getattr(cfg.rollout, "backend", "vllm")).strip().lower()
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
    elif cfg.rollout.vllm_enabled:
        vllm_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
        vllm_kwargs.setdefault("tensor_parallel_size", int(cfg.rollout.vllm_tp_size))
        vllm_kwargs.setdefault("gpu_memory_utilization", float(cfg.rollout.gpu_memory_utilization))
        vllm_kwargs.setdefault("weight_transfer_device", str(device))
        if int(cfg.rollout.vllm_tp_size) == 1:
            vllm_cvd = str(getattr(cfg.rollout, "vllm_cuda_visible_devices", "")).strip()
            if vllm_cvd:
                vllm_kwargs.setdefault("cuda_visible_devices", vllm_cvd)
                _log(f"vLLM CUDA_VISIBLE_DEVICES={vllm_kwargs['cuda_visible_devices']}")
        vllm_seed = int(cfg.rollout.vllm_seed) if cfg.rollout.vllm_seed is not None else int(cfg.train.seed)
        vllm_engine = VLLMRolloutEngine(
            init_ckpt=model_init_path,
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
        _log(
            f"Loaded {len(excluded_questions)} unique questions to exclude from "
            f"{excluded_questions_path}"
        )

    train_row_indices: List[int] = []
    removed_count = 0
    q_field = str(cfg.data.question_field)
    for row_idx in range(len(ds)):
        sample = ds[int(row_idx)]
        q_text = _question_text(sample.get(q_field, ""))
        if q_text in excluded_questions:
            removed_count += 1
            continue
        train_row_indices.append(int(row_idx))

    if excluded_questions_path:
        _log(
            f"Dataset filter by question text: before={len(ds)} after={len(train_row_indices)} "
            f"removed={removed_count}"
        )

    if len(train_row_indices) == 0:
        raise RuntimeError("No training rows remain after applying rsft_trained_questions filter")

    reward_rng = _make_rng(cfg.train.seed + 17)
    rollout_logger = RolloutLogger(os.path.join(cfg.train.output_dir, "rollouts"))

    ds_index = 0
    prev_train_mode: Optional[str] = None
    start_update = 1
    if resume_state is not None:
        ds_index = int(resume_state.get("ds_index", 0))
        prev_train_mode_raw = resume_state.get("prev_train_mode", None)
        prev_train_mode = str(prev_train_mode_raw) if prev_train_mode_raw is not None else None
        if "reward_rng_state" in resume_state and resume_state["reward_rng_state"] is not None:
            reward_rng.set_state(resume_state["reward_rng_state"])
        if "python_random_state" in resume_state and resume_state["python_random_state"] is not None:
            random.setstate(resume_state["python_random_state"])
        if "torch_rng_state" in resume_state and resume_state["torch_rng_state"] is not None:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available():
            cuda_state = resume_state.get("torch_cuda_rng_state_all")
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        start_update = int(resume_step) + 1
        _log(
            f"Resume state restored: next_update={start_update}, ds_index={ds_index}, "
            f"prev_train_mode={prev_train_mode}"
        )
    if start_update > int(cfg.train.updates):
        _log(
            f"Resume checkpoint is already at/after target updates "
            f"(next_update={start_update}, train.updates={int(cfg.train.updates)}). Nothing to do."
        )
        if vllm_engine is not None:
            vllm_engine.close()
        return

    try:
        for update in range(start_update, cfg.train.updates + 1):
            _t_update0 = time.perf_counter()
            _reset_reward_timing_acc()
            value_warmup_active = (update - 1) < int(cfg.ppo.value_warmup_steps)
            if value_warmup_active:
                _set_value_warmup_trainability(model=model, value_head=value_head, enabled=True)
                _assert_value_warmup_trainability(model=model, value_head=value_head, enabled=True)
                if warmup_optimizer is None:
                    raise AssertionError("Warmup is active but warmup optimizer is not initialized")
                active_optimizer = warmup_optimizer
                active_clip_params = value_head_params
                train_mode = "value_warmup"
            else:
                _set_value_warmup_trainability(model=model, value_head=value_head, enabled=False)
                _assert_value_warmup_trainability(model=model, value_head=value_head, enabled=False)
                active_optimizer = ppo_optimizer
                active_clip_params = ppo_params
                train_mode = "ppo"

            _assert_optimizer_matches_params(active_optimizer, active_clip_params)
            if prev_train_mode != train_mode:
                if train_mode == "ppo" and prev_train_mode == "value_warmup":
                    _log(f"Warmup ended at update={update}; switching to normal PPO training")
                else:
                    _log(f"Training mode switch at update={update}: mode={train_mode}")
                prev_train_mode = train_mode

            t_sync_sec = 0.0
            if vllm_engine is not None:
                _t_sync0 = time.perf_counter()
                synced = vllm_engine.maybe_sync_from_torch(model=model, tokenizer=tokenizer, update_idx=update)
                t_sync_sec += time.perf_counter() - _t_sync0
                # if synced:
                #     _log(f"vLLM policy sync complete at update={update}")

            trajectories: List[Trajectory] = []
            token_budget = 0
            full_generated_token_budget = 0
            prompt_counter = 0
            _t_rollout0 = time.perf_counter()

            while len(trajectories) < cfg.rollout.episodes_per_batch:
                remaining = cfg.rollout.episodes_per_batch - len(trajectories)
                rollouts_per_prompt = max(int(cfg.rollout.rollouts_per_prompt), 1)
                prompts_needed = max(1, int(math.ceil(float(remaining) / float(rollouts_per_prompt))))
                this_batch = min(int(cfg.rollout.vllm_batch_size), prompts_needed)

                prepared: List[Dict[str, object]] = []
                while len(prepared) < this_batch:
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
                    prompt_id = int(prompt_counter)
                    prompt_counter += 1
                    prepared.append(
                        {
                            "sample_id_base": f"u{update}_p{prompt_id}",
                            "prompt_id": prompt_id,
                            "question": question,
                            "true_digits": true_digits,
                            "prompt_text": prompt_text,
                            "prompt_ids": prompt_ids,
                            "prompt_attention_mask": prompt_attn,
                        }
                    )

                if not prepared:
                    continue

                if vllm_engine is not None:
                    if action_scope == "ppo_only_z_tokens_and_verify":
                        batch_trajs = _collect_rollouts_vllm_batch_multiround(
                            model=model,
                            value_head=value_head,
                            tokenizer=tokenizer,
                            vllm_engine=vllm_engine,
                            prepared=prepared,
                            rollouts_per_prompt=rollouts_per_prompt,
                            cfg=cfg,
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            verify_allowed_t=verify_allowed_t,
                            answer_token_id=int(answer_token_id),
                            finalize_token_id=int(finalize_token_id),
                            retry_token_id=int(retry_token_id),
                            digit_token_ids=digit_token_ids,
                            reward_rng=reward_rng,
                            logger=_log,
                        )
                    else:
                        batch_trajs = _collect_rollouts_vllm_batch(
                            model=model,
                            value_head=value_head,
                            tokenizer=tokenizer,
                            vllm_engine=vllm_engine,
                            prepared=prepared,
                            rollouts_per_prompt=rollouts_per_prompt,
                            cfg=cfg,
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            verify_allowed_t=verify_allowed_t,
                            answer_token_id=int(answer_token_id),
                            digit_token_ids=digit_token_ids,
                            reward_rng=reward_rng,
                            logger=_log,
                        )
                else:
                    if action_scope == "ppo_only_z_tokens_and_verify":
                        raise RuntimeError("ppo_only_z_tokens_and_verify currently requires vLLM rollout backend")
                    prepared_rollouts: List[Dict[str, object]] = []
                    for item in prepared:
                        sample_id_base = str(item["sample_id_base"])
                        for rollout_idx in range(rollouts_per_prompt):
                            expanded = dict(item)
                            expanded["sample_id"] = f"{sample_id_base}_r{rollout_idx}"
                            prepared_rollouts.append(expanded)
                    batch_trajs = [
                        _rollout_one_torch(
                            model=model,
                            value_head=value_head,
                            tokenizer=tokenizer,
                            prompt_id=int(item["prompt_id"]),
                            question=str(item["question"]),
                            true_digits=list(item["true_digits"]),
                            z_token_ids=z_token_ids,
                            digit_token_ids=digit_token_ids,
                            answer_token_id=int(answer_token_id),
                            max_new_tokens=cfg.rollout.max_new_tokens,
                            temperature=cfg.rollout.temperature,
                            top_p=cfg.rollout.top_p,
                            action_scope=action_scope,
                            digit_greedy=cfg.rollout.digit_greedy,
                            reward_cfg=cfg.reward,
                            reward_rng=reward_rng,
                            sample_id=str(item["sample_id"]),
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            verify_allowed_t=verify_allowed_t,
                        )
                        for item in prepared_rollouts
                    ]

                for traj in batch_trajs:
                    if not traj.actions:
                        continue
                    trajectories.append(traj)
                    token_budget += len(traj.actions)
                    full_generated_token_budget += int(traj.num_generated_total)
                    if token_budget >= int(cfg.rollout.max_tokens_per_batch):
                        break
                if token_budget >= int(cfg.rollout.max_tokens_per_batch):
                    break
            t_rollout_sec = time.perf_counter() - _t_rollout0

            if not trajectories:
                raise RuntimeError("No trajectories collected for PPO update")

            prompt_groups = _group_trajectories_by_prompt_id(trajectories)
            unique_prompt_ids = len(prompt_groups)
            avg_rollouts_per_prompt_actual = float(len(trajectories)) / float(max(unique_prompt_ids, 1))
            homogeneous_prompt_frac, mixed_prompt_frac = _prompt_reward_homogeneity_stats(prompt_groups)
            _prepare_normalized_advantages(trajectories=trajectories, cfg=cfg)
            adv_norm_mode_effective = (
                str(cfg.ppo.adv_norm_mode).strip().lower() if bool(cfg.ppo.normalize_advantages) else "none"
            )

            roll_rows: List[Dict[str, object]] = []
            for traj in trajectories:
                row = {
                    "schema_version": 2,
                    "id": traj.sample_id,
                    "prompt_id": int(traj.prompt_id),
                    "question": traj.question,
                    "input_ids": traj.prompt_ids,
                    "generated_z_ids": traj.generated_z_ids,
                    "generated_z_tokens": tokenizer.convert_ids_to_tokens(traj.generated_z_ids),
                    "generated_digit_ids": traj.generated_digit_ids,
                    "generated_digit_tokens": tokenizer.convert_ids_to_tokens(traj.generated_digit_ids),
                    "generated_verify_ids": traj.generated_verify_ids,
                    "generated_verify_tokens": tokenizer.convert_ids_to_tokens(traj.generated_verify_ids),
                    "terminated_by": traj.terminated_by,
                    "termination_reason": traj.termination_reason,
                    "num_generated": traj.num_generated_total,
                    "num_digits_generated": traj.num_digits_generated,
                    "digit_logits": traj.digit_logits,
                    "digit_probs": traj.digit_probs,
                    "digit_pred": traj.digit_pred,
                    "digit_true": traj.digit_true,
                    "full_generated_ids": traj.full_generated_ids,
                    "rounds_meta": traj.rounds_meta,
                    "reward_full": traj.reward_info["reward_full"],
                    "partial_scale": traj.reward_info["partial_scale"],
                    "keep_prob": traj.reward_info["keep_prob"],
                    "applied_mask": traj.reward_info["applied_mask"],
                    "applied_count": traj.reward_info["applied_count"],
                    "correct_count": traj.reward_info["correct_count"],
                    "reward_partial": traj.reward_info["reward_partial"],
                    "length_penalty": traj.reward_info["length_penalty"],
                    "reward_if_max_len": traj.reward_info["reward_if_max_len"],
                    "round_count": traj.reward_info.get("round_count", None),
                    "round_answer_rewards": traj.reward_info.get("round_answer_rewards", None),
                    "best_round_answer_reward": traj.reward_info.get("best_round_answer_reward", None),
                    "best_round_index": traj.reward_info.get("best_round_index", None),
                    "token_penalty": traj.reward_info.get("token_penalty", None),
                    "rounds_penalty": traj.reward_info.get("rounds_penalty", None),
                    "verify_tokens_per_round": traj.reward_info.get("verify_tokens_per_round", None),
                    "reward_final": traj.reward_info["reward_final"],
                    "actions": traj.actions,
                    "action_types": traj.action_types,
                    "logp_old": traj.logp_old,
                    "entropy": traj.entropy_old,
                    "values": traj.values_old,
                }
                if cfg.logging.log_action_tokens:
                    row["action_tokens"] = tokenizer.convert_ids_to_tokens(traj.actions)
                roll_rows.append(row)

            rollout_path = rollout_logger.write_step(step=update, rows=roll_rows)

            _t_backprop0 = time.perf_counter()
            active_optimizer.zero_grad(set_to_none=True)
            minibatch_count = 0

            pol_acc = 0.0
            val_acc = 0.0
            ent_acc = 0.0
            ent_loss_acc = 0.0
            clip_acc = 0.0
            kl_acc = 0.0
            kl_pen_acc = 0.0
            ce_acc = 0.0
            ce_examples_acc = 0

            trajectory_cache = _build_trajectory_device_cache(trajectories=trajectories, device=device)
            seq_lens = [c.seq_len for c in trajectory_cache]
            token_stats_kernel = _get_token_stats_kernel(
                compile_update_stats=bool(getattr(cfg.runtime, "compile_update_stats", False))
            )
            ce_selected_global: set[int] = set()
            if (not value_warmup_active) and bool(ce_enabled):
                ce_selected_global = set(
                    _select_ce_trajectory_indices(
                        batch_trajs=trajectories,
                        batch_frac_to_apply_ce=float(cfg.ppo.batch_frac_to_apply_ce),
                        ce_mode=ce_mode,
                    )
                )

            for _epoch in range(cfg.ppo.ppo_epochs):
                order = _build_minibatch_order(
                    seq_lens=seq_lens,
                    use_length_bucketing=bool(getattr(cfg.runtime, "use_length_bucketing", True)),
                    bucket_width=int(getattr(cfg.runtime, "length_bucket_width", 64)),
                )
                for start in range(0, len(order), cfg.ppo.minibatch_size):
                    batch_idx = order[start : start + cfg.ppo.minibatch_size]
                    batch_trajs = [trajectories[idx] for idx in batch_idx]
                    batch_cache = [trajectory_cache[idx] for idx in batch_idx]

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and cfg.runtime.use_bf16
                        else nullcontext()
                    )
                    with amp_ctx:
                        (
                            logp_new,
                            logp_ref,
                            values_new,
                            entropy_new,
                            logp_old,
                            advantages,
                            returns,
                            lengths,
                        ) = _action_stats_tensors_batched(
                            model=model,
                            ref_model=ref_model,
                            value_head=value_head,
                            trajs=batch_trajs,
                            traj_cache=batch_cache,
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            verify_allowed_t=verify_allowed_t,
                            z_id_to_local=z_id_to_local,
                            d_id_to_local=d_id_to_local,
                            v_id_to_local=v_id_to_local,
                            temperature=cfg.rollout.temperature,
                            pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0,
                            token_stats_kernel=token_stats_kernel,
                        )

                        total_tokens = int(lengths.sum().item())
                        if int(logp_new.numel()) != total_tokens:
                            raise RuntimeError(
                                f"Token count mismatch: T={int(logp_new.numel())}, sum(lengths)={total_tokens}"
                            )
                        if int(lengths.numel()) == 0:
                            continue

                        logp_new_f = logp_new.float()
                        logp_old_f = logp_old.float()
                        logp_ref_f = logp_ref.float()
                        advantages_f = advantages.float()
                        values_new_f = values_new.float()
                        returns_f = returns.float()
                        entropy_new_f = entropy_new.float()

                        log_ratio = logp_new_f - logp_old_f
                        ratio = torch.exp(log_ratio)
                        ratio_clipped = torch.clamp(ratio, 1.0 - cfg.ppo.clip_range, 1.0 + cfg.ppo.clip_range)
                        pg1 = ratio * advantages_f
                        pg2 = ratio_clipped * advantages_f
                        policy_loss_tok = -torch.min(pg1, pg2)
                        kl_tok = logp_new_f - logp_ref_f
                        lo = 1.0 - cfg.ppo.clip_range
                        hi = 1.0 + cfg.ppo.clip_range
                        clipped_tok = ((ratio < lo) | (ratio > hi)).float()
                        value_loss_tok = (values_new_f - returns_f).pow(2)

                        policy_loss = _segment_means(policy_loss_tok, lengths).mean()
                        clipfrac = _segment_means(clipped_tok, lengths).mean()
                        v_loss = _segment_means(value_loss_tok, lengths).mean()
                        entropy_mean = _segment_means(entropy_new_f, lengths).mean()
                        kl_mean = _segment_means(kl_tok, lengths).mean()
                        kl_penalty = float(cfg.ppo.kl_coef) * kl_mean
                        entropy_loss = -entropy_mean

                        ce_loss = torch.zeros((), dtype=torch.float32, device=logp_new_f.device)
                        ce_used = 0
                        if (not value_warmup_active) and bool(ce_enabled) and ce_selected_global:
                            ce_selected_global_in_batch = [int(i) for i in batch_idx if int(i) in ce_selected_global]
                            batch_local_pos = {int(global_i): local_i for local_i, global_i in enumerate(batch_idx)}
                            ce_selected = [
                                int(batch_local_pos[gidx]) for gidx in ce_selected_global_in_batch if gidx in batch_local_pos
                            ]
                            ce_out, ce_used = _compute_digit_ce_for_minibatch(
                                model=model,
                                batch_trajs=batch_trajs,
                                selected_indices=ce_selected,
                                digit_token_ids=digit_token_ids,
                                answer_token_id=int(answer_token_id),
                                pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0,
                            )
                            if ce_out is not None:
                                ce_loss = ce_out.float()
                                for gidx in ce_selected_global_in_batch:
                                    ce_selected_global.discard(int(gidx))

                        ce_weighted = float(cfg.ppo.alpha_sft) * ce_loss
                        if value_warmup_active:
                            loss = cfg.ppo.c_v * v_loss
                        else:
                            loss = (
                                policy_loss
                                + kl_penalty
                                + cfg.ppo.c_v * v_loss
                                + cfg.ppo.c_ent * entropy_loss
                                + ce_weighted
                            )
                        loss = loss / float(cfg.train.grad_accum_steps)

                    loss.backward()
                    minibatch_count += 1

                    pol_acc += float(policy_loss.detach().item())
                    val_acc += float(v_loss.detach().item())
                    ent_acc += float(entropy_mean.detach().item())
                    ent_loss_acc += float(entropy_loss.detach().item())
                    clip_acc += float(clipfrac.detach().item())
                    kl_acc += float(kl_mean.detach().item())
                    kl_pen_acc += float(kl_penalty.detach().item())
                    ce_acc += float(ce_loss.detach().item())
                    ce_examples_acc += int(ce_used)

                    if minibatch_count % int(cfg.train.grad_accum_steps) == 0:
                        torch.nn.utils.clip_grad_norm_(active_clip_params, cfg.ppo.max_grad_norm)
                        active_optimizer.step()
                        active_optimizer.zero_grad(set_to_none=True)

            if minibatch_count % int(cfg.train.grad_accum_steps) != 0:
                torch.nn.utils.clip_grad_norm_(active_clip_params, cfg.ppo.max_grad_norm)
                active_optimizer.step()
                active_optimizer.zero_grad(set_to_none=True)
            t_backprop_sec = time.perf_counter() - _t_backprop0

            ref_refresh_every = max(int(cfg.ppo.update_ref_model_each_steps), 1)
            if ref_model is not None and update % ref_refresh_every == 0:
                ref_model.load_state_dict(model.state_dict())
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)

            rewards = torch.tensor([float(t.reward_info["reward_final"]) for t in trajectories], dtype=torch.float32)

            old_values = torch.tensor([v for t in trajectories for v in t.values_old], dtype=torch.float32)
            old_returns = torch.tensor([r for t in trajectories for r in t.returns], dtype=torch.float32)
            ev = explained_variance(y_pred=old_values, y_true=old_returns)

            num_traj = float(len(trajectories))
            finalize_exact_rate = float(
                sum(1 for t in trajectories if str(t.reward_info.get("reward_selection_mode", "")) == "finalize_exact")
            ) / num_traj
            early_success_discount_rate = float(
                sum(
                    1
                    for t in trajectories
                    if str(t.reward_info.get("reward_selection_mode", "")) == "early_success_discounted"
                )
            ) / num_traj
            finalize_partial_rate = float(
                sum(
                    1
                    for t in trajectories
                    if str(t.reward_info.get("reward_selection_mode", "")) == "finalize_partial_or_zero"
                )
            ) / num_traj
            non_finalize_no_credit_rate = float(
                sum(
                    1
                    for t in trajectories
                    if str(t.reward_info.get("reward_selection_mode", "")) == "non_finalize_no_credit"
                )
            ) / num_traj
            avg_rounds = float(
                sum(float(t.reward_info.get("round_count", 0.0)) for t in trajectories)
            ) / num_traj
            avg_verify_tokens = float(sum(len(t.generated_verify_ids) for t in trajectories)) / num_traj
            any_exact_rate = float(
                sum(
                    1
                    for t in trajectories
                    if bool(t.reward_info.get("any_exact_match", t.reward_info.get("exact_match", False)))
                )
            ) / num_traj
            reward_base_mean = float(
                sum(float(t.reward_info.get("reward", 0.0)) for t in trajectories)
            ) / num_traj
            token_penalty_mean = float(
                sum(float(t.reward_info.get("token_penalty", 0.0)) for t in trajectories)
            ) / num_traj
            rounds_penalty_mean = float(
                sum(float(t.reward_info.get("rounds_penalty", 0.0)) for t in trajectories)
            ) / num_traj
            reward_final_mean = float(rewards.mean().item())

            denom = max(minibatch_count, 1)
            t_reward_sec = _get_reward_timing_acc()
            t_total_sec = time.perf_counter() - _t_update0
            _log(
                " | ".join(
                    [
                        f"update={update}",
                        f"finalize_exact_rate={finalize_exact_rate:.4f}",
                        f"early_success_discount_rate={early_success_discount_rate:.4f}",
                        f"finalize_partial_rate={finalize_partial_rate:.4f}",
                        f"non_finalize_no_credit_rate={non_finalize_no_credit_rate:.4f}",
                        f"avg_rounds={avg_rounds:.3f}",
                        f"avg_verify_tokens={avg_verify_tokens:.3f}",
                        f"any_exact_rate={any_exact_rate:.4f}",
                        f"reward_base_mean={reward_base_mean:.4f}",
                        f"token_penalty_mean={token_penalty_mean:.4f}",
                        f"rounds_penalty_mean={rounds_penalty_mean:.4f}",
                        f"reward_final_mean={reward_final_mean:.4f}",
                        f"entropy={ent_acc / denom:.4f}",
                        f"clipfrac={clip_acc / denom:.4f}",
                        f"policy_loss={pol_acc / denom:.4f}",
                        f"value_loss={val_acc / denom:.4f}",
                        f"explained_var={ev:.4f}",
                        f"t_update={t_total_sec:.3f}s",
                    ]
                )
            )

            if update % int(cfg.train.save_every) == 0:
                _save_checkpoint(
                    output_dir=cfg.train.output_dir,
                    step=update,
                    model=model,
                    value_head=value_head,
                    tokenizer=tokenizer,
                    ppo_optimizer=ppo_optimizer,
                    warmup_optimizer=warmup_optimizer,
                    ds_index=ds_index,
                    prev_train_mode=prev_train_mode,
                    reward_rng=reward_rng,
                    cfg=cfg,
                )
                _rotate_checkpoints(output_dir=cfg.train.output_dir, keep_last=int(cfg.train.keep_last))
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
