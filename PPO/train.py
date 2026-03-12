from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from PPO.hf_rollout import HFRolloutEngine
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.ppo_math import explained_variance
from PPO.reward import compute_reward, parse_answer_digits, parse_final_answer_to_digits
from PPO.rollout_contract import is_ppo_action, validate_action_scope
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single
from PPO.vllm_rollout import VLLMRolloutEngine
from phase1.dataset import SYSTEM_PROMPT

_REWARD_TIME_ACC_SEC: float = 0.0
_RUN_LOG_PATH: Optional[str] = None


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
    ) -> None:
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

        self.returns = [float(self.reward_info["reward_final"])] * len(actions)
        self.advantages = [float(self.reward_info["reward_final"]) - float(v) for v in values_old]
        self.advantages_norm = list(self.advantages)


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
        allowed_t = digit_allowed_t if action_type == "digit" else z_allowed_t

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


def _action_stats_tensors_batched(
    *,
    model,
    ref_model,
    value_head: ValueHead,
    trajs: Sequence[Trajectory],
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    temperature: float,
    pad_token_id: int,
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

    seqs: List[List[int]] = []
    atts: List[List[int]] = []
    for t in trajs:
        seq = list(t.prompt_ids) + list(t.actions)
        att = list(t.prompt_attention_mask) + [1] * len(t.actions)
        seqs.append(seq)
        atts.append(att)

    max_len = max(len(s) for s in seqs)
    bsz = len(seqs)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, (seq, att) in enumerate(zip(seqs, atts)):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :L] = torch.tensor(att, dtype=torch.long, device=device)

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
    bias = getattr(lm_head, "bias", None)
    z_b = bias.index_select(0, z_allowed_t) if bias is not None else None
    d_b = bias.index_select(0, digit_allowed_t) if bias is not None else None

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
    ref_bias = getattr(ref_lm_head, "bias", None)
    ref_z_b = ref_bias.index_select(0, z_allowed_t) if ref_bias is not None else None
    ref_d_b = ref_bias.index_select(0, digit_allowed_t) if ref_bias is not None else None

    logp_new_chunks: List[torch.Tensor] = []
    logp_ref_chunks: List[torch.Tensor] = []
    values_new_chunks: List[torch.Tensor] = []
    entropy_chunks: List[torch.Tensor] = []
    logp_old_chunks: List[torch.Tensor] = []
    adv_chunks: List[torch.Tensor] = []
    ret_chunks: List[torch.Tensor] = []
    lengths: List[int] = []

    for b, traj in enumerate(trajs):
        t_steps = len(traj.actions)
        if t_steps == 0:
            continue
        lengths.append(t_steps)

        p_len = len(traj.prompt_ids)
        state_positions = torch.arange(
            p_len - 1,
            p_len - 1 + t_steps,
            device=device,
            dtype=torch.long,
        )

        act_ids = torch.tensor(traj.actions, dtype=torch.long, device=device)  # [T]
        is_digit = torch.tensor([t == "digit" for t in traj.action_types], dtype=torch.bool, device=device)  # [T]

        local_z = z_id_to_local[act_ids]  # [T]
        local_d = d_id_to_local[act_ids]  # [T]
        invalid = torch.where(is_digit, local_d < 0, local_z < 0)
        if bool(invalid.any()):
            bad_idx = torch.nonzero(invalid, as_tuple=False).squeeze(-1)[:8]
            bad_ids = act_ids.index_select(0, bad_idx).tolist()
            bad_types = [traj.action_types[int(i)] for i in bad_idx.tolist()]
            raise RuntimeError(f"Found actions not in allowed set: ids={bad_ids}, types={bad_types}")

        h_states = hidden_all[b].index_select(0, state_positions)  # [T,H]
        z_logits = (h_states @ z_w.t()) / float(temperature)  # [T,|Z|]
        d_logits = (h_states @ d_w.t()) / float(temperature)  # [T,|D|]
        if z_b is not None:
            z_logits = z_logits + z_b
        if d_b is not None:
            d_logits = d_logits + d_b

        z_logp = torch.log_softmax(z_logits, dim=-1)
        d_logp = torch.log_softmax(d_logits, dim=-1)

        z_probs = z_logp.exp()
        d_probs = d_logp.exp()
        z_ent = -(z_probs * z_logp).sum(dim=-1)  # [T]
        d_ent = -(d_probs * d_logp).sum(dim=-1)  # [T]

        local_z_safe = local_z.clamp_min(0)
        local_d_safe = local_d.clamp_min(0)
        z_chosen = z_logp.gather(1, local_z_safe.view(-1, 1)).squeeze(1)  # [T]
        d_chosen = d_logp.gather(1, local_d_safe.view(-1, 1)).squeeze(1)  # [T]

        logp_vec = torch.where(is_digit, d_chosen, z_chosen)  # [T]
        ent_vec = torch.where(is_digit, d_ent, z_ent)  # [T]

        with torch.no_grad():
            ref_h_states = ref_hidden_all[b].index_select(0, state_positions)  # [T,H]
            ref_z_logits = (ref_h_states @ ref_z_w.t()) / float(temperature)  # [T,|Z|]
            ref_d_logits = (ref_h_states @ ref_d_w.t()) / float(temperature)  # [T,|D|]
            if ref_z_b is not None:
                ref_z_logits = ref_z_logits + ref_z_b
            if ref_d_b is not None:
                ref_d_logits = ref_d_logits + ref_d_b

            ref_z_logp = torch.log_softmax(ref_z_logits, dim=-1)
            ref_d_logp = torch.log_softmax(ref_d_logits, dim=-1)
            ref_z_chosen = ref_z_logp.gather(1, local_z_safe.view(-1, 1)).squeeze(1)  # [T]
            ref_d_chosen = ref_d_logp.gather(1, local_d_safe.view(-1, 1)).squeeze(1)  # [T]
            logp_ref_vec = torch.where(is_digit, ref_d_chosen, ref_z_chosen)  # [T]

        values = value_head(h_states.float()).squeeze(-1)

        logp_new_chunks.append(logp_vec)
        logp_ref_chunks.append(logp_ref_vec)
        values_new_chunks.append(values)
        entropy_chunks.append(ent_vec)
        logp_old_chunks.append(torch.tensor(traj.logp_old, dtype=torch.float32, device=device))
        adv_chunks.append(torch.tensor(traj.advantages_norm, dtype=torch.float32, device=device))
        ret_chunks.append(torch.tensor(traj.returns, dtype=torch.float32, device=device))

    if not logp_new_chunks:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return empty, empty, empty, empty, empty, empty, empty, torch.tensor(lengths, dtype=torch.long, device=device)

    return (
        torch.cat(logp_new_chunks, dim=0),
        torch.cat(logp_ref_chunks, dim=0),
        torch.cat(values_new_chunks, dim=0),
        torch.cat(entropy_chunks, dim=0),
        torch.cat(logp_old_chunks, dim=0),
        torch.cat(adv_chunks, dim=0),
        torch.cat(ret_chunks, dim=0),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _validate_actions_in_allowed(
    *,
    actions: Sequence[int],
    action_types: Sequence[str],
    z_allowed_set: set[int],
    digit_allowed_set: set[int],
) -> None:
    if len(actions) != len(action_types):
        raise RuntimeError("actions/action_types length mismatch")
    for a, t in zip(actions, action_types):
        aid = int(a)
        if t == "digit":
            if aid not in digit_allowed_set:
                raise RuntimeError(f"Digit action id {aid} not in digit allowed set")
        else:
            if aid not in z_allowed_set:
                raise RuntimeError(f"Z/answer action id {aid} not in Z allowed set")


def _rollout_one_torch(
    *,
    model,
    value_head: ValueHead,
    tokenizer,
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
    _validate_actions_in_allowed(
        actions=actions,
        action_types=action_types,
        z_allowed_set=z_allowed_set,
        digit_allowed_set=digit_allowed_set,
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
        reward_if_max_len=reward_cfg.reward_if_max_len,
        num_generated_tokens=int(seq.size(1) - len(prompt_ids)),
        generator=reward_rng,
    )
    _add_reward_timing_acc(time.perf_counter() - _t_reward0)

    return Trajectory(
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
    _validate_actions_in_allowed(
        actions=actions,
        action_types=action_types,
        z_allowed_set=z_allowed_set,
        digit_allowed_set=digit_allowed_set,
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
        reward_if_max_len=reward_cfg.reward_if_max_len,
        num_generated_tokens=len(generated_z_ids) + (1 if has_answer else 0) + len(generated_digit_ids),
        generator=reward_rng,
    )
    _add_reward_timing_acc(time.perf_counter() - _t_reward0)

    return Trajectory(
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
    cfg: Config,
    z_allowed_t: torch.Tensor,
    digit_allowed_t: torch.Tensor,
    answer_token_id: int,
    digit_token_ids: Sequence[int],
    reward_rng: torch.Generator,
    logger,
) -> List[Trajectory]:
    supports_token_prompts = vllm_engine.supports_prompt_token_ids()
    prompt_texts = [str(x["prompt_text"]) for x in prepared]
    prompt_ids_batch = [list(map(int, x["prompt_ids"])) for x in prepared]
    z_gen_rows = vllm_engine.generate_z(
        prompts=prompt_texts,
        prompt_token_ids=prompt_ids_batch if supports_token_prompts else None,
        max_new_tokens=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
    )

    with_answer_idx: List[int] = []
    z_prefix_by_idx: Dict[int, List[int]] = {}
    digit_prompt_ids_batch: List[List[int]] = []
    digit_prompt_texts: List[str] = []

    logged_example = False
    for i, row in enumerate(z_gen_rows):
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
            digit_prompt_ids = list(map(int, prepared[i]["prompt_ids"])) + z_prefix + [int(answer_token_id)]
            assert digit_prompt_ids[-1] == int(answer_token_id)
            prepared_prompt_ids = list(map(int, prepared[i]["prompt_ids"]))
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
    for i, item in enumerate(prepared):
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
            sample_id=str(item["sample_id"]),
            z_allowed_t=z_allowed_t,
            digit_allowed_t=digit_allowed_t,
            temperature=cfg.rollout.temperature,
            terminated_by=terminated_by,
        )
        trajectories.append(traj)

    return trajectories


def _normalize_advantages(trajectories: Sequence[Trajectory]) -> None:
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
        t.advantages_norm = [(a - mean) / denom for a in t.advantages]


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
    cap = int(math.ceil(frac * len(batch_trajs)))
    if cap <= 0:
        return []
    cap = min(cap, len(batch_trajs))

    valid = [i for i, t in enumerate(batch_trajs) if _traj_has_valid_ce_target(t)]
    if not valid:
        return []
    if cap > len(valid):
        cap = len(valid)
    if cap <= 0:
        return []

    mode = str(ce_mode).strip().lower()
    if mode == "random":
        picked = random.sample(valid, k=cap)
        picked.sort()
        return picked

    if mode != "successful_traces":
        raise ValueError(f"Unsupported ppo.ce_mode={ce_mode!r}; expected 'successful_traces' or 'random'")

    # Successful traces are exact-match only; no fallback to non-exact trajectories.
    success = [i for i in valid if bool(batch_trajs[i].reward_info.get("exact_match", False))]
    if not success:
        return []
    k = min(cap, len(success))
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
        temperature=temperature,
    )


def _save_checkpoint(
    *,
    output_dir: str,
    step: int,
    model,
    value_head: ValueHead,
    tokenizer,
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
            "value_head_state_dict": value_head.state_dict(),
        },
        os.path.join(ckpt_dir, "ppo_state.pt"),
    )


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
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    action_scope = validate_action_scope(cfg.rollout.action_scope)
    if action_scope == "ppo_full" and bool(cfg.rollout.digit_greedy):
        raise ValueError("digit_greedy=True is incompatible with ppo_full")
    ce_mode = str(cfg.ppo.ce_mode).strip().lower()
    if ce_mode not in ("successful_traces", "random"):
        raise ValueError(
            f"Unsupported ppo.ce_mode={cfg.ppo.ce_mode!r}; expected 'successful_traces' or 'random'"
        )
    if float(cfg.ppo.batch_frac_to_apply_ce) < 0.0:
        raise ValueError("ppo.batch_frac_to_apply_ce must be >= 0")

    z_token_ids, z_style = introspect_z_token_ids_and_style(tokenizer)
    if not z_token_ids:
        raise RuntimeError("No Z tokens found in tokenizer (checked lowercase <z_i> then uppercase <Z_i>)")
    if z_style == "upper":
        _log("WARNING: using uppercase <Z_i> tokens fallback; lowercase <z_i> not found")

    answer_token_id = resolve_answer_token_id(tokenizer, answer_token=cfg.model.answer_token)
    validate_answer_token_single(tokenizer, cfg.model.answer_token, answer_token_id)
    digit_token_ids = resolve_digit_token_ids(tokenizer)

    z_allowed_t = torch.tensor(list(z_token_ids) + [int(answer_token_id)], dtype=torch.long, device=device)
    digit_allowed_t = torch.tensor(list(digit_token_ids), dtype=torch.long, device=device)

    _log(
        f"Action scope={action_scope} | Z tokens={len(z_token_ids)} ({z_style}) | "
        f"answer_token_id={answer_token_id}"
    )

    hidden_size = int(model.config.hidden_size)
    value_head = ValueHead(hidden_size=hidden_size).to(device)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model output embeddings (LM head) are unavailable")
    vocab_size = int(lm_head.weight.size(0))
    z_id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    d_id_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    z_id_to_local[z_allowed_t] = torch.arange(z_allowed_t.numel(), device=device, dtype=torch.long)
    d_id_to_local[digit_allowed_t] = torch.arange(digit_allowed_t.numel(), device=device, dtype=torch.long)
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

    params = list(model.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )

    rollout_backend = str(getattr(cfg.rollout, "backend", "vllm")).strip().lower()
    vllm_engine: Optional[Any] = None
    if rollout_backend == "hf":
        vllm_engine = HFRolloutEngine(
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=z_allowed_t.tolist(),
            digit_allowed_token_ids=digit_allowed_t.tolist(),
            sync_every=int(cfg.rollout.vllm_sync_every),
            logger=_log,
        )
    elif cfg.rollout.vllm_enabled:
        vllm_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
        vllm_kwargs.setdefault("tensor_parallel_size", int(cfg.rollout.vllm_tp_size))
        vllm_kwargs.setdefault("weight_transfer_device", str(device))
        if int(cfg.rollout.vllm_tp_size) == 1:
            vllm_cvd = str(getattr(cfg.rollout, "vllm_cuda_visible_devices", "")).strip()
            if vllm_cvd:
                vllm_kwargs.setdefault("cuda_visible_devices", vllm_cvd)
                _log(f"vLLM CUDA_VISIBLE_DEVICES={vllm_kwargs['cuda_visible_devices']}")
        vllm_seed = int(cfg.rollout.vllm_seed) if cfg.rollout.vllm_seed is not None else int(cfg.train.seed)
        vllm_engine = VLLMRolloutEngine(
            init_ckpt=cfg.model.init_ckpt,
            tokenizer=tokenizer,
            answer_token_id=int(answer_token_id),
            z_allowed_token_ids=z_allowed_t.tolist(),
            digit_allowed_token_ids=digit_allowed_t.tolist(),
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

    reward_rng = _make_rng(cfg.train.seed + 17)
    rollout_logger = RolloutLogger(os.path.join(cfg.train.output_dir, "rollouts"))

    ds_index = 0
    try:
        for update in range(1, cfg.train.updates + 1):
            _t_update0 = time.perf_counter()
            _reset_reward_timing_acc()

            t_sync_sec = 0.0
            if vllm_engine is not None:
                _t_sync0 = time.perf_counter()
                synced = vllm_engine.maybe_sync_from_torch(model=model, tokenizer=tokenizer, update_idx=update)
                t_sync_sec += time.perf_counter() - _t_sync0
                # if synced:
                #     _log(f"vLLM policy sync complete at update={update}")

            trajectories: List[Trajectory] = []
            token_budget = 0
            _t_rollout0 = time.perf_counter()

            while len(trajectories) < cfg.rollout.episodes_per_batch:
                remaining = cfg.rollout.episodes_per_batch - len(trajectories)
                this_batch = min(int(cfg.rollout.vllm_batch_size), int(remaining))

                prepared: List[Dict[str, object]] = []
                while len(prepared) < this_batch:
                    sample = ds[int(ds_index % len(ds))]
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
                            "sample_id": f"u{update}_i{len(trajectories) + len(prepared)}",
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
                    batch_trajs = _collect_rollouts_vllm_batch(
                        model=model,
                        value_head=value_head,
                        tokenizer=tokenizer,
                        vllm_engine=vllm_engine,
                        prepared=prepared,
                        cfg=cfg,
                        z_allowed_t=z_allowed_t,
                        digit_allowed_t=digit_allowed_t,
                        answer_token_id=int(answer_token_id),
                        digit_token_ids=digit_token_ids,
                        reward_rng=reward_rng,
                        logger=_log,
                    )
                else:
                    batch_trajs = [
                        _rollout_one_torch(
                            model=model,
                            value_head=value_head,
                            tokenizer=tokenizer,
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
                        )
                        for item in prepared
                    ]

                for traj in batch_trajs:
                    if not traj.actions:
                        continue
                    trajectories.append(traj)
                    token_budget += len(traj.actions)
                    if token_budget >= int(cfg.rollout.max_tokens_per_batch):
                        break
                if token_budget >= int(cfg.rollout.max_tokens_per_batch):
                    break
            t_rollout_sec = time.perf_counter() - _t_rollout0

            if not trajectories:
                raise RuntimeError("No trajectories collected for PPO update")

            if cfg.ppo.normalize_advantages:
                _normalize_advantages(trajectories)

            roll_rows: List[Dict[str, object]] = []
            for traj in trajectories:
                row = {
                    "schema_version": 2,
                    "id": traj.sample_id,
                    "question": traj.question,
                    "input_ids": traj.prompt_ids,
                    "generated_z_ids": traj.generated_z_ids,
                    "generated_z_tokens": tokenizer.convert_ids_to_tokens(traj.generated_z_ids),
                    "generated_digit_ids": traj.generated_digit_ids,
                    "generated_digit_tokens": tokenizer.convert_ids_to_tokens(traj.generated_digit_ids),
                    "terminated_by": traj.terminated_by,
                    "num_generated": traj.num_generated_total,
                    "num_digits_generated": traj.num_digits_generated,
                    "digit_logits": traj.digit_logits,
                    "digit_probs": traj.digit_probs,
                    "digit_pred": traj.digit_pred,
                    "digit_true": traj.digit_true,
                    "reward_full": traj.reward_info["reward_full"],
                    "partial_scale": traj.reward_info["partial_scale"],
                    "keep_prob": traj.reward_info["keep_prob"],
                    "applied_mask": traj.reward_info["applied_mask"],
                    "applied_count": traj.reward_info["applied_count"],
                    "correct_count": traj.reward_info["correct_count"],
                    "reward_partial": traj.reward_info["reward_partial"],
                    "length_penalty": traj.reward_info["length_penalty"],
                    "reward_if_max_len": traj.reward_info["reward_if_max_len"],
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
            optimizer.zero_grad(set_to_none=True)
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

            order = list(range(len(trajectories)))
            random.shuffle(order)

            for _epoch in range(cfg.ppo.ppo_epochs):
                random.shuffle(order)
                for start in range(0, len(order), cfg.ppo.minibatch_size):
                    batch_idx = order[start : start + cfg.ppo.minibatch_size]
                    batch_trajs = [trajectories[idx] for idx in batch_idx]

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
                            z_allowed_t=z_allowed_t,
                            digit_allowed_t=digit_allowed_t,
                            z_id_to_local=z_id_to_local,
                            d_id_to_local=d_id_to_local,
                            temperature=cfg.rollout.temperature,
                            pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0,
                        )

                        lengths_list = [int(x) for x in lengths.tolist()]
                        total_tokens = int(sum(lengths_list))
                        if int(logp_new.numel()) != total_tokens:
                            raise RuntimeError(
                                f"Token count mismatch: T={int(logp_new.numel())}, sum(lengths)={total_tokens}"
                            )

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

                        ppo_loss_split = torch.split(policy_loss_tok, lengths_list)
                        clip_split = torch.split(clipped_tok, lengths_list)
                        value_split = torch.split(value_loss_tok, lengths_list)
                        entropy_split = torch.split(entropy_new_f, lengths_list)
                        kl_split = torch.split(kl_tok, lengths_list)

                        policy_means = [chunk.mean() for chunk, L in zip(ppo_loss_split, lengths_list) if L > 0]
                        clip_means = [chunk.mean() for chunk, L in zip(clip_split, lengths_list) if L > 0]
                        value_means = [chunk.mean() for chunk, L in zip(value_split, lengths_list) if L > 0]
                        entropy_means = [chunk.mean() for chunk, L in zip(entropy_split, lengths_list) if L > 0]
                        kl_means = [chunk.mean() for chunk, L in zip(kl_split, lengths_list) if L > 0]

                        if not policy_means:
                            continue

                        policy_loss = torch.stack(policy_means).mean()
                        clipfrac = torch.stack(clip_means).mean()
                        v_loss = torch.stack(value_means).mean()
                        entropy_mean = torch.stack(entropy_means).mean()
                        kl_mean = torch.stack(kl_means).mean()
                        kl_penalty = float(cfg.ppo.kl_coef) * kl_mean
                        entropy_loss = -entropy_mean

                        ce_loss = torch.zeros((), dtype=torch.float32, device=logp_new_f.device)
                        ce_used = 0
                        if bool(cfg.ppo.apply_ce):
                            ce_selected = _select_ce_trajectory_indices(
                                batch_trajs=batch_trajs,
                                batch_frac_to_apply_ce=float(cfg.ppo.batch_frac_to_apply_ce),
                                ce_mode=ce_mode,
                            )
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

                        ce_weighted = float(cfg.ppo.alpha_sft) * ce_loss
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
                        torch.nn.utils.clip_grad_norm_(params, cfg.ppo.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            if minibatch_count % int(cfg.train.grad_accum_steps) != 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.ppo.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            t_backprop_sec = time.perf_counter() - _t_backprop0

            ref_refresh_every = max(int(cfg.ppo.update_ref_model_each_steps), 1)
            if update % ref_refresh_every == 0:
                ref_model.load_state_dict(model.state_dict())
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)

            rewards = torch.tensor([float(t.reward_info["reward_final"]) for t in trajectories], dtype=torch.float32)
            exact_rate = float(
                sum(1 for t in trajectories if bool(t.reward_info.get("exact_match", False)))
            ) / float(len(trajectories))
            answered = sum(1 for t in trajectories if t.terminated_by == "answer_with_5_digits")
            answer_rate = float(answered) / float(len(trajectories))
            avg_len = float(sum(t.num_generated_total for t in trajectories)) / float(len(trajectories))

            old_values = torch.tensor([v for t in trajectories for v in t.values_old], dtype=torch.float32)
            old_returns = torch.tensor([r for t in trajectories for r in t.returns], dtype=torch.float32)
            ev = explained_variance(y_pred=old_values, y_true=old_returns)

            denom = max(minibatch_count, 1)
            t_reward_sec = _get_reward_timing_acc()
            t_total_sec = time.perf_counter() - _t_update0
            _log(
                " | ".join(
                    [
                        f"update={update}",
                        f"episodes={len(trajectories)}",
                        f"tokens={sum(len(t.actions) for t in trajectories)}",
                        f"reward_mean={float(rewards.mean().item()):.4f}",
                        f"exact={exact_rate:.4f}",
                        f"answer_rate={answer_rate:.4f}",
                        f"avg_len={avg_len:.2f}",
                        f"entropy={ent_acc / denom:.4f}",
                        f"kl={kl_acc / denom:.4f}",
                        f"kl_penalty={kl_pen_acc / denom:.4f}",
                        f"entropy_loss={ent_loss_acc / denom:.4f}",
                        f"clipfrac={clip_acc / denom:.4f}",
                        f"policy_loss={pol_acc / denom:.4f}",
                        f"value_loss={val_acc / denom:.4f}",
                        f"ce_loss={ce_acc / denom:.4f}",
                        f"ce_examples={ce_examples_acc}",
                        f"explained_var={ev:.4f}",
                        f"rollouts={rollout_path}",
                        f"t_sync={t_sync_sec:.3f}s",
                        f"t_rollout={t_rollout_sec:.3f}s",
                        f"t_reward={t_reward_sec:.3f}s",
                        f"t_backprop={t_backprop_sec:.3f}s",
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
