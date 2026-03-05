from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from PPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.ppo_math import clipped_policy_loss, explained_variance, value_mse_loss
from PPO.reward import compute_reward, parse_answer_digits, parse_final_answer_to_digits
from PPO.rollout_contract import is_ppo_action, validate_action_scope
from PPO.rollout_logger import RolloutLogger
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single
from PPO.vllm_rollout import VLLMRolloutEngine
from phase1.dataset import SYSTEM_PROMPT


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
    print(f"{ts} | {msg}")


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
    probs = torch.softmax(logits, dim=-1)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
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

    logits_all = out.logits[0].to(torch.float32)
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
        probs_allowed = torch.softmax(allowed_logits, dim=-1)

        local_matches = torch.nonzero(allowed_t == action_id, as_tuple=False)
        if local_matches.numel() == 0:
            raise RuntimeError(f"Action id {action_id} not in allowed set for type={action_type}")
        local_idx = int(local_matches[0].item())

        logp_list.append(log_probs_allowed[local_idx])
        entropy_list.append((-(probs_allowed * torch.log(probs_allowed.clamp_min(1e-12))).sum()))

    h_states = hidden_all.index_select(0, state_positions)
    values = value_head(h_states.float()).squeeze(-1)
    logp = torch.stack(logp_list, dim=0)
    entropy = torch.stack(entropy_list, dim=0)
    return logp, values, entropy


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
    vllm_engine: VLLMRolloutEngine,
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

        if not logged_example:
            logger(
                f"vLLM Z example | finish_reason={finish_reason} | stop_reason={stop_reason} | "
                f"answer_in_token_ids={answer_in_tokens} | has_answer={has_answer}"
            )
            logged_example = True

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    model.train()

    action_scope = validate_action_scope(cfg.rollout.action_scope)
    if action_scope == "ppo_full" and bool(cfg.rollout.digit_greedy):
        raise ValueError("digit_greedy=True is incompatible with ppo_full")

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

    params = list(model.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )

    vllm_engine: Optional[VLLMRolloutEngine] = None
    if cfg.rollout.vllm_enabled:
        vllm_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
        vllm_kwargs.setdefault("device", "cuda:1")
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
            seed=int(cfg.train.seed),
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
            if vllm_engine is not None:
                synced = vllm_engine.maybe_sync_from_torch(model=model, tokenizer=tokenizer, update_idx=update)
                if synced:
                    _log(f"vLLM policy sync complete at update={update}")

            trajectories: List[Trajectory] = []
            token_budget = 0

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

            optimizer.zero_grad(set_to_none=True)
            minibatch_count = 0

            pol_acc = 0.0
            val_acc = 0.0
            ent_acc = 0.0
            clip_acc = 0.0

            order = list(range(len(trajectories)))
            random.shuffle(order)

            for _epoch in range(cfg.ppo.ppo_epochs):
                random.shuffle(order)
                for start in range(0, len(order), cfg.ppo.minibatch_size):
                    batch_idx = order[start : start + cfg.ppo.minibatch_size]

                    logp_new_chunks: List[torch.Tensor] = []
                    values_new_chunks: List[torch.Tensor] = []
                    entropy_chunks: List[torch.Tensor] = []
                    logp_old_chunks: List[torch.Tensor] = []
                    adv_chunks: List[torch.Tensor] = []
                    ret_chunks: List[torch.Tensor] = []

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and cfg.runtime.use_bf16
                        else nullcontext()
                    )
                    with amp_ctx:
                        for idx in batch_idx:
                            traj = trajectories[idx]
                            logp_new_t, values_new_t, entropy_t = _recompute_trajectory(
                                model=model,
                                value_head=value_head,
                                traj=traj,
                                z_allowed_t=z_allowed_t,
                                digit_allowed_t=digit_allowed_t,
                                temperature=cfg.rollout.temperature,
                            )

                            logp_new_chunks.append(logp_new_t)
                            values_new_chunks.append(values_new_t)
                            entropy_chunks.append(entropy_t)
                            logp_old_chunks.append(
                                torch.tensor(traj.logp_old, dtype=logp_new_t.dtype, device=logp_new_t.device)
                            )
                            adv_chunks.append(
                                torch.tensor(traj.advantages_norm, dtype=logp_new_t.dtype, device=logp_new_t.device)
                            )
                            ret_chunks.append(
                                torch.tensor(traj.returns, dtype=logp_new_t.dtype, device=logp_new_t.device)
                            )

                        logp_new = torch.cat(logp_new_chunks, dim=0)
                        values_new = torch.cat(values_new_chunks, dim=0)
                        entropy_new = torch.cat(entropy_chunks, dim=0)
                        logp_old = torch.cat(logp_old_chunks, dim=0)
                        advantages = torch.cat(adv_chunks, dim=0)
                        returns = torch.cat(ret_chunks, dim=0)

                        policy_loss, clipfrac = clipped_policy_loss(
                            logp_new=logp_new,
                            logp_old=logp_old,
                            advantages=advantages,
                            clip_range=cfg.ppo.clip_range,
                        )
                        v_loss = value_mse_loss(values=values_new, returns=returns)
                        entropy_loss = -entropy_new.mean()

                        loss = policy_loss + cfg.ppo.c_v * v_loss + cfg.ppo.c_ent * entropy_loss
                        loss = loss / float(cfg.train.grad_accum_steps)

                    loss.backward()
                    minibatch_count += 1

                    pol_acc += float(policy_loss.detach().item())
                    val_acc += float(v_loss.detach().item())
                    ent_acc += float(entropy_new.detach().mean().item())
                    clip_acc += float(clipfrac.detach().item())

                    if minibatch_count % int(cfg.train.grad_accum_steps) == 0:
                        torch.nn.utils.clip_grad_norm_(params, cfg.ppo.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            if minibatch_count % int(cfg.train.grad_accum_steps) != 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.ppo.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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
                        f"clipfrac={clip_acc / denom:.4f}",
                        f"policy_loss={pol_acc / denom:.4f}",
                        f"value_loss={val_acc / denom:.4f}",
                        f"explained_var={ev:.4f}",
                        f"rollouts={rollout_path}",
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
