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
from transformers import AutoTokenizer

from PPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from PPO.masking import build_allowed_token_ids
from PPO.ppo_math import clipped_policy_loss, explained_variance, value_mse_loss
from PPO.reward import compute_reward, parse_final_answer_to_digits
from PPO.rollout_logger import RolloutLogger
from z_pipeline.phase23.model import UnifiedZSoftModel
from z_pipeline.phase23.utils import build_prompt, set_seed


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # Explicitly zero-init final layer for stable early PPO updates.
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
        logp_old: List[float],
        values_old: List[float],
        entropy_old: List[float],
        terminated_by: str,
        digit_logits: List[List[float]],
        digit_probs: List[List[float]],
        digit_pred: List[int],
        digit_true: List[int],
        reward_info: Dict[str, object],
    ) -> None:
        self.sample_id = sample_id
        self.question = question
        self.prompt_ids = prompt_ids
        self.prompt_attention_mask = prompt_attention_mask
        self.actions = actions
        self.logp_old = logp_old
        self.values_old = values_old
        self.entropy_old = entropy_old
        self.terminated_by = terminated_by
        self.digit_logits = digit_logits
        self.digit_probs = digit_probs
        self.digit_pred = digit_pred
        self.digit_true = digit_true
        self.reward_info = reward_info

        self.returns = [float(self.reward_info["reward_final"])] * len(actions)
        self.advantages = [float(self.reward_info["reward_final"]) - float(v) for v in values_old]
        self.advantages_norm = list(self.advantages)


def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{ts} | {msg}")


def _resolve_checkpoint_path(init_ckpt: str) -> str:
    if os.path.isdir(init_ckpt):
        return init_ckpt

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=init_ckpt)
    except Exception:
        # Fall back to direct path/repo usage in from_phase1.
        return init_ckpt


def _load_phase23_bundle(cfg: Config, device: torch.device):
    ckpt_ref = _resolve_checkpoint_path(cfg.model.init_ckpt)
    state_path = os.path.join(ckpt_ref, "phase23_state.pt")
    config_path = os.path.join(ckpt_ref, "config.json")

    if os.path.isdir(ckpt_ref) and os.path.exists(state_path) and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            phase23_cfg = json.load(f)
        model_cfg = phase23_cfg.get("model", {})
        bundle = UnifiedZSoftModel.from_phase1(
            phase1_dir=model_cfg["phase1_dir"],
            v_z=int(model_cfg.get("v_z", cfg.model.v_z_fallback)),
            device=device,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            z_prefix=str(model_cfg.get("z_prefix", cfg.model.z_prefix)),
            latent_token=str(model_cfg.get("latent_token", "<|latent|>")),
            answer_token=str(model_cfg.get("answer_token", cfg.model.answer_token)),
        )
        state = torch.load(state_path, map_location="cpu")
        bundle.model.load_state_dict(state, strict=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_ref, trust_remote_code=True)
            bundle.tokenizer = tokenizer
        except Exception:
            pass
        return bundle

    return UnifiedZSoftModel.from_phase1(
        phase1_dir=cfg.model.init_ckpt,
        v_z=cfg.model.v_z_fallback,
        device=device,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        z_prefix=cfg.model.z_prefix,
        answer_token=cfg.model.answer_token,
    )


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


def _make_rng(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device="cpu") if device.type == "cpu" else torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _compute_digit_outputs(phase23: UnifiedZSoftModel, seq_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = torch.ones_like(seq_ids)
    core = phase23._core_model()
    out = core(
        input_ids=seq_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_last = out.last_hidden_state[0]

    answer_mask = seq_ids[0] == phase23.answer_token_id
    if bool(answer_mask.any()):
        pos = int(torch.argmax(answer_mask.to(torch.int64)).item())
    else:
        pos = int(seq_ids.size(1) - 1)

    h = hidden_last[pos]
    digit_logits = torch.stack([head(h) for head in phase23.digit_heads], dim=0)
    digit_probs = torch.softmax(digit_logits, dim=-1)
    digit_pred = torch.argmax(digit_logits, dim=-1)
    return digit_logits, digit_probs, digit_pred


def _forward_last_with_cache(core, input_ids, attention_mask, past_key_values):
    return core(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        output_hidden_states=False,
        return_dict=True,
    )


def _allowed_logits_from_hidden(
    phase23: UnifiedZSoftModel,
    hidden: torch.Tensor,
    allowed_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute logits only for allowed token ids.
    hidden: [N,H]
    returns: [N,A]
    """
    if hidden.dim() != 2:
        raise ValueError("hidden must be [N,H]")
    head = phase23._get_lm_head()
    weight = head.weight.index_select(0, allowed_idx)
    h = hidden.to(dtype=weight.dtype)
    logits = h @ weight.t()
    bias = getattr(head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, allowed_idx).to(device=logits.device, dtype=logits.dtype)
    return logits


def _compute_digit_outputs_with_attention(
    phase23: UnifiedZSoftModel,
    seq_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = phase23._core_model()
    out = core(
        input_ids=seq_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_last = out.last_hidden_state[0]

    answer_mask = (seq_ids[0] == phase23.answer_token_id) & (attention_mask[0] == 1)
    if bool(answer_mask.any()):
        pos = int(torch.argmax(answer_mask.to(torch.int64)).item())
    else:
        pos = int((attention_mask[0].sum() - 1).clamp(min=0).item())

    h = hidden_last[pos]
    digit_logits = torch.stack([head(h) for head in phase23.digit_heads], dim=0)
    digit_probs = torch.softmax(digit_logits, dim=-1)
    digit_pred = torch.argmax(digit_logits, dim=-1)
    return digit_logits, digit_probs, digit_pred


def _rollout_one(
    *,
    phase23: UnifiedZSoftModel,
    value_head: ValueHead,
    tokenizer,
    question: str,
    true_digits: Sequence[int],
    allowed_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    reward_cfg,
    reward_rng: torch.Generator,
    sample_id: str,
) -> Trajectory:
    prompt = build_prompt(tokenizer, question)
    prompt_pack = tokenizer(prompt, add_special_tokens=False, return_attention_mask=True)
    prompt_ids = prompt_pack["input_ids"]
    prompt_attn = prompt_pack.get("attention_mask")
    if prompt_attn is None:
        prompt_attn = [1] * len(prompt_ids)
    if not prompt_ids:
        raise RuntimeError("Prompt tokenization produced an empty sequence")

    device = next(phase23.parameters()).device
    seq = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_attn = torch.tensor(prompt_attn, dtype=torch.long, device=device).unsqueeze(0)
    allowed_idx_t = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=device)

    logp_old: List[float] = []
    values_old: List[float] = []
    entropy_old: List[float] = []
    actions: List[int] = []
    terminated_by = "max_new_tokens"

    core = phase23._core_model()

    with torch.no_grad():
        out0 = core(
            input_ids=seq,
            attention_mask=seq_attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out0.past_key_values
        valid_pos = (seq_attn.sum(dim=1) - 1).clamp(min=0)
        bidx = torch.arange(seq.size(0), device=device)
        h_last = out0.last_hidden_state[bidx, valid_pos, :]

        for _step in range(max_new_tokens):
            allowed_logits = _allowed_logits_from_hidden(phase23, h_last, allowed_idx_t).squeeze(0)
            allowed_logits = allowed_logits / float(temperature)
            logp_allowed = torch.log_softmax(allowed_logits, dim=-1)
            allowed_probs = torch.softmax(allowed_logits, dim=-1)
            entropy = -(allowed_probs * torch.log(allowed_probs.clamp_min(1e-12))).sum()
            sampled_local = int(torch.multinomial(allowed_probs, num_samples=1).item())
            action = int(allowed_idx_t[sampled_local].item())

            v = value_head(h_last.float()).squeeze(-1)
            logp_old.append(float(logp_allowed[sampled_local].item()))
            values_old.append(float(v.item()))
            entropy_old.append(float(entropy.item()))
            actions.append(int(action))

            action_t = torch.tensor([[int(action)]], dtype=torch.long, device=device)
            seq = torch.cat([seq, action_t], dim=1)
            seq_attn = torch.cat(
                [seq_attn, torch.ones((1, 1), dtype=seq_attn.dtype, device=device)],
                dim=1,
            )
            if int(action) == int(phase23.answer_token_id):
                terminated_by = "answer"
                break
            out1 = _forward_last_with_cache(
                core=core,
                input_ids=action_t,
                attention_mask=seq_attn,
                past_key_values=past,
            )
            past = out1.past_key_values
            h_last = out1.last_hidden_state[:, -1, :]

        digit_logits_t, digit_probs_t, digit_pred_t = _compute_digit_outputs_with_attention(
            phase23=phase23,
            seq_ids=seq,
            attention_mask=seq_attn,
        )

    pred_digits = [int(x) for x in digit_pred_t.tolist()]
    reward_info = compute_reward(
        pred_digits=pred_digits,
        true_digits=true_digits,
        terminated_by_answer=(terminated_by == "answer"),
        partial_scale=reward_cfg.partial_scale,
        keep_prob=reward_cfg.keep_prob,
        length_penalty=reward_cfg.length_penalty,
        num_generated_tokens=len(actions),
        generator=reward_rng,
    )

    return Trajectory(
        sample_id=sample_id,
        question=question,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attn,
        actions=actions,
        logp_old=logp_old,
        values_old=values_old,
        entropy_old=entropy_old,
        terminated_by=terminated_by,
        digit_logits=digit_logits_t.float().cpu().tolist(),
        digit_probs=digit_probs_t.float().cpu().tolist(),
        digit_pred=pred_digits,
        digit_true=[int(x) for x in true_digits],
        reward_info=reward_info,
    )


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
    phase23: UnifiedZSoftModel,
    value_head: ValueHead,
    traj: Trajectory,
    allowed_idx_t: torch.Tensor,
    allowed_id_to_local: Dict[int, int],
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(phase23.parameters()).device
    seq_ids = torch.tensor(traj.prompt_ids + traj.actions, dtype=torch.long, device=device).unsqueeze(0)
    full_attn_list = list(traj.prompt_attention_mask) + [1] * len(traj.actions)
    attn = torch.tensor(full_attn_list, dtype=torch.long, device=device).unsqueeze(0)

    core = phase23._core_model()
    out = core(
        input_ids=seq_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = out.last_hidden_state[0]  # [L,H]

    p_len = len(traj.prompt_ids)
    t_steps = len(traj.actions)
    if t_steps == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return empty, empty, empty

    full_attn = attn.squeeze(0).to(torch.long)  # [L]
    if not bool(torch.all(full_attn == 1)):
        raise RuntimeError(
            "Sparse attention masks not supported in PPO v1; "
            "expected all-ones attention for unpadded sequences."
        )
    state_positions = torch.arange(
        p_len - 1,
        p_len - 1 + t_steps,
        device=device,
        dtype=torch.long,
    )

    h_states = hidden.index_select(0, state_positions)  # [T,H]
    logits_allowed = _allowed_logits_from_hidden(phase23, h_states, allowed_idx_t).to(torch.float32)  # [T,A]
    logits_allowed = logits_allowed / float(temperature)

    log_probs_allowed = torch.log_softmax(logits_allowed, dim=-1)
    probs_allowed = torch.softmax(logits_allowed, dim=-1)
    entropy = -(probs_allowed * torch.log(probs_allowed.clamp_min(1e-12))).sum(dim=-1)  # [T]

    action_pos_list = [int(allowed_id_to_local.get(int(a), -1)) for a in traj.actions]
    action_pos = torch.tensor(action_pos_list, dtype=torch.long, device=device)
    if (action_pos < 0).any():
        actions_t = torch.tensor(traj.actions, dtype=torch.long, device=device)
        bad = actions_t[action_pos < 0][:10].tolist()
        raise RuntimeError(f"Found action ids not in allowed set: {bad}")

    logp_new = log_probs_allowed.gather(1, action_pos.view(-1, 1)).squeeze(1)  # [T]
    values = value_head(h_states.float()).squeeze(-1)  # [T]
    return logp_new, values, entropy


def _save_checkpoint(
    *,
    output_dir: str,
    step: int,
    phase23: UnifiedZSoftModel,
    value_head: ValueHead,
    tokenizer,
    cfg: Config,
) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    phase23.base.save_pretrained(os.path.join(ckpt_dir, "model"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.save(
        {
            "phase23_state_dict": phase23.state_dict(),
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
    set_seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.train.output_dir, "rollouts"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    bundle = _load_phase23_bundle(cfg, device=device)
    tokenizer = bundle.tokenizer
    phase23 = bundle.model
    phase23.train()

    allowed_token_ids = build_allowed_token_ids(tokenizer, answer_token=cfg.model.answer_token)
    answer_token_id = int(allowed_token_ids[-1])
    z_token_ids = allowed_token_ids[:-1]
    if set(int(x) for x in z_token_ids) != set(int(x) for x in phase23.z_token_ids):
        raise RuntimeError("Tokenizer-introspected Z token ids do not match model.z_token_ids")
    if int(answer_token_id) != int(phase23.answer_token_id):
        raise RuntimeError("Tokenizer answer token id does not match model.answer_token_id")
    _log(f"Allowed Z tokens: {len(z_token_ids)} | answer_token_id={answer_token_id}")
    allowed_idx_t = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=device)
    allowed_id_to_local = {int(tok_id): i for i, tok_id in enumerate(allowed_token_ids)}

    hidden_size = int(phase23._core_model().config.hidden_size)
    value_head = ValueHead(hidden_size=hidden_size).to(device)

    params = list(phase23.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )

    ds = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)
    if len(ds) == 0:
        raise RuntimeError("Training dataset is empty")

    reward_rng = _make_rng(cfg.train.seed + 17, device=device)
    rollout_logger = RolloutLogger(os.path.join(cfg.train.output_dir, "rollouts"))

    ds_index = 0
    for update in range(1, cfg.train.updates + 1):
        trajectories: List[Trajectory] = []
        token_budget = 0

        while len(trajectories) < cfg.rollout.episodes_per_batch:
            sample = ds[int(ds_index % len(ds))]
            ds_index += 1

            question = str(sample[cfg.data.question_field])
            true_digits = parse_final_answer_to_digits(sample[cfg.data.answer_field])
            if true_digits is None:
                continue

            traj = _rollout_one(
                phase23=phase23,
                value_head=value_head,
                tokenizer=tokenizer,
                question=question,
                true_digits=true_digits,
                allowed_token_ids=allowed_token_ids,
                max_new_tokens=cfg.rollout.max_new_tokens,
                temperature=cfg.rollout.temperature,
                reward_cfg=cfg.reward,
                reward_rng=reward_rng,
                sample_id=f"u{update}_i{len(trajectories)}",
            )

            if not traj.actions:
                continue
            trajectories.append(traj)
            token_budget += len(traj.actions)
            if token_budget >= int(cfg.rollout.max_tokens_per_batch):
                break

        if not trajectories:
            raise RuntimeError("No trajectories collected for PPO update")

        if cfg.ppo.normalize_advantages:
            _normalize_advantages(trajectories)

        roll_rows: List[Dict[str, object]] = []
        for traj in trajectories:
            row = {
                "id": traj.sample_id,
                "input_ids": traj.prompt_ids,
                "generated_ids": traj.actions,
                "terminated_by": traj.terminated_by,
                "num_generated": len(traj.actions),
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
                "reward_final": traj.reward_info["reward_final"],
                "actions": traj.actions,
                "logp_old": traj.logp_old,
                "entropy": traj.entropy_old,
                "values": traj.values_old,
                "question": traj.question,
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

        for epoch in range(cfg.ppo.ppo_epochs):
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
                            phase23=phase23,
                            value_head=value_head,
                            traj=traj,
                            allowed_idx_t=allowed_idx_t,
                            allowed_id_to_local=allowed_id_to_local,
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
        answer_rate = float(sum(1 for t in trajectories if t.terminated_by == "answer")) / float(len(trajectories))
        avg_len = float(sum(len(t.actions) for t in trajectories)) / float(len(trajectories))

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
                phase23=phase23,
                value_head=value_head,
                tokenizer=tokenizer,
                cfg=cfg,
            )
            _rotate_checkpoints(output_dir=cfg.train.output_dir, keep_last=int(cfg.train.keep_last))


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
