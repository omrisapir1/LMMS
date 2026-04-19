from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from GRPO.conf import Config, DEFAULT_SET_ALLOWED_PREFIXES
from GRPO.rollout import GRPOTrajectory, collect_grpo_batch
from PPO.masking import introspect_z_token_ids_and_style, resolve_answer_token_id
from PPO.token_contract import resolve_digit_token_ids, validate_answer_token_single
from PPO.train import (
    _build_minibatch_order,
    _build_prompt_text,
    _extract_true_digits,
    _load_rsft_trained_questions,
    _question_text,
)
from PPO.vllm_rollout import VLLMRolloutEngine

_RUN_LOG_PATH: Optional[str] = None

try:
    import bitsandbytes as bnb  # type: ignore
except Exception:
    bnb = None



@dataclass
class TrajectoryDeviceCache:
    seq_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_ids: torch.Tensor
    action_phase: torch.Tensor
    state_positions: torch.Tensor
    seq_len: int
    action_len: int
    old_logp: torch.Tensor
    advantages: torch.Tensor


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


def _make_rng(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


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
    p = argparse.ArgumentParser(description="Two-stage GRPO trainer")
    p.add_argument("--set", action="append", default=[], help="Override config, e.g. train.lr=3e-5")
    return p


def _phase_id(action_type: str) -> int:
    t = str(action_type)
    if t in ("z", "answer"):
        return 0
    if t == "digit":
        return 1
    raise RuntimeError(f"Unsupported action type: {action_type!r}")


def _build_trajectory_device_cache(
    *,
    trajectories: Sequence[GRPOTrajectory],
    device: torch.device,
) -> List[TrajectoryDeviceCache]:
    out: List[TrajectoryDeviceCache] = []
    for traj in trajectories:
        prompt_len = int(len(traj.prompt_ids))
        action_len = int(len(traj.actions))
        if action_len != len(traj.action_types):
            raise RuntimeError("actions/action_types length mismatch")
        if action_len != len(traj.old_logp):
            raise RuntimeError("actions/old_logp length mismatch")
        if action_len != len(traj.advantages):
            raise RuntimeError("actions/advantages length mismatch")

        seq_ids = torch.tensor(list(traj.prompt_ids) + list(traj.actions), dtype=torch.long, device=device)
        attention = torch.tensor(
            list(traj.prompt_attention_mask) + [1] * action_len,
            dtype=torch.long,
            device=device,
        )
        if action_len > 0:
            state_positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + action_len,
                dtype=torch.long,
                device=device,
            )
        else:
            state_positions = torch.empty((0,), dtype=torch.long, device=device)

        out.append(
            TrajectoryDeviceCache(
                seq_ids=seq_ids,
                attention_mask=attention,
                action_ids=torch.tensor(traj.actions, dtype=torch.long, device=device),
                action_phase=torch.tensor([_phase_id(t) for t in traj.action_types], dtype=torch.long, device=device),
                state_positions=state_positions,
                seq_len=int(seq_ids.numel()),
                action_len=action_len,
                old_logp=torch.tensor(traj.old_logp, dtype=torch.float32, device=device),
                advantages=torch.tensor(traj.advantages, dtype=torch.float32, device=device),
            )
        )
    return out


def _token_stats_two_phase(
    hidden_states: torch.Tensor,
    action_ids: torch.Tensor,
    action_phase: torch.Tensor,
    z_w: torch.Tensor,
    z_b: Optional[torch.Tensor],
    d_w: torch.Tensor,
    d_b: Optional[torch.Tensor],
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    z_temperature: float,
    d_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_z = z_id_to_local[action_ids]
    local_d = d_id_to_local[action_ids]
    is_z = action_phase == 0
    is_d = action_phase == 1
    invalid = (is_z & (local_z < 0)) | (is_d & (local_d < 0))

    z_logits = (hidden_states @ z_w.t()) / float(z_temperature)
    d_logits = (hidden_states @ d_w.t()) / float(d_temperature)
    if z_b is not None:
        z_logits = z_logits + z_b
    if d_b is not None:
        d_logits = d_logits + d_b

    z_logp = torch.log_softmax(z_logits, dim=-1)
    d_logp = torch.log_softmax(d_logits, dim=-1)
    z_probs = z_logp.exp()
    d_probs = d_logp.exp()

    local_z_safe = local_z.clamp_min(0)
    local_d_safe = local_d.clamp_min(0)
    z_chosen = z_logp.gather(1, local_z_safe.view(-1, 1)).squeeze(1)
    d_chosen = d_logp.gather(1, local_d_safe.view(-1, 1)).squeeze(1)
    z_ent = -(z_probs * z_logp).sum(dim=-1)
    d_ent = -(d_probs * d_logp).sum(dim=-1)

    logp_vec = torch.where(is_z, z_chosen, d_chosen)
    ent_vec = torch.where(is_z, z_ent, d_ent)
    return logp_vec, ent_vec, invalid


def _action_stats_tensors_batched(
    *,
    model,
    trajs: Sequence[GRPOTrajectory],
    traj_cache: Sequence[TrajectoryDeviceCache],
    z_allowed_t: torch.Tensor,
    d_allowed_t: torch.Tensor,
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    z_temperature: float,
    d_temperature: float,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    if not trajs:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty_l

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
        raise RuntimeError("Model output embeddings are unavailable")
    weight = lm_head.weight
    z_w = weight.index_select(0, z_allowed_t)
    d_w = weight.index_select(0, d_allowed_t)
    bias = getattr(lm_head, "bias", None)
    z_b = bias.index_select(0, z_allowed_t) if bias is not None else None
    d_b = bias.index_select(0, d_allowed_t) if bias is not None else None

    lengths_all = torch.tensor([c.action_len for c in cache], dtype=torch.long, device=device)
    nonzero_rows = torch.nonzero(lengths_all > 0, as_tuple=False).squeeze(-1)
    if nonzero_rows.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty_l

    lengths = lengths_all.index_select(0, nonzero_rows)
    hidden_nz = hidden_all.index_select(0, nonzero_rows)

    batch_ids = torch.repeat_interleave(
        torch.arange(nonzero_rows.numel(), dtype=torch.long, device=device),
        lengths,
    )
    state_positions = torch.cat([cache[int(i)].state_positions for i in nonzero_rows.tolist()], dim=0)
    action_ids = torch.cat([cache[int(i)].action_ids for i in nonzero_rows.tolist()], dim=0)
    action_phase = torch.cat([cache[int(i)].action_phase for i in nonzero_rows.tolist()], dim=0)
    old_logp = torch.cat([cache[int(i)].old_logp for i in nonzero_rows.tolist()], dim=0)
    advantages = torch.cat([cache[int(i)].advantages for i in nonzero_rows.tolist()], dim=0)

    h_states = hidden_nz[batch_ids, state_positions]
    logp_new, entropy_new, invalid = _token_stats_two_phase(
        h_states,
        action_ids,
        action_phase,
        z_w,
        z_b,
        d_w,
        d_b,
        z_id_to_local,
        d_id_to_local,
        float(z_temperature),
        float(d_temperature),
    )
    if bool(invalid.any()):
        raise RuntimeError("Found actions not in allowed action vocab for phase")

    return logp_new, entropy_new, old_logp, advantages, lengths


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
        os.path.join(ckpt_dir, "grpo_state.pt"),
    )


def _prepare_dataset_rows(cfg: Config, tokenizer) -> List[Dict[str, object]]:
    ds = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)

    trained_questions: set[str] = set()
    path = str(getattr(cfg.data, "rsft_trained_questions_path", "") or "").strip()
    if path:
        try:
            trained_questions, abs_path = _load_rsft_trained_questions(path)
            _log(f"Loaded RSFT-trained question filter: {len(trained_questions)} from {abs_path}")
        except Exception as exc:
            _log(f"Skipping RSFT question filter due to load error: {type(exc).__name__}: {exc}")

    rows: List[Dict[str, object]] = []
    for i, sample in enumerate(ds):
        q = _question_text(sample.get(cfg.data.question_field))
        if not q:
            continue
        if trained_questions and q in trained_questions:
            continue

        true_digits = _extract_true_digits(
            sample=sample,
            answer_digits_field=cfg.data.answer_digits_field,
            answer_field=cfg.data.answer_field,
        )
        if true_digits is None:
            continue

        prompt_text = _build_prompt_text(tokenizer, q)
        pack = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=True)
        prompt_ids = [int(x) for x in list(pack.get("input_ids", []))]
        prompt_attn = [int(x) for x in list(pack.get("attention_mask", [1] * len(prompt_ids)))]
        if len(prompt_ids) == 0:
            continue

        rows.append(
            {
                "prompt_id": int(i),
                "question": q,
                "true_digits": list(true_digits),
                "prompt_ids": prompt_ids,
                "prompt_attention_mask": prompt_attn,
                "sample_id_base": f"p{i}",
            }
        )

    if len(rows) == 0:
        raise RuntimeError("No valid training rows after filtering/parsing")
    _log(f"Prepared GRPO dataset rows: {len(rows)}")
    return rows


def _assign_old_logp(
    *,
    model,
    trajectories: Sequence[GRPOTrajectory],
    device: torch.device,
    z_allowed_t: torch.Tensor,
    d_allowed_t: torch.Tensor,
    z_id_to_local: torch.Tensor,
    d_id_to_local: torch.Tensor,
    z_temperature: float,
    d_temperature: float,
    pad_token_id: int,
    use_bf16: bool,
) -> None:
    if len(trajectories) == 0:
        return

    cache = _build_trajectory_device_cache(trajectories=trajectories, device=device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (torch.cuda.is_available() and use_bf16)
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        logp_new, _entropy, _old, _adv, lengths = _action_stats_tensors_batched(
            model=model,
            trajs=trajectories,
            traj_cache=cache,
            z_allowed_t=z_allowed_t,
            d_allowed_t=d_allowed_t,
            z_id_to_local=z_id_to_local,
            d_id_to_local=d_id_to_local,
            z_temperature=float(z_temperature),
            d_temperature=float(d_temperature),
            pad_token_id=int(pad_token_id),
        )

    offset = 0
    for traj, L in zip(trajectories, lengths.tolist()):
        ll = int(L)
        traj.old_logp = [float(x) for x in logp_new[offset : offset + ll].detach().cpu().tolist()]
        offset += ll


def train(cfg: Config) -> None:
    if str(cfg.rollout.backend).lower() != "vllm":
        raise ValueError("GRPO currently supports rollout.backend='vllm' only")

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    _set_run_log_path(os.path.join(cfg.train.output_dir, "train.log"))

    _set_seed(cfg.train.seed)
    reward_rng = _make_rng(cfg.train.seed + 17)

    device = torch.device(cfg.rollout.torch_device)
    _log(f"Loading tokenizer/model from {cfg.model.init_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init_ckpt, trust_remote_code=cfg.model.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.init_ckpt,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.to(device)
    model.train()

    answer_token_id = int(resolve_answer_token_id(tokenizer, answer_token=cfg.model.answer_token))
    validate_answer_token_single(tokenizer, cfg.model.answer_token, answer_token_id)
    z_token_ids, z_style = introspect_z_token_ids_and_style(tokenizer)
    if len(z_token_ids) == 0:
        raise RuntimeError("No Z tokens found in tokenizer vocabulary")
    digit_token_ids = resolve_digit_token_ids(tokenizer)

    _log(
        f"Token setup: z_count={len(z_token_ids)} z_style={z_style} "
        f"answer_id={answer_token_id} digit_ids={digit_token_ids}"
    )

    z_allowed = [int(x) for x in z_token_ids] + [int(answer_token_id)]
    d_allowed = [int(x) for x in digit_token_ids]
    z_allowed_t = torch.tensor(z_allowed, dtype=torch.long, device=device)
    d_allowed_t = torch.tensor(d_allowed, dtype=torch.long, device=device)

    max_vocab_id = int(max(max(z_allowed), max(d_allowed)))
    z_id_to_local = torch.full((max_vocab_id + 1,), -1, dtype=torch.long, device=device)
    d_id_to_local = torch.full((max_vocab_id + 1,), -1, dtype=torch.long, device=device)
    z_id_to_local[z_allowed_t] = torch.arange(z_allowed_t.numel(), dtype=torch.long, device=device)
    d_id_to_local[d_allowed_t] = torch.arange(d_allowed_t.numel(), dtype=torch.long, device=device)

    rows = _prepare_dataset_rows(cfg, tokenizer)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    pad_token_id = int(tokenizer.pad_token_id)

    if bnb is not None:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=float(cfg.train.lr),
            betas=tuple(cfg.train.betas),
            eps=float(cfg.train.eps),
            weight_decay=float(cfg.train.weight_decay),
        )
        _log("Optimizer: bitsandbytes AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.train.lr),
            betas=tuple(cfg.train.betas),
            eps=float(cfg.train.eps),
            weight_decay=float(cfg.train.weight_decay),
        )
        _log("Optimizer: torch AdamW (bitsandbytes not available)")

    vllm_kwargs = dict(cfg.rollout.vllm_engine_kwargs)
    if str(cfg.rollout.vllm_cuda_visible_devices).strip():
        vllm_kwargs.setdefault("cuda_visible_devices", str(cfg.rollout.vllm_cuda_visible_devices))
    vllm_kwargs.setdefault("gpu_memory_utilization", float(cfg.rollout.gpu_memory_utilization))
    vllm_kwargs.setdefault("tensor_parallel_size", int(cfg.rollout.vllm_tp_size))

    vllm_engine = VLLMRolloutEngine(
        init_ckpt=cfg.model.init_ckpt,
        tokenizer=tokenizer,
        answer_token_id=answer_token_id,
        z_allowed_token_ids=z_token_ids,
        digit_allowed_token_ids=digit_token_ids,
        trust_remote_code=cfg.model.trust_remote_code,
        engine_kwargs=vllm_kwargs,
        output_dir=cfg.train.output_dir,
        tmp_ckpt_dir=cfg.rollout.vllm_tmp_ckpt_dir,
        sync_every=int(cfg.rollout.vllm_sync_every),
        seed=int(cfg.rollout.vllm_seed if cfg.rollout.vllm_seed is not None else cfg.train.seed),
        logger=_log,
    )

    try:
        for step in range(1, int(cfg.train.updates) + 1):
            t0 = time.perf_counter()
            vllm_engine.maybe_sync_from_torch(model, tokenizer, step)

            k = min(int(cfg.rollout.prompts_per_update), len(rows))
            prepared = random.sample(rows, k=k)
            trajectories, rollout_stats = collect_grpo_batch(
                prepared=prepared,
                tokenizer=tokenizer,
                vllm_engine=vllm_engine,
                cfg=cfg,
                answer_token_id=answer_token_id,
                digit_token_ids=digit_token_ids,
                reward_rng=reward_rng,
            )
            if len(trajectories) == 0:
                raise RuntimeError("No trajectories collected for update")

            _assign_old_logp(
                model=model,
                trajectories=trajectories,
                device=device,
                z_allowed_t=z_allowed_t,
                d_allowed_t=d_allowed_t,
                z_id_to_local=z_id_to_local,
                d_id_to_local=d_id_to_local,
                z_temperature=float(cfg.rollout.z_temperature),
                d_temperature=float(cfg.rollout.digit_temperature),
                pad_token_id=pad_token_id,
                use_bf16=bool(cfg.runtime.use_bf16),
            )

            cache_all = _build_trajectory_device_cache(trajectories=trajectories, device=device)
            seq_lens = [c.seq_len for c in cache_all]

            updates_done = 0
            total_loss = 0.0
            total_pg = 0.0
            total_ent = 0.0

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (torch.cuda.is_available() and bool(cfg.runtime.use_bf16))
                else nullcontext()
            )

            optimizer.zero_grad(set_to_none=True)
            accum = 0
            for _epoch in range(int(cfg.grpo.ppo_epochs)):
                order = _build_minibatch_order(
                    seq_lens=seq_lens,
                    use_length_bucketing=bool(cfg.runtime.use_length_bucketing),
                    bucket_width=int(cfg.runtime.length_bucket_width),
                )

                mb = max(1, int(cfg.grpo.minibatch_size))
                for start in range(0, len(order), mb):
                    idxs = order[start : start + mb]
                    traj_mb = [trajectories[i] for i in idxs]
                    cache_mb = [cache_all[i] for i in idxs]

                    with amp_ctx:
                        logp_new, entropy, logp_old, adv, _lengths = _action_stats_tensors_batched(
                            model=model,
                            trajs=traj_mb,
                            traj_cache=cache_mb,
                            z_allowed_t=z_allowed_t,
                            d_allowed_t=d_allowed_t,
                            z_id_to_local=z_id_to_local,
                            d_id_to_local=d_id_to_local,
                            z_temperature=float(cfg.rollout.z_temperature),
                            d_temperature=float(cfg.rollout.digit_temperature),
                            pad_token_id=pad_token_id,
                        )

                        ratio = torch.exp(logp_new - logp_old)
                        ratio_clip = ratio.clamp(
                            1.0 - float(cfg.grpo.clip_range),
                            1.0 + float(cfg.grpo.clip_range),
                        )
                        s1 = ratio * adv
                        s2 = ratio_clip * adv
                        pg_obj = torch.minimum(s1, s2).mean()
                        ent_mean = entropy.mean()
                        loss = -(pg_obj + float(cfg.grpo.c_ent) * ent_mean)
                        scaled_loss = loss / float(max(1, int(cfg.train.grad_accum_steps)))

                    scaled_loss.backward()
                    accum += 1
                    total_loss += float(loss.detach().item())
                    total_pg += float(pg_obj.detach().item())
                    total_ent += float(ent_mean.detach().item())
                    updates_done += 1

                    if accum >= int(max(1, cfg.train.grad_accum_steps)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grpo.max_grad_norm))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0

            if accum > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grpo.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            rewards = [float(t.reward_info.get("reward_final", 0.0)) for t in trajectories if t.reward_info.get("phase") == "digit"]
            z_rewards = [float(t.reward_info.get("z_reward_mean_children", 0.0)) for t in trajectories if t.reward_info.get("phase") == "z"]
            adv_abs = [abs(float(a)) for t in trajectories for a in t.advantages]
            dt = time.perf_counter() - t0

            _log(
                " | ".join(
                    [
                        f"step={step}",
                        f"traj={len(trajectories)}",
                        f"digit_reward_mean={(sum(rewards)/len(rewards)) if rewards else 0.0:.4f}",
                        f"z_reward_mean={(sum(z_rewards)/len(z_rewards)) if z_rewards else 0.0:.4f}",
                        f"digit_exact_rate={rollout_stats.get('digit_exact_rate', 0.0):.4f}",
                        f"adv_abs_mean={(sum(adv_abs)/len(adv_abs)) if adv_abs else 0.0:.4f}",
                        f"loss={(total_loss/max(1, updates_done)):.4f}",
                        f"pg={(total_pg/max(1, updates_done)):.4f}",
                        f"ent={(total_ent/max(1, updates_done)):.4f}",
                        f"sec={dt:.2f}",
                    ]
                )
            )

            if int(cfg.train.save_every) > 0 and (step % int(cfg.train.save_every) == 0):
                _save_checkpoint(
                    output_dir=cfg.train.output_dir,
                    step=step,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    cfg=cfg,
                )

    finally:
        try:
            vllm_engine.close()
        except Exception:
            pass


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = Config()
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"Invalid --set format: {kv!r}")
        key, raw = kv.split("=", 1)
        _apply_override(cfg, key.strip(), raw.strip())
    train(cfg)


if __name__ == "__main__":
    main()
