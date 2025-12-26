from typing import Any, Dict
import random

import torch
from torch.optim import AdamW

from env.math_env import MathEnv
from ppo.buffer import TrajectoryBuffer
from ppo.advantage import compute_advantages
from ppo.loss import compute_ppo_losses
from LMMS_logging.episode_logger import log_episode


# Minibatch iterator for PPO updates
# - Tensors are indexed with a LongTensor of indices
# - Python lists (e.g., phases, action_kinds, allowed_action_ids) are reassembled by element-wise selection
# - Advantages must already be attached to the batch for consistent slicing

def iterate_minibatches(batch: Dict[str, Any], minibatch_size: int, shuffle: bool = True):
    N = batch["actions"].shape[0]
    if shuffle:
        indices = torch.randperm(N)
    else:
        indices = torch.arange(N)

    for start in range(0, N, minibatch_size):
        idx = indices[start:start + minibatch_size]
        idx_list = idx.tolist()
        yield {
            k: (v[idx] if torch.is_tensor(v) else [v[i] for i in idx_list])
            for k, v in batch.items()
        }


class PPOTrainer:
    def __init__(self, cfg: Dict[str, Any], policy, tokenizer, device: torch.device, run_id: str = "run"):
        self.cfg = cfg
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.run_id = run_id
        self.global_step = 0
        self.batch_idx = 0

        # Environment for rollouts (Phase-1)
        self.env = MathEnv(cfg, tokenizer, debug=False)

        # Optimizer settings
        lr_policy = float(cfg["ppo"]["learning_rate"]["policy"])
        lr_value = float(cfg["ppo"]["learning_rate"]["value"])
        weight_decay = float(cfg["ppo"]["optimizer"]["weight_decay"]) if "optimizer" in cfg["ppo"] else 0.0
        self.max_grad_norm = float(cfg["ppo"]["optimizer"]["max_grad_norm"]) if "optimizer" in cfg["ppo"] else 1.0
        self.entropy_coef = float(cfg["ppo"]["entropy_coefficient"]) if "entropy_coefficient" in cfg["ppo"] else 0.0

        # Parameter groups (cached once)
        self._policy_params_cached = list(
            p for n, p in self.policy.named_parameters()
            if n.startswith("lm.") or n.startswith("answer_head.")
        )
        self._value_params_cached = list(self.policy.value_head.parameters())

        if len(self._policy_params_cached) == 0:
            raise RuntimeError("No trainable parameters found for policy optimizer.")
        if len(self._value_params_cached) == 0:
            raise RuntimeError("No parameters found for value optimizer (value_head).")

        self.policy_optimizer = AdamW(self._policy_params_cached, lr=lr_policy, weight_decay=weight_decay)
        self.value_optimizer = AdamW(self._value_params_cached, lr=lr_value, weight_decay=weight_decay)

        # Training schedule
        self.rollout_batch_size = int(cfg["training"]["rollout_batch_size"])
        self.updates_per_epoch = int(cfg["training"]["updates_per_epoch"])
        self.clip_epsilon = float(cfg["ppo"]["clip_epsilon"])  # PPO clipping
        # New: minibatch size for PPO updates
        self.ppo_minibatch_size = int(cfg["training"]["mini_batch_rollout_batch_size"])

    def _sample_datapoint(self, dataset) -> Dict[str, Any]:
        idx = random.randint(0, len(dataset) - 1)
        ex = dataset[idx]
        # Expected keys in filtered dataset
        question = ex.get("question") if isinstance(ex, dict) else ex["question"]
        if question is None:
            # fallback
            question = ex.get("question_text") if isinstance(ex, dict) else ex["question_text"]

        label = {
            "label_int": int(ex.get("final_ans") if isinstance(ex, dict) else ex["final_ans"]),
            "length_ans": int(ex.get("length_ans") if isinstance(ex, dict) else ex["length_ans"]),
        }

        return {"question": question, "label": label}

    def _policy_forward_on_batch(self, batch: Dict[str, torch.Tensor]):
        """Re-evaluate policy on the exact prefixes used for each PPO step.

        Returns:
            logits_token_new: [N, V] last-token logits
            logits_answer_new: [N, 10] classification head logits
            values_new: [N] value estimates computed from detached hidden states

        Notes:
            - Both heads (token LM logits and answer_head logits) are always computed for all steps.
            - Mixed-action masking/usage occurs later inside compute_ppo_losses; no phase-conditional logic here.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward through backbone to get logits and hidden states
        outputs = self.policy.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits_full = outputs.logits  # [N, T, V]
        logits_token_new = logits_full[:, -1, :]  # [N, V]

        # Last hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [N, H]

        # NOTE:
        # answer_head is intentionally computed for all steps (including token-only steps).
        # Masking and usage are handled later in the PPO loss.
        # This avoids phase-conditional branching in the forward pass.
        logits_answer_new = self.policy.answer_head(last_hidden)  # [N, 10]
        values_new = self.policy.value_head(last_hidden).squeeze(-1)  # [N]

        return logits_token_new, logits_answer_new, values_new

    def train_epoch(self, dataset) -> Dict[str, float]:
        buffer = TrajectoryBuffer()
        episodes_collected = 0
        for _ in range(self.rollout_batch_size):
            dp = self._sample_datapoint(dataset)
            episode = self._collect_single(dp["question"], dp["label"])  # helper below
            buffer.add_episode(episode)
            episodes_collected += 1
        if len(buffer) == 0:
            raise RuntimeError("No steps in buffer after rollout collection.")

        # Batch-based episode logging (before PPO updates)
        log_cfg = self.cfg.get("logging", {})
        if log_cfg.get("log_episodes", False):
            stride = int(log_cfg.get("log_every_n_in_batch"))
            for i in range(0, len(buffer._episodes), stride):
                ep = buffer._episodes[i]
                # Reconstruct full sequence step-by-step from first prefix
                full_ids: list[int] = []
                if ep["input_ids_steps"]:
                    full_ids.extend(ep["input_ids_steps"][0])
                for action, kind, inserted in zip(
                        ep["actions"],
                        ep["action_kinds"],
                        ep["inserted_token_ids_steps"],
                ):
                    if kind == "token":
                        full_ids.append(action)
                        if inserted:
                            full_ids.extend(inserted)
                full_tokens = self.tokenizer.convert_ids_to_tokens(full_ids)

                # Extract predicted integer from answer action
                pred_int = None
                for action, kind in zip(ep["actions"], ep["action_kinds"]):
                    if kind == "answer":
                        pred_int = int(action)
                        break
                if pred_int is None:
                    pred_int = 0

                question = ep.get("question", "")
                K = int(ep.get("K", 0))
                label_int = int(ep.get("label_int", 0))
                reward = float(ep.get("reward", 0.0))
                log_episode(
                    run_id=self.run_id,
                    global_step=self.global_step,
                    batch_idx=self.batch_idx,
                    episode_idx=i,
                    question=question,
                    K=K,
                    full_sequence_tokens=full_tokens,
                    pred_int=pred_int,
                    label_int=label_int,
                    reward=reward,
                )

        # 2) Build PPO batch (includes padded input_ids/attention_mask)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        # Store the entire PPO batch on CPU to reduce GPU memory usage
        batch = buffer.get_batch(torch.device("cpu"), pad_token_id=pad_id)

        # Optional safety assertion: exactly one answer action per episode
        try:
            assert batch["action_kinds"].count("answer") == len(buffer._episodes)
        except Exception:
            raise RuntimeError(
                f"Protocol violation: expected one 'answer' action per episode; got {batch['action_kinds'].count('answer')} answers for {len(buffer._episodes)} episodes"
            )

        # 3) Compute advantages (computed once per batch in Phase-1)
        advantages = compute_advantages(batch, normalize=True)
        if advantages.shape != batch["rewards"].shape:
            raise RuntimeError("Advantages shape mismatch vs rewards.")
        if not torch.isfinite(advantages).all():
            raise RuntimeError("Advantages contain non-finite values.")
        # Attach to batch for deterministic minibatch slicing
        batch["advantages"] = advantages

        # 4) PPO updates (minibatched)
        last_metrics: Dict[str, float] = {}
        for _ in range(self.updates_per_epoch):
            for mb in iterate_minibatches(batch, self.ppo_minibatch_size, shuffle=True):
                # Move only minibatch tensors to GPU; keep lists on CPU
                mb = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in mb.items()}

                logits_token_new, logits_answer_new, values_new = self._policy_forward_on_batch(mb)

                # Compute losses with mixed action spaces
                losses = compute_ppo_losses(
                    logits_token_new=logits_token_new,
                    logits_answer_new=logits_answer_new,
                    values_new=values_new,
                    batch=mb,
                    advantages=mb["advantages"],
                    clip_epsilon=self.clip_epsilon,
                )

                policy_loss = losses["policy_loss"]
                value_loss = losses["value_loss"]
                entropy_loss = losses["entropy_loss"]
                ratio = losses["ratio"]

                # Safety checks
                for name, t in {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy_loss": entropy_loss,
                    "ratio": ratio,
                }.items():
                    if not torch.isfinite(t).all():
                        raise RuntimeError(f"Non-finite tensor detected in {name}.")

                # Policy backward pass
                self.policy_optimizer.zero_grad(set_to_none=True)
                (policy_loss + self.entropy_coef * entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self._policy_params_cached, self.max_grad_norm
                )
                self.policy_optimizer.step()

                # Value backward pass
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._value_params_cached, self.max_grad_norm
                )
                self.value_optimizer.step()

                # Record/aggregate metrics (use last minibatch/update for now)
                last_metrics = {
                    "mean_reward": float(batch["rewards"].mean().item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy_loss": float(entropy_loss.item()),
                    "mean_advantage": float(advantages.mean().item()),
                    "ratio_mean": float(ratio.mean().item()),
                    "ratio_std": float(ratio.std().item()),
                    "episodes_collected": float(episodes_collected),
                    "total_steps": float(len(buffer)),
                }

        # 5) Cleanup
        buffer.clear()
        self.global_step += 1
        self.batch_idx += 1

        total_norm = torch.norm(
            torch.stack([
                p.grad.norm()
                for p in self._policy_params_cached
                if p.grad is not None
            ])
        )
        print(" policy_grad_norm:", total_norm.item())

        return last_metrics

    def _collect_single(self, question: str, label: Dict[str, Any]) -> Dict[str, Any]:
        # Use policy/env/tokenizer to collect one episode
        from ppo.rollout import collect_rollout
        episode = collect_rollout(
            policy=self.policy,
            env=self.env,
            tokenizer=self.tokenizer,
            device=self.device,
            question=question,
            label=label,
        )
        # Attach metadata for logging
        episode["question"] = question
        episode["label_int"] = int(label["label_int"]) if "label_int" in label else 0
        # Remove length_ans assumption from logging; keep if needed for external refs
        episode["length_ans"] = int(label.get("length_ans", 0))
        return episode
