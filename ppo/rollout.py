from typing import Any, Dict, List, Tuple

import torch

from env.math_env import MathEnv
from policy.policy_model import PolicyModel

SEPERATOR = "<|im_end|>\n<|im_start|>assistant\n"

def _append_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, token_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(token_ids, list):
        token_ids = [token_ids]
    add = torch.tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device).view(1, -1)
    input_ids = torch.cat([input_ids, add], dim=1)
    att = torch.ones_like(add)
    attention_mask = torch.cat([attention_mask, att], dim=1)
    return input_ids, attention_mask


def collect_rollout(
    policy: PolicyModel,
    env: MathEnv,
    tokenizer,
    device: torch.device,
    question: str,
    label: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collect one full Phase-1 episode rollout.

    Phase-1 decoding:
    - Latent <z*> actions: sampled (temperature-controlled via config)
    - Answer digits: sampled from classification head (0-9)
    """
    # Read decoding strategies and temperature from config
    rollout_cfg = env.cfg["rollout"]
    latent_cfg = rollout_cfg["latent"]
    answer_cfg = rollout_cfg["answer"]
    latent_strategy = str(latent_cfg["strategy"])  # expected "sample"
    latent_temperature = float(latent_cfg["temperature"])  # e.g., 1.0
    answer_strategy = str(answer_cfg["strategy"])  # expected "sample"
    answer_temperature = float(answer_cfg["temperature"])  # e.g., 1.0
    # Debug flag
    debug_ppo = bool(env.cfg.get("debug", {}).get("ppo", True))

    # Reset environment
    env.reset(question, label)
    st = env._require_state()

    # Build initial sequence: encode question with special tokens
    question_ids = tokenizer.encode(question + SEPERATOR, add_special_tokens=True)
    input_ids = torch.tensor(question_ids, dtype=torch.long, device=device).view(1, -1)
    attention_mask = torch.ones_like(input_ids)

    actions: List[int] = []
    logprobs: List[float] = []
    values: List[float] = []
    entropies: List[float] = []
    phases: List[str] = []
    action_kinds: List[str] = []
    # sequences
    input_ids_steps: List[List[int]] = []
    attention_mask_steps: List[List[int]] = []
    inserted_token_ids_steps: List[List[int]] = []
    # Record allowed ids per step for exact mask replication in loss
    allowed_action_ids_steps: List[List[int]] = []

    z_count = 0
    answer_count = 0

    # Debug: per-step rollout logs
    debug_rollout: List[Dict[str, Any]] = []

    while not env.is_done():
        # a) Action space
        space = env.get_action_space()
        # b) Forward policy under no_grad (rollout should not build graphs)
        with torch.no_grad():
            outputs = policy(input_ids=input_ids, attention_mask=attention_mask)

        # Safety: must not mask after termination
        assert not env.is_done(), "Attempted to sample action after episode termination."

        phase = env._require_state().phase

        # Mandatory runtime safety checks for phase-kind routing
        if space.kind == "token" and phase == "answer":
            raise RuntimeError("Token action space used during answer phase")
        if space.kind == "answer" and phase != "answer":
            raise RuntimeError("Answer action space used outside answer phase")

        # c) Route logits and apply appropriate mask
        if space.kind == "token":
            # Use LM logits at last position
            logits = outputs.logits  # [1, T, V]
            values_tensor = outputs.values  # [1, T]
            logits_t = logits[:, -1, :]  # [1, V]
            value_t = values_tensor[:, -1]  # [1]

            # Record allowed ids for this step (vocab token ids)
            allowed_action_ids_steps.append(list(space.allowed_ids))

            # Apply token mask
            mask = space.logit_mask(vocab_size=logits_t.size(-1), device=logits_t.device, dtype=logits_t.dtype)
            masked_logits = logits_t + mask  # [1, V]

            # e) Logprob from unscaled masked token logits
            log_probs_untempered = torch.log_softmax(masked_logits, dim=-1)
            # Also compute tempered for sanity
            log_probs_tempered = torch.log_softmax(masked_logits / latent_temperature, dim=-1)
            # Select action using TEMPERED distribution (sampling-time)
            if phase == "latent":
                if latent_strategy != "sample":
                    raise ValueError(f"Unsupported latent decoding strategy: {latent_strategy}")
                # Apply temperature for sampling (decoding-only; policy unchanged)
                logits_for_sampling = masked_logits / latent_temperature
                dist = torch.distributions.Categorical(logits=masked_logits / latent_temperature)
                action = dist.sample()  # [1]
                entropy = dist.entropy()  # [1]
            elif phase == "answer":
                # Strict protocol: tokens must not be used during answer phase
                raise RuntimeError("Token action space used during answer phase")
            else:
                raise RuntimeError(f"Unknown env phase '{phase}' during rollout.")

            # Rollout logprob_old MUST use UN-TEMPERED logits
            logprob_untempered = log_probs_untempered.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob_tempered = log_probs_tempered.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob = logprob_untempered
            if debug_ppo:

                debug_rollout.append({
                    "step_idx": len(actions),
                    "phase": phase,
                    "kind": space.kind,
                    "latent_temperature": latent_temperature,
                    "masked_logits_min": float(masked_logits.min().item()),
                    "masked_logits_max": float(masked_logits.max().item()),
                    "logprob_untempered": float(logprob_untempered.item()),
                    "logprob_tempered": float(logprob_tempered.item()),
                    "logprob_old": float(logprob.item()),
                    "action": int(action.item()),
                    "allowed_ids_count": int(len(space.allowed_ids)),
                })

            # f) Record step
            actions.append(int(action.item()))
            logprobs.append(float(logprob.item()))
            values.append(float(value_t.item()))
            entropies.append(float(entropy.item()))
            phases.append(phase)

            action_kinds.append("token")
            # Record the exact prefix used for this action
            input_ids_steps.append(input_ids.squeeze(0).tolist())
            attention_mask_steps.append(attention_mask.squeeze(0).tolist())

            # g) Step env with the sampled token id
            env.step(int(action.item()))

            # h) Append sampled token id to input_ids
            input_ids, attention_mask = _append_tokens(input_ids, attention_mask, [int(action.item())])

            # i) Append environment-inserted scaffold tokens AFTER
            inserted = env.get_inserted_token_ids()
            if inserted:
                input_ids, attention_mask = _append_tokens(input_ids, attention_mask, inserted)
                inserted_token_ids_steps.append(list(inserted))
            else:
                inserted_token_ids_steps.append([])

            # Track counts
            if phase == "latent":
                z_count += 1

        elif space.kind == "answer":
            # Use classification head logits
            if not hasattr(outputs, "answer_logits"):
                raise RuntimeError("PolicyModel missing answer_logits for classification phase")
            answer_logits = outputs.answer_logits  # [1, 10]
            if answer_logits.ndim == 3:
                # [B, T, 10] -> take last token
                answer_logits = answer_logits[:, -1, :]
            elif answer_logits.ndim != 2:
                raise RuntimeError(f"Unexpected answer_logits shape: {answer_logits.shape}")
            values_tensor = outputs.values  # [1, T]
            value_t = values_tensor[:, -1]  # [1]

            # Record allowed ids for this step (class indices)
            allowed_action_ids_steps.append(list(space.allowed_ids))

            # Apply class mask (10 classes)
            num_classes = answer_logits.size(-1)
            if num_classes != 10:
                raise RuntimeError(f"Answer head size {num_classes} != expected 10")
            mask = space.class_mask(num_classes=num_classes, device=answer_logits.device, dtype=answer_logits.dtype)
            masked_logits = answer_logits + mask  # [1, 10]

            # e) Logprob from masked classification logits
            log_probs_untempered = torch.log_softmax(masked_logits, dim=-1)
            log_probs_tempered = torch.log_softmax(masked_logits / answer_temperature, dim=-1)
            # Select classification action using TEMPERED distribution
            if phase != "answer":
                raise RuntimeError("Answer action space used outside answer phase")
            if answer_strategy != "sample":
                raise ValueError(f"Unsupported answer decoding strategy: {answer_strategy}")
            logits_for_sampling = masked_logits / answer_temperature
            dist = torch.distributions.Categorical(logits=logits_for_sampling)
            action = dist.sample()  # [1]

            # No entropy bonus by default for answer phase
            entropy = torch.zeros_like(value_t)

            # Rollout logprob_old MUST use UN-TEMPERED logits
            logprob_untempered = log_probs_untempered.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob_tempered = log_probs_tempered.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            logprob = logprob_untempered
            if debug_ppo:
                debug_rollout.append({
                    "step_idx": len(actions),
                    "phase": phase,
                    "kind": space.kind,
                    "answer_temperature": answer_temperature,
                    "masked_logits_min": float(masked_logits.min().item()),
                    "masked_logits_max": float(masked_logits.max().item()),
                    "logprob_untempered": float(logprob_untempered.item()),
                    "logprob_tempered": float(logprob_tempered.item()),
                    "logprob_old": float(logprob.item()),
                    "action": int(action.item()),
                    "allowed_ids_count": int(len(space.allowed_ids)),
                })

            # f) Record step
            act_int = int(action.item())
            if act_int not in range(10):
                raise RuntimeError("Invalid answer class selected")
            actions.append(act_int)
            logprobs.append(float(logprob.item()))
            values.append(float(value_t.item()))
            entropies.append(float(entropy.item()))
            phases.append(phase)
            action_kinds.append("answer")
            # Record the exact prefix used for this action
            input_ids_steps.append(input_ids.squeeze(0).tolist())
            attention_mask_steps.append(attention_mask.squeeze(0).tolist())

            # g) Step env with class index; DO NOT touch text sequence
            env.step(act_int)

            # h) Do NOT append any tokens to the input sequence for classification answers
            inserted_token_ids_steps.append([])

            # Track counts
            answer_count += 1

        else:
            raise RuntimeError(f"Unknown ActionSpace kind '{space.kind}'")

    # Episode end
    reward = env.get_reward()

    # Required checks
    st = env._require_state()
    if z_count != st.K:
        raise RuntimeError(f"Recorded z-actions {z_count} != K {st.K}")
    if answer_count != 1:
        raise RuntimeError("Expected exactly one answer classification action.")
    if not (len(actions) == len(logprobs) == len(values) == len(entropies) == len(phases) == len(input_ids_steps) == len(attention_mask_steps) == len(inserted_token_ids_steps)):
        raise RuntimeError("Step list lengths mismatch.")

    # Validate answer range if present
    if answer_count == 1:
        ans_idx = None
        for i, ph in enumerate(phases):
            if ph == "answer":
                ans_idx = i
                break
        if ans_idx is None:
            raise RuntimeError("No answer phase recorded despite answer_count=1")
        if actions[ans_idx] not in range(10):
            raise RuntimeError("Final answer action out of range [0..9]")

    if len(action_kinds) != len(actions):
        raise RuntimeError("action_kinds length mismatch with actions")


    return {
        "K": st.K,
        "length_ans": st.length_ans,
        "actions": actions,
        "logprob_old": logprobs,
        "value_old": values,
        "entropy": entropies,
        "phases": phases,
        "action_kinds": action_kinds,
        "reward": float(reward),
        "z_count": z_count,
        "answer_count": answer_count,
        # sequences
        "input_ids_steps": input_ids_steps,
        "attention_mask_steps": attention_mask_steps,
        "inserted_token_ids_steps": inserted_token_ids_steps,
        # include per-step allowed ids (token ids for latent; class indices for answer)
        "allowed_action_ids_steps": allowed_action_ids_steps,
        # Debug
        "debug_rollout": debug_rollout,
    }
