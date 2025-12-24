from typing import Any, Dict, List, Tuple

import torch

from env.math_env import MathEnv
from policy.policy_model import PolicyModel


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
    - Answer digits: greedy argmax (via config)
    """
    # Read decoding strategies and temperature from config
    rollout_cfg = env.cfg["rollout"]
    latent_cfg = rollout_cfg["latent"]
    answer_cfg = rollout_cfg["answer"]
    latent_strategy = str(latent_cfg["strategy"])  # expected "sample"
    latent_temperature = float(latent_cfg["temperature"])  # e.g., 1.0
    answer_strategy = str(answer_cfg["strategy"])  # expected "sample"
    answer_temperature = float(answer_cfg["temperature"])  # e.g., 1.0

    # Reset environment
    env.reset(question, label)
    st = env._require_state()

    # Build initial sequence: encode question with special tokens
    question_ids = tokenizer.encode(question, add_special_tokens=True)
    input_ids = torch.tensor(question_ids, dtype=torch.long, device=device).view(1, -1)
    attention_mask = torch.ones_like(input_ids)

    actions: List[int] = []
    logprobs: List[float] = []
    values: List[float] = []
    entropies: List[float] = []
    phases: List[str] = []
    # New: store per-step sequences (prefix that produced the action)
    input_ids_steps: List[List[int]] = []
    attention_mask_steps: List[List[int]] = []
    inserted_token_ids_steps: List[List[int]] = []

    z_count = 0
    digit_count = 0

    while not env.is_done():
        # a) Action space
        space = env.get_action_space()

        # b) Forward policy under no_grad (rollout should not build graphs)
        with torch.no_grad():
            outputs = policy(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, T, V]
            values_tensor = outputs.values  # [1, T]
            logits_t = logits[:, -1, :]  # [1, V]
            value_t = values_tensor[:, -1]  # [1]

        # Safety: must not mask after termination
        assert not env.is_done(), "Attempted to sample action after episode termination."

        # c) Apply action mask
        mask = space.logit_mask(vocab_size=logits_t.size(-1), device=logits_t.device, dtype=logits_t.dtype)
        masked_logits = logits_t + mask  # [1, V]

        # d) Select action per phase using config-driven strategies
        phase = env._require_state().phase
        if phase == "latent":
            if latent_strategy != "sample":
                raise ValueError(f"Unsupported latent decoding strategy: {latent_strategy}")
            # Apply temperature for sampling (decoding-only; policy unchanged)
            logits_for_sampling = masked_logits / latent_temperature
            dist = torch.distributions.Categorical(logits=logits_for_sampling)
            action = dist.sample()  # [1]
            entropy = dist.entropy()  # [1]
        elif phase == "answer":
            if answer_strategy == "sample":
                logits_for_sampling = masked_logits / answer_temperature
                dist = torch.distributions.Categorical(logits=logits_for_sampling)
                action = dist.sample()  # [1]

                entropy = torch.zeros_like(value_t)  # we dont want entropy loss for answer phase
            else:
                raise ValueError(f"Unsupported answer decoding strategy: {answer_strategy}")
        else:
            raise RuntimeError(f"Unknown env phase '{phase}' during rollout.")

        # e) Logprob (even for greedy) from unscaled masked logits
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        logprob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # [1]

        # f) Record step
        actions.append(int(action.item()))
        logprobs.append(float(logprob.item()))
        values.append(float(value_t.item()))
        entropies.append(float(entropy.item()))
        phases.append(phase)
        # Record the exact prefix used for this action
        input_ids_steps.append(input_ids.squeeze(0).tolist())
        attention_mask_steps.append(attention_mask.squeeze(0).tolist())

        # g) Step env
        env.step(int(action.item()))

        # h) Append chosen action token FIRST
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
        elif phase == "answer":
            digit_count += 1


        print("[ROLLOUT]")

        print(" phase:", phase)
        print(" action_id:", int(action.item()))
        print(" action_tok:", tokenizer.convert_ids_to_tokens([int(action.item())])[0])
        print(" logprob_old:", float(logprob.item()))
        print(" masked_logits max/min:", float(masked_logits.max().item()), float(masked_logits.min().item()))
        # also show how many tokens are allowed by mask
        allowed = torch.isfinite(mask).sum().item()  # mask is 0 or -inf usually
        print(" allowed_count:", int(allowed))
        print("-----")

    # Episode end
    reward = env.get_reward()

    # Required checks
    st = env._require_state()
    if z_count != st.K:
        raise RuntimeError(f"Recorded z-actions {z_count} != K {st.K}")
    if digit_count != st.length_ans:
        raise RuntimeError(f"Recorded digit actions {digit_count} != length_ans {st.length_ans}")
    if not (len(actions) == len(logprobs) == len(values) == len(entropies) == len(phases) == len(input_ids_steps) == len(attention_mask_steps) == len(inserted_token_ids_steps)):
        raise RuntimeError("Step list lengths mismatch.")

    return {
        "K": st.K,
        "length_ans": st.length_ans,
        "actions": actions,
        "logprob_old": logprobs,
        "value_old": values,
        "entropy": entropies,
        "phases": phases,
        "reward": float(reward),
        "z_count": z_count,
        "digit_count": digit_count,
        # sequences
        "input_ids_steps": input_ids_steps,
        "attention_mask_steps": attention_mask_steps,
        "inserted_token_ids_steps": inserted_token_ids_steps,
    }
