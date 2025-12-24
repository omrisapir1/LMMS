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
        # log_probs = torch.log_softmax(masked_logits, dim=-1)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        logprob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # [1]

        # Extra debug: compare scaled vs unscaled logprob, mask composition, and temperatures
        try:
            with torch.no_grad():
                # Allowed set info
                allowed = (mask == 0.0)
                allowed_count = int(allowed.sum().item())
                allowed_ids = allowed.nonzero(as_tuple=False).view(-1).tolist()
                # Show up to first 10 allowed ids and tokens
                show_ids = allowed_ids[:10]
                show_toks = []
                if len(show_ids) > 0:
                    try:
                        show_toks = tokenizer.convert_ids_to_tokens(show_ids)
                    except Exception:
                        show_toks = [str(i) for i in show_ids]

                # Alt unscaled logprob
                log_probs_unscaled = torch.log_softmax(masked_logits, dim=-1)
                logprob_unscaled = log_probs_unscaled.gather(-1, action.unsqueeze(-1)).squeeze(-1)

                # Raw/Masked logits for chosen action
                a = int(action.item())
                raw_logit = float(logits_t[0, a].item())
                masked_logit = float(masked_logits[0, a].item())
                scaled_logit = float((logits_for_sampling[0, a]).item())

                # LogSumExp diagnostics
                lse_unscaled = float(torch.logsumexp(masked_logits, dim=-1).item())
                lse_scaled = float(torch.logsumexp(logits_for_sampling, dim=-1).item())

                # Token name
                try:
                    action_tok = tokenizer.convert_ids_to_tokens([a])[0]
                except Exception:
                    action_tok = str(a)

                print("\n[ROLLOUT STEP DEBUG]")
                print(f" phase: {phase}")
                print(f" prefix_len: {input_ids.size(1)}  last_prefix_ids: {input_ids[0, -min(8, input_ids.size(1)):].tolist()}")
                print(f" allowed_count: {allowed_count}  sample_allowed_ids[:10]: {show_ids}  sample_allowed_toks[:10]: {show_toks}")
                print(f" temps: latent_T={latent_temperature} answer_T={answer_temperature}")
                print(f" action_id: {a}  action_tok: {action_tok}  in_allowed: {bool(mask[0, a].item() == 0.0)}")
                print(f" raw_logit: {raw_logit:.6f}  masked_logit: {masked_logit:.6f}  scaled_logit: {scaled_logit:.6f}")
                print(f" logprob_scaled(old): {float(logprob.item()):.9f}  logprob_unscaled_alt: {float(logprob_unscaled.item()):.9f}")
                print(f" logsumexp_unscaled: {lse_unscaled:.6f}  logsumexp_scaled: {lse_scaled:.6f}")

                # Probability mass over allowed set should be ~1.0
                probs_scaled = torch.softmax(logits_for_sampling, dim=-1)
                mass_allowed = float(probs_scaled[0][allowed[0]].sum().item())
                print(f" prob_mass_allowed_scaled: {mass_allowed:.9f}")
        except Exception as e:
            print(f"[ROLLOUT DEBUG ERROR] {e}")

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
    print(" value_t:", float(value_t.item()))
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
