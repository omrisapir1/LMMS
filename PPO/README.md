# Phase-4 PPO: Discrete Z-Token Reinforcement Learning

This phase fine-tunes the **Phase3 SFT discrete Z-token model** using **PPO**.

The model reasons by generating discrete `<z_i>` tokens and terminates with `<ANSWER>`.  
Digit prediction is performed using the **LM head restricted to digit tokens `{0..9}`** (Phase-B decoding).

This stage directly optimizes expected answer reward under sampling and is aligned with `pass@N` evaluation.

---

## High-Level Goal

We want the policy to:

- Generate Z programs that lead to correct 5-digit answers.
- Terminate correctly with `<ANSWER>`.
- Maximize expected reward under sampling.
- Be ready for real `pass@N` style evaluation.

---

## Model Structure

### Policy (Z Generator)

The model autoregressively generates tokens from a **restricted action space**:

```text
Action space = Z_token_ids + [answer_token_id]
```

All other vocabulary logits are masked (e.g., set to `-1e4`).

The policy learns over:

- `<z_0> ... <z_{V-1}>`
- `<ANSWER>`

No natural-language tokens are allowed during PPO.

---

### Digit Decoder (Phase-B)

After rollout terminates:

- If `<ANSWER>` was generated:
  - Predict 5 digit tokens using the LM head
  - Vocabulary restricted to `{0..9}`
- If `<ANSWER>` was not generated:
  - Reward = 0

Digit decoding is controlled by:

```text
digit_greedy: bool
```

If `True`:
- Digits are generated greedily (recommended for stability).

If `False`:
- Digits are sampled (higher reward variance).

---

## Episode / Rollout Definition

Each episode consists of:

1. Input prompt (question only).
2. Autoregressive generation from `{Z tokens, <ANSWER>}`.
3. Stop when:
   - `<ANSWER>` is generated, or
   - `max_new_tokens` is reached.

Reward is computed once per episode.

---

## Reward Function

Let:

- `y`: ground-truth digits `[5]`
- `y_hat`: predicted digits `[5]`
- `keep_prob[pos]`: probability of including digit position if `y[pos] == 0`
- `partial_scale`: scaling factor

---

### Full Reward

If all digits match:

```text
reward = 1.0
```

---

### Partial Reward

We prevent over-rewarding trailing zeros using a stochastic mask.

For each position `pos`:

```text
If y[pos] != 0:
m[pos] = 1
If y[pos] == 0:
m[pos] ~ Bernoulli(keep_prob[pos])
```

Compute:

```text
applied = sum(m)
correct = sum(m[pos] * 1[y_hat[pos] == y[pos]])
```

Partial reward:

```text
if applied == 0:
partial_reward = 0.0
else:
partial_reward = partial_scale * (correct / applied)
```

Final reward:

```text
If full match:
reward = 1.0
Else:
reward = partial_reward
```

---

### Length Penalty (Optional)

To discourage unnecessarily long Z traces:

```text
reward_final = reward - length_penalty * num_generated_tokens
```

If rollout terminated by `max_new_tokens`:

```text
reward = 0.0
```

---

## PPO Components

### Value Head (Critic)

We add a value head on top of hidden states:

```python
self.value_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.Tanh(),
    nn.Linear(hidden_size, 1),
)
```

Per-step baseline:

```text
V_t = value_head(h_t)
```

If rollout length is T:

```text
values: [B, T]
```

### Advantage

We use episodic reward only:

```text
A_t = R - V_t
```

No GAE initially.

### PPO Objective

For each timestep t:

Stored during rollout:

```text
logp_old_t
```

Recomputed during update:

```text
logp_new_t
```

Ratio:

```text
r_t = exp(logp_new_t - logp_old_t)
```

Clipped policy objective:

```text
L_policy = -mean_t(
    min(
        r_t * A_t,
        clip(r_t, 1 - eps, 1 + eps) * A_t
    )
)
```

Value loss:

```text
L_value = mean_t((V_t - R)^2)
```

Entropy regularization:

```text
L_entropy = -mean_t(H(pi(. | s_t)))
```

Total loss:

```text
L = L_policy + c_v * L_value + c_ent * L_entropy
```

### Digit Gradient Mode (Configurable)

We support two update modes controlled by a config flag:

```text
digit_backprop_mode: "ppo_only" | "ppo_plus_supervised"
```

#### Mode A — "ppo_only" (Standard PPO)

Digit logits are recomputed during update.

Reward is treated as a scalar.

No gradient flows through digit prediction.

Gradients flow only through:

- Policy log-probs
- Value head

This is standard PPO and recommended initially.

#### Mode B — "ppo_plus_supervised"

Digit logits are recomputed during update.

In addition to PPO loss, we optionally add:

A supervised digit CE loss.

Gradients flow through digit decoding stage.

This hybrid mode can stabilize learning but mixes RL and supervised signals.

### Masking Requirements

Masking must be identical in:

- Rollout
- PPO update forward pass

Allowed tokens:

```text
allowed_token_ids = z_token_ids + [answer_token_id]
```

All other logits must be masked.

Digit decoding must restrict to:

```text
digit_token_ids = tokens for "0"..."9"
```

### Rollout Logging (JSONL)

Each rollout should log:

#### Top-Level

- id
- question
- generated_z_tokens
- terminated_by ("answer" or "max_new_tokens")
- num_generated

#### Digit Prediction

- digit_logits
- digit_probs
- digit_pred
- digit_true

#### Reward Breakdown

- reward_full
- partial_scale
- keep_prob
- applied_mask
- applied_count
- correct_count
- reward_partial
- length_penalty
- reward_final

#### PPO Per-Step

- actions
- logp_old
- entropy
- values

This allows full debugging and reproducibility.

### Configuration (conf.py)

#### Rollout

- max_new_tokens
- temperature
- top_p
- digit_greedy
- length_penalty

#### Reward

- partial_scale
- keep_prob
- reward_if_max_len = 0.0

#### PPO

- clip_range
- c_v
- c_ent
- lr
- num_epochs
- minibatch_size
- max_grad_norm

#### Digit Backprop Mode

- digit_backprop_mode

## What This Phase Optimizes

This PPO phase trains the model to:

- Produce Z programs that maximize digit correctness.
- Improve expected reward under sampling.
- Align training objective with pass@N.
- Maintain controllable exploration via entropy.

This is the final stage before large-scale evaluation or deployment.
