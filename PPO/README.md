# Phase-4 PPO: Discrete Z-Token RL (No Latent Tokens)

This phase fine-tunes the **Phase23 discrete Z-token model** using **PPO**.

Key idea:
- The model reasons by generating **only** `<Z_i>` tokens and then `<ANSWER>`.
- No `<|latent|>` tokens are used in PPO.
- Reward is computed from the model's **digit-head answer prediction** at `<ANSWER>` (or final token if `<ANSWER>` never appears).

---

## What We Train

We start from a Phase23 checkpoint that already supports:
- Autoregressive generation over `{<Z_0>...<Z_{V-1}>, <ANSWER>}`
- Digit prediction via digit heads at `<ANSWER>`

PPO trains:
- A **policy** over Z tokens + `<ANSWER>`
- A **value head** baseline for each generation step

We disable KL-to-reference (optional entropy regularization is used instead).

---

## Episode / Trajectory Definition

Each episode is a single rollout:

1. Input prompt (question only; prompt format TBD).
2. Autoregressively generate tokens from the restricted action space.
   - Allowed actions: `Z tokens` + `<ANSWER>`.
3. Stop when either:
   - `<ANSWER>` is generated, or
   - `max_new_tokens` is reached.

**Terminal condition reward:**
- If stopped by `<ANSWER>`: compute reward normally.
- If stopped by `max_new_tokens`: `reward = 0`.

---

## Action Space Masking

During rollout and PPO updates:
- Only `{Z tokens, <ANSWER>}` are allowed.
- All other vocabulary logits are masked out (for example, set to `-1e4`).

This ensures PPO cannot escape into natural-language tokens.

---

## Reward Function

We compute reward from the **digit prediction** at the end of the rollout.

### Full Reward

If all digits match the ground-truth digits:
- `reward = 1.0`

### Partial Reward (When Not All Digits Correct)

We use a per-digit keep-prob trick to avoid overly rewarding trailing zeros.

Definitions:
- `y`: ground-truth digits `[5]` (MSB-first)
- `y_hat`: predicted digits `[5]`
- `keep_prob[pos]`: probability of including digit `pos` in reward when `y[pos] == 0`

Example:

```python
keep_prob = (0.02, 0.05, 0.1, 0.5, 1.0)
# For positions where label digit == 0, include with keep_prob[pos].
# For label digit != 0, always include.
```

For a given example:

```text
Sample a mask m[pos] in {0, 1}

If y[pos] != 0: m[pos] = 1
If y[pos] == 0: m[pos] ~ Bernoulli(keep_prob[pos])

applied = sum(m)  # number of digits used for reward
correct = sum(m[pos] * 1[y_hat[pos] == y[pos]])
```

Partial reward:

```python
if applied == 0:
    partial_reward = 0.0
else:
    partial_reward = partial_scale * (correct / applied)
```

Final reward:
- If all digits are correct: `reward = 1.0`
- Else: `reward = partial_reward`

### Length Penalty (Optional)

We optionally add a small per-token penalty to discourage unnecessarily long Z traces:

```python
reward_final = reward - length_penalty * num_generated_tokens
```

Set `length_penalty = 0` to disable.

---

## Value Function (Critic)

We add a value head on top of the model's hidden states:

```python
self.value_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.Tanh(),
    nn.Linear(hidden_size, 1),
)
```

### Per-Step Value

PPO requires a baseline at each action step:

```text
V_t = value_head(h_t) for each generation step t
```

Where `h_t` is the policy hidden state used to produce logits for the next token at step `t`.

If the rollout produces `T` tokens, values have shape:

```text
values: [B, T]  # or [T] per sample in logging
```

---

## Advantage (Episodic Reward Only)

We use a single terminal reward `R` per rollout, and compute a simple advantage per step:

```text
A_t = R - V_t
```

No GAE initially.

---

## PPO Objective (No KL)

We use the PPO clipped objective. For each step `t`:
- Store `logp_old_t = log pi_old(a_t | s_t)` during rollout.
- Recompute `logp_new_t = log pi_theta(a_t | s_t)` during update.

Ratio:

```text
r_t = exp(logp_new_t - logp_old_t)
```

Clipped policy objective:

```text
L_policy = -mean_t(min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t))
```

Value loss:

```text
L_value = mean_t((V_t - R)^2)
```

Entropy regularization (recommended since KL is disabled):

```text
L_entropy = -mean_t(H(pi_theta(. | s_t)))
```

Total loss:

```text
L = L_policy + c_v * L_value + c_ent * L_entropy
```

Where:
- `eps`: PPO clip range (for example, `0.2`)
- `c_v`: value loss coefficient
- `c_ent`: entropy coefficient

---

## Why We Must Store `logp_old`

PPO needs `logp_old` (or a frozen old policy) to compute ratio `r_t`.

We use the simplest approach:
- Store per-step `logp_old_t` during rollout.
- No need to store full logits.

---

## Rollout Logging (Saved to Disk)

We log each rollout to disk for debugging and analysis, including:
- Generated Z tokens
- Whether `<ANSWER>` was produced
- Digit prediction probabilities
- Reward breakdown
- Per-step PPO statistics

### Suggested Per-Rollout JSONL Schema

Top-level fields:
- `id`: unique rollout id
- `input_ids`: prompt token ids (or raw question text if preferred)
- `generated_ids`: generated token ids (Z tokens + maybe `<ANSWER>`)
- `terminated_by`: `"answer"` or `"max_new_tokens"`
- `num_generated`: int

Answer/digits:
- `digit_logits`: `[5,10]` (or compressed)
- `digit_probs`: `[5,10]` (recommended for interpretability)
- `digit_pred`: `[5]`
- `digit_true`: `[5]`

Reward:
- `reward_full`: `0/1`
- `partial_scale`: float
- `keep_prob`: `[5]`
- `applied_mask`: `[5]` (sampled mask for this example)
- `applied_count`: int
- `correct_count`: int
- `reward_partial`: float
- `length_penalty`: float
- `reward_final`: float

Per-step PPO:
- `actions`: `[T]` token ids
- `logp_old`: `[T]`
- `entropy`: `[T]` (optional but useful)
- `values`: `[T]`

This log is enough to:
- Recompute learning signals
- Debug collapse
- Analyze length/reward dynamics

---

## Configuration (`conf.py`)

All PPO knobs are controlled from `conf.py`.

Rollout:
- `max_new_tokens`
- `length_penalty`
- `temperature` / sampling settings (if used)

Reward:
- `partial_scale`
- `keep_prob` (length 5)
- `reward_if_max_len = 0.0` (fixed)

PPO:
- `clip_range` (`eps`)
- `c_v` (value loss coefficient)
- `c_ent` (entropy coefficient)
- `lr`
- `num_epochs`
- `minibatch_size`
- `max_grad_norm`

Masking:
- Ensure allowed tokens list is `z_token_ids + [answer_token_id]`.

---

## Implementation Notes / Checklist

Loader:
- Load Phase23 model + tokenizer.
- Ensure action masking is identical in rollout and update.

Rollout:
- Generate tokens step-by-step (or use `generate` + recompute per-step logps).
- Store per-step: actions, `logp_old`, values, optional entropy.

Terminal computation:
- If terminated by max tokens: `reward = 0`.
- Else compute digit reward (full/partial) from digit heads.

Update:
- Recompute `logp_new` and values for the same states/actions.
- Compute advantages: `A_t = R - V_t`.
- Apply PPO clipped objective + value loss + entropy.

Logging:
- Save rollout JSONL records to disk continuously.
- Include digit probabilities and reward breakdown.

---

## What We Intentionally Defer

- Prompt/template finalization
- Evaluation harness
- Model saving/pushing after PPO

These will be added after the first working PPO loop is stable.
