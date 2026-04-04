# Phase 4 — RSFT (Rejection Sampling Fine-Tuning)

## Overview

This phase trains the model using **rejection sampling + supervised fine-tuning**.

For each prompt:
1. generate multiple candidate trajectories
2. keep only **fully correct digit matches**
3. select the **shortest valid trajectory**
4. train using cross-entropy on the selected trajectory

---

## System Architecture

Training uses **two GPUs**:

### 1. Training GPU (Transformers)
- holds trainable model
- performs forward + backward
- applies CE updates

### 2. Rollout GPU (vLLM)
- holds a copy of the model
- generates rollouts
- runs batched sampling

### Weight Sync
- training model → vLLM model
- frequency controlled by config

---

## Training Loop

Each step:

### 1. Sample prompts

### 2. Generate rollouts
For each prompt:
- generate `rollouts_per_prompt` candidates

Generation:
- Z tokens: sampled (temperature + top_p)
- digits: greedy

### 3. Evaluate rollouts
For each rollout:
- compute final digit exact match

### 4. Filter
For each prompt:
- keep only rollouts with exact match
- if none → discard prompt

### 5. Select
From valid rollouts:
- select the one with **shortest Z length**

### 6. Build training batch
Each accepted example contains:
- prompt
- Z tokens
- `<ANSWER>`
- 5 digits

Continue sampling until reaching `train_batch_size`.

### 7. Train
Run CE on accepted batch using masked losses (defined below).

---

## Loss

Two losses are computed:

### 1. Z + `<ANSWER>` loss

- applies only on:
  - Z tokens
  - `<ANSWER>` token
- all other tokens are masked out

```
L_z_ans = CE(logits, targets, mask = is_z_or_answer)
```

---

### 2. Digit loss

- applies only on:
  - digit tokens (0–9)
- all other tokens are masked out

```
L_digits = CE(logits, targets, mask = is_digit)
```

---

### Final loss

```
L = w_z_ans * L_z_ans + w_digits * L_digits
```

Recommended defaults:

```
w_z_ans = 1.0
w_digits = 1.0
```

---

## Configuration

### Rollout

```
vllm_batch_size: int
rollouts_per_prompt: int
max_new_tokens: int
temperature: float
top_p: float
digit_greedy: bool = True
```

---

### Training

```
train_batch_size: int
grad_accum_steps: int
lr: float
weight_decay: float
betas: Tuple[float, float]
```

---

### Evaluation

```
eval_every_steps: int
pass_at_n: int
```

---

## Batch Construction

Each step builds a batch of **accepted examples**.

Process:
- sample prompts
- generate rollouts
- filter + select
- accumulate accepted examples

Stop when:

```
len(accepted_examples) >= train_batch_size
```

---

## Evaluation

Run every `eval_every_steps`.

Metrics:
- greedy exact match
- pass@N
- mean Z length
- no_answer_before_kmax_rate

---

## Logging

Each step log:

### Rollout
- prompts sampled
- total rollouts
- accepted rollouts
- acceptance rate
- mean accepted Z length

### Training
- L_z_ans
- L_digits
- total loss
- grad norm

### Timing
- rollout time
- train time
- sync time

### Evaluation
- greedy exact
- pass@N

---

## Notes

- Only accepted (exact-match) trajectories are used for training
- No replay buffer is used
- One trajectory per prompt is selected
- Digits are always generated greedily during rollout

---

## Summary

```
generate → filter exact → select shortest → train with masked CE
```
