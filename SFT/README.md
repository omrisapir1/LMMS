# Phase 3 — Discrete Latent SFT (Z-Program Supervision)

This stage trains a pretrained decoder-only LLM to execute **discrete latent programs** learned from Phase2 (codebook) and to emit a correct 5-digit answer.

This phase serves as the bridge:

Phase1 (Coconut continuous latent)  
→ Phase2 (VQ Codebook → discrete Z)  
→ Phase3 (SFT over Z programs)  
→ Phase4 (PPO over Z actions)

The model learns to generate:

prompt + Z_17 Z_153 Z_8 <ANSWER> 01589

with:

- CE supervision on Z tokens
- CE supervision on `<ANSWER>`
- CE supervision on 5 digit tokens
- No loss on the prompt

This is **not RL**. This is supervised behavior cloning over discrete latent programs.

---

# 1) High-Level Objective

Given a training example:

- `question`
- `z_ids: List[int]`
- `answer_digits: List[int]` (exactly 5 digits)

We train the model autoregressively to predict:

prompt → Z tokens → <ANSWER> → 5 digits

with structured masking and weighted losses.

This produces a policy that:

- Knows how to generate latent programs
- Knows how to stop with `<ANSWER>`
- Knows how to produce 5-digit numeric answers
- Is structurally constrained for PPO

---

# 2) Dataset Contract

## 2.1 Training Dataset (HF path provided)

Each row must contain:

- `question: str`
- `z_ids: List[int]`  (length = K)
- `answer_digits: List[int]` (length = 5)

The dataset is the output of Phase2 export.

## 2.2 Evaluation Dataset

Separate HF path.

Used for pass@N evaluation via vLLM.

---

# 3) Tokenization Contract

## 3.1 Required Tokens

We must add:

<z_0> ... <z_{V-1}>

Where:

- `V` = vocab_size from CodebookConfig
- V is configurable (since multiple codebooks exist)

`<ANSWER>` already exists from Phase1.

## 3.2 Tokenizer Requirements

- Each `<z_i>` must tokenize to **exactly one token**
- `<ANSWER>` must tokenize to one token
- Digits `"0".."9"` must be single tokens

After adding Z tokens:

```python
tokenizer.add_tokens(z_tokens)
model.resize_token_embeddings(len(tokenizer))
```

---

# 4) Model Modifications

## 4.1 Embedding Matrix

New rows added for:

- `<z_0> ... <z_{V-1}>`

## 4.2 LM Head

LM head is **NOT tied** to embeddings.

Therefore:

- New rows must also be added to `lm_head.weight`

---

# 5) Warmup Phase

Before full SFT, perform a warmup phase.

## 5.1 Goal

Train only the newly added token rows:

- embedding rows for Z tokens
- lm_head rows for Z tokens

All other parameters remain frozen.

## 5.2 Config Parameter

```
warmup_steps: int
```

During warmup:

- Freeze all parameters
- Unfreeze only:
  - `embedding.weight[z_token_ids]`
  - `lm_head.weight[z_token_ids]`

After `warmup_steps`, unfreeze full model.

---

# 6) Sequence Construction

For each example:

```
input = prompt + Z tokens + <ANSWER> + 5 digits
```

Example:

```
<system+user prompt>
<z_17> <z_153> <z_8> <ANSWER> 0 1 5 8 9
```

---

# 7) Two-Phase Restricted Logits Masking

Restricted masking is applied:

- During training forward pass
- During evaluation generation

## 7.1 Z Phase

While generating Z tokens and `<ANSWER>`:

Allowed tokens:

```
{Z tokens} ∪ {<ANSWER>}
```

All other tokens must be masked to `-inf`.

Digits are NOT allowed before `<ANSWER>`.

## 7.2 Digit Phase

After `<ANSWER>` is emitted:

Allowed tokens:

```
{"0".."9"}
```

No Z tokens allowed.

Exactly 5 digits are generated.

---

# 8) Loss Definition

Loss is computed only on:

- Z tokens
- `<ANSWER>` token
- 5 digit tokens

Prompt tokens are fully masked.

## 8.1 Loss Weights

```
w_z = 0.1
w_answer = 0.5
w_digits = 1.0
```

These are configurable.

## 8.2 Z Label Smoothing

Apply label smoothing only to Z tokens:

```
z_label_smoothing = 0.05
```

Digit tokens use standard CE (no smoothing).

## 8.3 Total Loss

Let:

- `L_z` = CE over Z tokens
- `L_answer` = CE over `<ANSWER>`
- `L_digits` = CE over 5 digits

Then:

```
L_total = w_z * L_z
        + w_answer * L_answer
        + w_digits * L_digits
```

---

# 9) Training Forward Masking (Important)

Restricted logits masking is applied during forward pass.

This ensures:

- Model cannot allocate probability to illegal tokens
- Model respects grammar
- PPO transition is smooth

This is mandatory.

---

# 10) Evaluation Protocol

Evaluation uses vLLM.

## 10.1 Generation

For each prompt:

1. Generate tokens with restricted mask:
   - Only Z tokens + `<ANSWER>`
2. Stop when `<ANSWER>` is emitted OR when `Kmax` reached.
3. After `<ANSWER>`, allow only digit tokens.
4. Generate exactly 5 digits.

## 10.2 pass@N Metric

For each question:

- Sample `N` solutions (e.g. 8, 16)
- Compute whether at least one solution matches ground-truth digits

Metric:

```
pass@N = fraction of questions with ≥1 correct solution
```

We do NOT use greedy-only accuracy as primary metric.

---

# 11) Config Parameters

Example SFT config:

```python
vocab_size: int
warmup_steps: int
z_label_smoothing: float = 0.05

w_z: float = 0.1
w_answer: float = 0.5
w_digits: float = 1.0

eval_interval_steps: int
pass_at_n: int
k_max: int
```

---

# 12) Metrics to Log

During training:

- `L_total`
- `L_z`
- `L_answer`
- `L_digits`
- Z token accuracy
- Digit exact match accuracy
- Average Z length
- Rate of “no `<ANSWER>` before Kmax”

During eval:

- pass@N
- Greedy exact match (secondary)
- Distribution of generated Z lengths

---

# 13) Why This Is the Correct Bridge to PPO

After SFT:

- Z tokens represent discrete actions
- `<ANSWER>` represents stop action
- Model already trained to respect grammar
- Digit head trained to map latent program to final answer

PPO will:

- Replace CE objective
- Use digit correctness reward
- Adjust Z policy for better reward
- Potentially shorten programs

Because masking and grammar are already enforced in SFT,
PPO only needs to optimize action selection, not structure.

---

# 14) Failure Modes to Watch

1. Z collapse (low entropy)
2. Model emits digits before `<ANSWER>` (masking bug)
3. Model fails to emit `<ANSWER>`
4. Z loss dominates digit loss (wrong weights)
5. Overfitting to teacher Z sequences (too high `w_z`)

---

# 15) Final Summary

Phase 3 (SFT over discrete Z programs):

- Converts discrete codebook latents into executable programs
- Teaches stop behavior
- Supervises final answer digits
- Enforces strict structural grammar
- Prepares policy for PPO optimization

This phase is purely supervised and forms the stable initialization for RL fine-tuning.