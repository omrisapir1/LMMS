# Phase 3 — Curriculum SFT for Mixed Text-to-Z Reasoning

This stage trains a pretrained decoder-only LLM to move **gradually** from full-text reasoning traces to fully discrete latent reasoning programs.

Instead of training only on:

```text
question + Z tokens + <ANSWER> + 5 digit tokens
```

we now train with a **curriculum** that starts with mostly text thoughts and gradually replaces the reasoning trace with Z tokens **from right to left**.

This phase serves as the bridge:

Phase1 (continuous / text-style reasoning signals)  
→ Phase2 (VQ codebook → discrete Z tokens)  
→ Phase3 (curriculum SFT over mixed text + Z reasoning)  
→ Phase4 (fully discrete PPO / RL)

The final target format remains:

```text
Question + reasoning_trace + <ANSWER> + 5 digit tokens
```

where `reasoning_trace` changes by curriculum phase:

- early phases: mostly text thoughts + short Z suffix
- late phases: mostly Z tokens
- final phase: only Z tokens

Supervision remains **restricted** to:

- Z tokens
- `<ANSWER>` token
- 5 digit answer tokens

There is **no CE loss** on question tokens or text-thought tokens.

This stage is still **supervised learning**, not RL.

---

# 1) High-Level Objective

For each training example we have:

- `question`
- `splitted_solution: List[str]` — one text thought per step
- `z_ids: List[int]` — one Z token id per step
- `answer_digits: List[int]` — exactly 5 digits

Important invariant:

```text
len(splitted_solution) == len(z_ids)
```

for every row in the dataset.

We train the model autoregressively on a curriculum of mixed reasoning traces:

```text
question → text thoughts prefix → Z suffix → <ANSWER> → 5 digits
```

where the Z suffix grows over phases until the model reaches:

```text
question → full Z program → <ANSWER> → 5 digits
```

This produces a policy that:

- first learns with easier text-heavy reasoning prefixes
- gradually adapts to compressed latent reasoning
- learns to emit `<ANSWER>` after the latent program
- learns to emit the final 5-digit numeric answer
- is structurally aligned with the downstream fully-discrete setup

---

# 2) Dataset Contract

## 2.1 Required Columns

Each dataset row must contain:

- `question: str`
- `splitted_solution: List[str]`
- `z_ids: List[int]`
- `answer_digits: List[int]` (length = 5)
- `tokens_count: int`

## 2.2 Semantics

### `splitted_solution`

A list of textual reasoning steps.

Example:

```python
[
  "First compute the number of groups.",
  "Now simplify the fraction.",
  "Therefore the final intermediate value is 18."
]
```

### `z_ids`

A list of discrete codebook ids aligned 1:1 with `splitted_solution`.

If a row has `k = len(z_ids)`, then the row also has exactly `k` text thoughts.

### `tokens_count`

`tokens_count` is a **single precomputed column per row** representing the token count of the **original full-solution text**.

It is used for per-phase filtering by sequence budget.

It does **not** represent a phase-specific mixed-sequence token count.

## 2.3 Answer Digits

`answer_digits` are always interpreted as exactly 5 digit tokens.

Example:

```python
[0, 1, 5, 8, 9]
```

---

# 3) Curriculum Definition

Training proceeds through ordered curriculum phases. Each phase specifies what fraction of the reasoning trace is replaced by Z tokens.

Replacement always happens **from right to left**.

## 3.1 Phase Ratios

The intended default phases are:

1. `0.00`
2. `0.25`
3. `0.50`
4. `0.75`
5. `1.00`

## 3.2 Number of Z Tokens Per Sample

Let:

- `k = len(z_ids)`
- `r = z_ratio`

Then the number of Z tokens used for a sample is:

- for phases below `100%`:

```python
num_z = max(1, floor(k * r))
```

- for the `100%` phase:

```python
num_z = k
```

This means the `0%` phase is implemented as:

```python
num_z = 1
```

So “0% Z” should be interpreted as:

> zero percent, rounded up to a minimum of one trailing Z token.

## 3.3 Mixed Sequence Construction

For a row with `k` thoughts and `num_z` trailing Z tokens:

- keep the first `k - num_z` thoughts as text
- replace the last `num_z` thoughts with their aligned Z tokens

Text thoughts are merged using:

```python
"\n\n".join(thoughts)
```

### Example: 0% phase (minimum 1 Z)

```text
Question + first (k-1) text thoughts + last Z token + <ANSWER> + 5 digits
```

### Example: 25% phase

For `k = 100`:

```text
Question + first 75 text thoughts + last 25 Z tokens + <ANSWER> + 5 digits
```

### Example: 100% phase

```text
Question + all Z tokens + <ANSWER> + 5 digits
```

No text thoughts remain in the reasoning trace.

---

# 4) Per-Phase Data Filtering and Reshuffling

Each curriculum phase has its own training constraints and therefore its own effective dataset.

For each phase:

1. filter rows using the phase-specific `max_tokens`
2. rebuild the mixed reasoning trace for that phase
3. reshuffle the dataset
4. train for the configured number of epochs

Because each phase has a different token budget, the set of usable examples may differ between phases.

---

# 5) Tokenization Contract

## 5.1 Required Special Tokens

We must add:

```text
<z_0> ... <z_{V-1}>
```

where:

- `V = vocab_size`
- `vocab_size` comes from the Phase2 codebook export / config

`<ANSWER>` is also required.

## 5.2 Tokenizer Requirements

- each `<z_i>` must tokenize to exactly one token
- `<ANSWER>` must tokenize to exactly one token
- digits `"0" ... "9"` must each be single tokens

After adding Z tokens:

```python
tokenizer.add_tokens(z_tokens)
model.resize_token_embeddings(len(tokenizer))
```

---

# 6) Model Modifications

## 6.1 Embedding Matrix

New rows are added for:

- `<z_0> ... <z_{V-1}>`

## 6.2 LM Head

If `lm_head` is untied from the embeddings, corresponding rows must also exist in:

- `lm_head.weight`

---

# 7) Sequence Format

For each example, the serialized target conceptually looks like:

```text
question + mixed_reasoning_trace + <ANSWER> + 5 digit tokens
```

Examples:

### Early curriculum phase

```text
<question>
<text thought 1>
<text thought 2>
...
<z_17> <z_153> <z_8>
<ANSWER>
0 1 5 8 9
```

### Final curriculum phase

```text
<question>
<z_4> <z_17> <z_153> <z_8>
<ANSWER>
0 1 5 8 9
```

---

# 8) Restricted Logits and Supervision Regions

The training sequence now contains three different logical regions:

1. text-thought region
2. Z / `<ANSWER>` region
3. digit region

These regions are handled differently.

## 8.1 Text-Thought Region

The text-thought prefix is included in the autoregressive sequence, but:

- it receives **no CE loss**
- it does **not** participate in the restricted CE objective

In other words, text reasoning tokens are present in the sequence but are not supervised targets for the SFT loss.

## 8.2 Z Region

When the target position corresponds to Z-program prediction and stop prediction:

Allowed tokens are:

```text
{Z tokens} ∪ {<ANSWER>}
```

CE loss is applied only on:

- Z token positions
- the `<ANSWER>` position

## 8.3 Digit Region

After `<ANSWER>`:

Allowed tokens are:

```text
{"0" ... "9"}
```

Exactly 5 digit tokens are produced.

CE loss is applied on those 5 positions.

---

# 9) Loss Definition

The supervised CE objective is computed only on:

- Z tokens
- `<ANSWER>` token
- 5 digit tokens

No CE is applied on:

- question / prompt tokens
- text-thought tokens

## 9.1 Weighted CE Objective

Let:

- `L_z` = CE over Z token positions
- `L_answer` = CE over `<ANSWER>`
- `L_digits` = CE over the 5 answer digits

Then:

```text
L_ce = alpha_z * L_z
     + alpha_answer * L_answer
     + alpha_digits * L_digits
```

where the three weights are phase/run-configurable.

Example config names:

```python
alpha_z
alpha_answer
alpha_digits
```

## 9.2 Text Tokens Are Masked Out

Even in mixed phases, text thoughts are not part of the supervised objective.

They are present only as part of the autoregressive training sequence and context.

---

# 10) Counterfactual Loss (CF Loss)

Some phases may optionally enable an additional **Counterfactual Loss** regularizer.

Its purpose is to encourage the model to make the answer digits depend meaningfully on the Z-token reasoning suffix.

## 10.1 Motivation

Without this regularizer, the model may under-use the Z tokens, especially in early curriculum phases where a long text prefix already carries strong answer information.

The counterfactual objective is intended to ensure that changing or corrupting the Z reasoning trace changes the digit distribution in a measurable way.

## 10.2 High-Level Behavior

The implementation compares:

- clean digit logits
- counterfactual digit logits derived from a perturbed Z suffix

and computes a regularizer over the answer-digit region.

Supported counterfactual variants include:

- `truncate`
- `reverse`
- `random`

The current implementation behaves as follows:

- `truncate` / `reverse`: penalize cases where symmetric KL divergence is too small
- `random`: maximize uncertainty / entropy under corrupted Z context

Formally, the helper computes a loss from:

- `clean_digit_logits`
- `cf_digit_logits`
- `digit_valid_mask`
- `eligible_mask`
- `variant_name`
- `kl_margin`
- `eps`

and returns:

- scalar CF loss
- mean symmetric KL
- mean entropy

## 10.3 Phase Control

CF loss is enabled or disabled **per curriculum phase**.

If disabled for a phase, training uses CE only.

If enabled, the total objective is:

```text
L_total = L_ce + lambda_cf * L_cf
```

where `lambda_cf` is configurable.

---

# 11) Checkpointing

The model is saved:

- after every completed epoch within a phase
- and again at phase end

This makes it possible to:

- inspect intermediate curriculum checkpoints
- resume from phase boundaries
- compare training quality across phases

---

# 12) Training Schedule Format

The recommended config structure is an ordered list of phase definitions.

Example:

```yaml
phases:
  - z_ratio: 0.00
    min_z_tokens: 1
    batch_size:  ...
    gradient_accumulation_steps: ...
    max_tokens: ...
    epochs: 0.5
    cf_loss: false

  - z_ratio: 0.25
    min_z_tokens: 1
    batch_size: ...
    gradient_accumulation_steps: ...
    max_tokens: ...
    epochs: 1.0
    cf_loss: true

  - z_ratio: 0.50
    min_z_tokens: 1
    batch_size: ...
    gradient_accumulation_steps: ...
    max_tokens: ...
    epochs: 1.0
    cf_loss: true

  - z_ratio: 0.75
    min_z_tokens: 1
    batch_size: ...
    gradient_accumulation_steps: ...
    max_tokens: ...
    epochs: 1.0
    cf_loss: true

  - z_ratio: 1.00
    min_z_tokens: 1
    batch_size: ...
    gradient_accumulation_steps: ...
    max_tokens: ...
    epochs: 1.0
    cf_loss: true
```

Notes:

- `epochs` may be fractional, e.g. `0.5`
- `max_tokens` is phase-specific
- `batch_size` is phase-specific
- `gradient_accumulation_steps` is phase-specific
- `cf_loss` is phase-specific
- `min_z_tokens` is effectively `1` for all sub-100% phases

---

# 13) Suggested Config Fields

A practical config will usually include:

```python
vocab_size: int
alpha_z: float
alpha_answer: float
alpha_digits: float
lambda_cf: float
cf_variant: str
cf_kl_margin: float
cf_eps: float
save_every_epoch: bool = True
save_phase_end: bool = True
phases: list[PhaseConfig]
```

Where each `PhaseConfig` contains at least:

```python
z_ratio: float
min_z_tokens: int = 1
batch_size: int
gradient_accumulation_steps: int
max_tokens: int
epochs: float
cf_loss: bool
```

---

# 14) Logging

During training it is useful to log:

- total loss
- CE total loss
- `L_z`
- `L_answer`
- `L_digits`
- CF loss (when enabled)
- mean symmetric KL (when CF enabled)
- mean entropy (when CF enabled)
- number of rows kept after phase filtering
- current phase id / ratio
- average number of Z tokens per sample in the phase
- average number of text thoughts per sample in the phase

---

# 15) What Changed Relative to the Old Phase-3 Design

The previous design trained directly on:

```text
question + full Z sequence + <ANSWER> + 5 digits
```

The new design instead trains with a curriculum:

```text
question + text-prefix / Z-suffix mixture + <ANSWER> + 5 digits
```

Key changes:

- full-Z-only training is replaced by multi-phase curriculum training
- text thoughts are introduced as unsupervised intermediate context
- Z tokens grow from a minimum trailing suffix to the full reasoning trace
- each phase has its own token budget and optimization settings
- optional counterfactual regularization can be enabled per phase
- warmup is removed entirely
- evaluation during training is removed

---

# 16) Final Summary

Phase 3 now trains the model to transition from natural-language reasoning traces to fully discrete latent reasoning programs.

It does so by:

- replacing thoughts with Z tokens from right to left
- keeping a minimum of one Z token even in the `0%` phase
- supervising only Z / `<ANSWER>` / digit targets
- optionally enforcing Z usefulness through counterfactual regularization
- progressively ending at the fully discrete format required for downstream RL

The final curriculum phase is still the same target structure needed later:

```text
Question + Z tokens + <ANSWER> + 5 digit tokens
```

but the path to get there is now smoother, more data-efficient, and better aligned with gradual compression of reasoning.

---

# 17) Resume Semantics (Implementation Note)

Current training resume is **coarse-grained**:

- resume state tracks phase index + epoch index (plus global step)
- it does **not** track exact consumed dataloader batch offsets

Therefore:

- if training stops mid-epoch, resume restarts that epoch from its beginning
- if training stops mid fractional-epoch segment, resume restarts that partial segment from its beginning

This is expected behavior unless finer-grained batch-level resume state is added in a future change.
