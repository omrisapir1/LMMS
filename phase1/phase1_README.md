# Phase 1 — Latent Distillation (Coconut-Style)

This repository implements **Phase 1** of LMMS: training a pretrained decoder-only LLM to solve math problems using **latent computation** while supervising only the **final answer digits**.

The model consumes a question prompt, runs an internal latent execution loop using `<|latent|>` placeholder slots, emits `<ANSWER>`, and then produces **exactly 5 digit tokens** autoregressively (zero-padded).

Phase 1 is **not** RL. It is supervised learning with a masked cross-entropy loss that only applies to the 5 digit tokens after `<ANSWER>`.

---

## 0) What Phase 1 Must Do

Given an example with:

- question text
- a parsed list of reasoning steps (`thoughts`)
- `K = len(thoughts)`
- a numeric answer `answer`

We build a training sequence that looks like:

```text
<System + User Prompt containing the question>
<reasoning text and/or latent placeholders ending with <ANSWER>>
<d0><d1><d2><d3><d4>
```

`python`  
`Copy code`

Where `<d0..d4>` are digit tokens `"0"`..`"9"` representing a **zero-padded 5-digit** integer.

Example answer `142` becomes digits: `0 0 1 4 2`, i.e. `<ANSWER>00142`.

**Loss is applied ONLY on those five digit positions.** Everything else is masked out.

---

## 1) Configuration Contract (Single Source of Truth)

Training is driven by `Phase1Config`:

```python
from dataclasses import dataclass

@dataclass
class Phase1Config:
    # Training
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 2

    # Curriculum / stages
    max_thoughts: int = 8
    max_length: int = 2048
    eval_interval_batches: int = 500

    min_delta: float = 0.01  # 1% improvement threshold
    stage_patience: tuple = (3, 3, 3, 3, 3, 3, 3, 5)
    max_steps_first_stage: int = 2
    permutation_loss_interval_batches: int = 8

    keep_prob: tuple[float, ...] = (0.05, 0.1, 0.15, 0.75, 1.0)

    # Dataset (Hugging Face)
    dataset_name: str = "omrisap/GSM8k-Aug_qwen_62K_CoTsplitted"
    dataset_train_split: str = "train"
    dataset_eval_split: str = "eval"

    # Model
    base_model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    torch_dtype: str = "bfloat16"

    # Logging
    log_dir: str = "runs/phase1"
    logg_loss_interval_batches: int = 10
```

Everything described below must map cleanly onto this config.

## 2) Data Contract (Hugging Face Dataset)

The dataset is stage-agnostic and expects preprocessed records containing:

- `question: str`
- `answer: int | str`  
  Numeric answer used for 5-digit supervision (zero-padded). Expected range: 0..99999.
- `thoughts: List[str]`
- `K: int`  
  Number of thoughts (typically K == len(thoughts))

If thoughts or K are missing, the dataset should raise an error.

### 2.1 Upstream preprocessing

Upstream preprocessing derives thoughts and K from a generated solution string using:

- `split_thoughts()` in `split_logic.py`

This phase assumes that preprocessing already happened; Phase 1 training consumes the dataset as-is.

## 3) Tokenization Requirements

Phase 1 relies on two special tokens:

- `<|latent|>` — latent placeholder token
- `<ANSWER>` — answer marker token

Both tokens must exist in the tokenizer vocabulary. The training pipeline must ensure these tokens are added (if missing) and that the model’s embeddings are resized accordingly.

### 3.1 Digit tokens

The answer is supervised autoregressively using digit tokens `"0".."9"`.

Assumption: digits are single tokens. Verify this for your tokenizer.

## 4) Input Construction

For each sample:

- Build a chat-style prompt (system + user) that contains the question.
- Append an `answer_text` string constructed by `format_answer(thoughts, num_latent)`.
- Append exactly five digit tokens after `<ANSWER>` (teacher forcing labels).

The final model inputs are standard `input_ids + attention_mask` (normal causal attention).

No attention masking is applied to hide question/thought/latent tokens.

## 5) Latent/Text Formatting Rule (format_answer)

Let:

- `K = len(thoughts)`
- `num_latent = num_latent_fn(K) clamped to [0, K]`

Rules:

- `num_latent == 0`:
  - output all thoughts
  - no latent placeholders
- `num_latent == 1`:
  - output all thoughts except the last
  - append `<|latent|><ANSWER>`
- `num_latent >= 2`:
  - emit a left block of `num_latent - 1` latent placeholders on its own line (`<|latent|>` repeated contiguously)
  - emit middle thoughts: `thoughts[num_latent-1 : -1]` joined by newlines
  - append `<|latent|><ANSWER>` directly after the last included thought line (no extra newline inserted before it)

Example (`thoughts=[t1,t2,t3,t4], K=4`):

- `num_latent=0 → t1\nt2\nt3\nt4`
- `num_latent=1 → t1\nt2\nt3\n<|latent|><ANSWER>`
- `num_latent=3 → <|latent|><|latent|>\nt3<|latent|><ANSWER>`

This implies latent placeholders are not always a single contiguous block immediately before `<ANSWER>`.

## 6) Curriculum / Stages (Stage-Agnostic Dataset)

The dataset does not decide curriculum behavior. Curriculum is controlled by:

- a stage-dependent `num_latent_fn(K)` supplied by the training/eval loop
- optional row filtering/sampling implemented in training/eval code

### 6.1 Evaluation stage behavior

Evaluation uses the following mapping:

- For stages 1..7:
  - `num_latent = min(stage, K-1)`
  - exclude rows where `K == 1`
- For stage 8:
  - `num_latent = K`
  - include all rows (including `K == 1`, which implies `num_latent = 1`)

This logic is implemented by constructing `num_latent_fn(K)` and passing it into the dataset / collator.

## 7) Stage Progression (StageManager)

Stage progression is patience-based, not threshold-based.

At each evaluation window:

- if validation accuracy improves by at least `min_delta`:
  - update best score
  - reset the no-improvement counter
- otherwise:
  - increment the no-improvement counter
- when counter reaches `stage_patience[current_stage]`:
  - advance stage
  - reset stage baseline (`best_val_acc`) for the new stage

Additionally, stage 1 can be force-exited after `max_steps_first_stage` optimizer steps.

Config knobs:

- `min_delta`
- `stage_patience`
- `max_steps_first_stage`

## 8) Model: Prefix-Optimized Coconut Latent Execution

### 8.1 Goal

Replace `<|latent|>` placeholder embeddings with hidden states computed from the prefix.

This creates a Coconut-style latent computation loop without generating intermediate text.

### 8.2 Algorithm

Let the input sequence contain latent placeholders at positions `p1 < p2 < ... < pn`.

For each placeholder position `p`:

- Run a forward pass on the prefix `input_ids[:p]` (and matching attention mask).
- Take hidden state at position `p-1`.
- Replace the embedding at position `p` with that hidden state.

After all latent slots are filled:

- Run one final full-sequence forward pass with the modified embeddings.
- Locate `<ANSWER>` token position.
- Produce logits for the next tokens and train/generate the 5 digits autoregressively.

### 8.3 Why prefix-only forwards are valid

Hidden state at `p-1` depends only on positions `<= p-1`, so full-sequence recomputation for each latent step is unnecessary.

### 8.4 Required outputs from the model forward

The model forward used by training/eval should return:

- logits for next-token prediction over the full vocab (standard LM logits)
- optionally, a second set of logits for the auxiliary permutation/truncation path (Section 9)

## 9) Permutation + Truncation Auxiliary Path (Train + Eval)

Phase 1 uses a second latent-execution path during both training and evaluation.

It is used to:

- compute `acc_perm` (evaluation metric)
- compute an auxiliary permutation-sensitivity loss (training term)

### 9.1 Auxiliary path construction

For each sample, let `n` be the number of latent placeholders.

- If `n <= 1`:
  - auxiliary path is disabled
- If `n > 1`:
  - with probability `perm_truncate_ratio`: truncate-by-half
  - otherwise: permute latent fill order

Truncate-by-half:

- `m = max(1, ceil(n/2))`
- run latent execution only for the first `m` latent slots (natural order)
- skip the remaining latent slots in the auxiliary path

Permute order:

- if `n == 2`: swap `[1, 0]`
- if `n > 2`: reverse `[n-1, ..., 0]`

### 9.2 Training loss: permutation sensitivity

Training adds an auxiliary loss that encourages the auxiliary-path digit distributions to differ from the normal-path digit distributions.

Compute (for the five digit positions after `<ANSWER>`):

- Extract digit-token probabilities from the full-vocab logits for:
  - normal path
  - auxiliary path
- Compute symmetric KL per digit position, average across the 5 positions → `sym_kl` per sample
- Define:
  - high loss when `sym_kl` is small (outputs too similar)
  - low loss when `sym_kl` is large (outputs differ)

A typical form:

```text
loss = exp(-sym_kl)
```

This objective is meaningful only for `n >= 2`.

### 9.3 When to apply permutation loss

Compute and add this auxiliary loss every:

- `permutation_loss_interval_batches`

At other steps, train with digit CE loss only.

## 10) Answer Loss (Masked CE with Zero Downsampling)

### 10.1 Masked digit-only CE

Compute next-token cross entropy but keep only the five digit positions after `<ANSWER>`.

All other tokens contribute 0 loss and receive no gradient.

### 10.2 Zero-digit downsampling with fixed keep probabilities

Digit supervision can be dominated by zeros. We downsample zero-label digit positions using a fixed `keep_prob` tuple of length 5:

`python`  
`Copy code`

```python
keep_prob = (0.05, 0.1, 0.15, 0.75, 1.0)
```

For each of the five digit positions `i`:

- if the true digit label is non-zero → always include loss
- if the true digit label is zero → include loss with probability `keep_prob[i]`

This mask is sampled independently per position.

Example (keep_prob = (0.05, 0.1, 0.15, 0.75, 1.0)):

Target digits: 0 0 0 9 8

- pos0=0: include loss with prob 0.05 (drop 95%)
- pos1=0: include loss with prob 0.10 (drop 90%)
- pos2=0: include loss with prob 0.15 (drop 85%)
- pos3=9: always include loss
- pos4=8: always include loss


keep_prob[i] applies only when the label digit is zero; non-zero digits always contribute.

Evaluation uses full digit supervision (no downsampling).


## 11) Training Loop (What Must Be Implemented)

A complete training implementation must include:

- Load `Phase1Config`.
- Set seeds.
- Load tokenizer for `base_model`, add special tokens if needed, resize embeddings.
- Load pretrained base model (`base_model`) in `torch_dtype`.
- Build train/eval datasets from HF using split names.
- Implement collation:
  - tokenize prompt + `answer_text` + 5 digit targets
  - truncate to `max_length`
  - create labels and a digit-loss mask for the 5 positions after `<ANSWER>`
- Implement stage-dependent evaluation:
  - compute accuracy on the 5-digit exact match
  - compute `acc_perm` via auxiliary path when enabled
- Implement StageManager:
  - advance stages by patience + `min_delta`
  - force exit stage 1 after `max_steps_first_stage`
- Implement training steps:
  - forward normal path
  - compute masked digit CE with zero downsampling (training only)
  - optionally forward auxiliary path and add permutation loss every `permutation_loss_interval_batches`
  - gradient accumulation
  - optimizer step + scheduler (optional)
- Logging:
  - log losses every `logg_loss_interval_batches`
  - evaluate every `eval_interval_batches`
  - write checkpoints into `log_dir`

## 12) Metrics

At minimum:

- `digit_accuracy` or `exact_5digit_accuracy`  
  Exact match across all 5 digits after `<ANSWER>`
- `acc_perm`  
  Same metric computed on the auxiliary path output (when enabled)
- training losses:
  - digit CE loss (masked + downsampled)
  - permutation sensitivity loss (when applied)

## 13) Run Notes

Typical entrypoints:

- `python -m phase1.train`
- `python phase1/train.py`

### Phase2 Codebook Dataset Generation

Generate `(question, answer_digits, K_star, latent_vectors)` parquet shards from a trained Phase1 checkpoint:

```bash
python -m phase1.generate_codebook_dataset \
  --ckpt_dir phase1/runs/phase1/last_checkpoint \
  --dataset_name omrisap/GSM8k-Aug_qwen_62K_CoTsplitted \
  --split train \
  --output_dir phase1/codebook_data \
  --k_max 20 \
  --batch_size 8 \
  --shard_size 1000
```

Dry run:

```bash
python -m phase1.generate_codebook_dataset \
  --ckpt_dir phase1/runs/phase1/last_checkpoint \
  --dataset_name omrisap/GSM8k-Aug_qwen_62K_CoTsplitted \
  --split train \
  --output_dir phase1/codebook_data_dry \
  --k_max 20 \
  --batch_size 8 \
  --max_rows 32
```

## Summary

Phase 1 provides:

- Autoregressive 5-digit answer supervision after `<ANSWER>`
- Stage-agnostic dataset; curriculum controlled by stage-dependent `num_latent_fn(K)`
- Prefix-optimized Coconut latent execution over `<|latent|>` placeholders
- Patience-based stage progression
- Auxiliary permutation-or-truncate path used in both training and evaluation
- Fixed per-position `keep_prob` to downsample zero-label digit losses during training only

Later phases will build on this latent execution pathway with action-based objectives and reinforcement learning.
