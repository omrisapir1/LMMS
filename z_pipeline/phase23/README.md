# Phase23-GS (Merged Mode): Discrete Latent Execution via Gumbel-Softmax ST

This package implements **Phase2.5-GS / Merged Mode (GS-ST)** latent reasoning for decoder-only LLMs.

"Phase23-GS" is an internal nickname referring to the merge of:
- Coconut-style latent reasoning
- Discrete Z-token execution
- Phase-3-compatible LM-head semantics

using the **Gumbel-Softmax Straight-Through (GS-ST)** estimator.

Phase23-GS corresponds to Phase 2.5-GS in the overall pipeline.

It keeps a mostly standard Hugging Face CausalLM, preserves `.generate()`, and adds:

- Discrete latent execution at `<|latent|>` positions using GS-ST.
- A Z-token policy implemented directly in the LM head (no separate selector).
- Digit answer heads at `<ANSWER>` for supervised numeric prediction.
- Counterfactual answer loss that perturbs latent Z sequences and maximizes output divergence.
- Optional batch-level and in-sequence Z regularization to prevent collapse.

Unlike Phase23 (soft mixtures), this phase executes **hard discrete Z tokens in the forward pass**, while remaining fully differentiable during training.

---

## Why Phase 2.5-GS exists

Phase 2.5-GS is a deliberate transition stage between:

- **Soft latent reasoning** (Phase 2 / Coconut-style execution)
- **Discrete latent policy optimization** (Phase 3 / PPO)

It teaches the LM head itself to act as a **latent action policy** over Z tokens *without requiring any Z supervision*.

By the end of this phase:

- Z tokens are true discrete actions
- The model is PPO-ready
- The model is compatible with standard HF `.generate()` and vLLM

---

## Core idea

For each latent slot position `p_i` in the sequence (corresponding to a `<|latent|>` token):

1. Run prefix forward to get the pre-injection hidden state `u_i`.
2. Compute logits over Z tokens using LM head rows:
   - `s_i = W_Z @ u_i`
3. Sample a Z distribution using Gumbel-Softmax:
   - `p_i = softmax((s_i + g_i) / tau)`
4. Perform **GS-ST execution**:
   - **Forward (hard):** `k = argmax(p_i)`, inject `E_Z[k]`
   - **Backward (soft):** gradients flow as if `e_i = p_i @ E_Z`
5. Replace the embedding at `p_i` with the injected Z embedding and continue the forward pass.

This makes Z tokens:
- Discrete in execution
- Differentiable in training
- Interpretable as latent actions
- Discrete actions represented as Z tokens

---

## `<ANSWER>` semantics

- `<ANSWER>` is a real token in the vocabulary.
- It terminates latent execution.
- Digit heads read the hidden state at `<ANSWER>`.

This is the **only position with next-token supervision** in this phase.

---

## Losses

Phase23-GS intentionally avoids any cross-entropy supervision over Z tokens. All learning signals are compatible with GS-ST.

### Primary losses

- `AnswerDigitLoss`
  Cross-entropy over 5 digit heads at `<ANSWER>`.

- `AnswerTokenSFTLoss`
  Cross-entropy enforcing that `<ANSWER>` follows the final Z token.

- `CounterfactualAnswerLoss`
  Negative JS divergence between digit distributions produced by:
  - the original Z sequence
  - a perturbed Z sequence (reverse / resample)

This explicitly enforces that Z tokens are **causally relevant** to the final answer.

### Optional regularizers

- `BatchZDiversityLoss`
  KL divergence between the batch-averaged soft Z distribution (`p_z`) and uniform.
  Prevents global Z collapse.

- `InSequenceZConsistencyLoss`
  KL divergence between adjacent Z distributions within a single sequence.
  Encourages stable latent trajectories.

No loss in this phase requires ground-truth Z labels.

---

## Package layout

- `conf.py`: dataclass configs for model/data/loss/train
- `dataset.py`: dataset building, collate, and K-bucket rebalanced sampler
- `model.py`: GS-ST latent execution model and `from_phase1(...)` loader
- `loss.py`: canonical loss definitions
- `train.py`: training loop + checkpoint saving
- `eval.py`: eval loop with same losses/forward API
- `utils.py`: shared helpers
- `__init__.py`: package exports

---

## Data contract

Each example must contain:

```json
{
  "question": "...",
  "K": 1,
  "answer_digits": 12345
}
```

Where:

- `K` is in `[1, k_max]` (default `k_max=20`)
- `answer_digits` is an integer in `[0, 99999]`

The dataset builds per-sample tokens as:

`[question_tokens..., <|latent|> x K, <ANSWER>]`

Padding is suffix-only in `collate_fn`.

### K-bucket balancing

Training can rebalance by K buckets with target distribution:

- `K1`, `K2`, `K3`, `K4_7`, `K8_12`, `K13_20`

Buckets are used only for sampling balance, not model conditioning.

## Model API

### `UnifiedZGSModel.forward(...)`

Arguments:

- `input_ids`, `attention_mask`
- `tau: float` (Gumbel-Softmax temperature)
- `return_distributions: bool`

Returns:

- always: `digit_logits` with shape `[B, 5, 10]`
- auxiliary: `slot_mask` with shape `[B, Kmax]`
- when `return_distributions=True`:
  - `p_z` shape `[B, Kmax, Vz]` (soft distributions used for gradients)

### `UnifiedZGSModel.forward_with_fixed_z(...)`

Used by counterfactual loss.

- Inputs: `input_ids`, `attention_mask`, `z_override`
- Behavior: injects provided Z tokens or distributions
- Output: `digit_logits [B, 5, 10]`

### `generate()` and `generate_with_digits()`

- `generate()` uses HF `.generate()` with logits masking over `{Z tokens + <ANSWER>}`.
- `generate_with_digits()` runs generation, then computes digit-head logits at `<ANSWER>`.

No latent injection occurs during inference.
GS-ST is applied only during training; inference uses standard discrete token generation.

## Configuration reference (`conf.py`)

### ModelConfig

- `phase1_dir: str`
- `v_z: int`
- `gumbel_tau_start: float`
- `gumbel_tau_end: float`
- `gumbel_anneal_steps: int`

### DataConfig

- `dataset_name: Optional[str]`
- `data_path: Optional[str]`
- `train_split, eval_split`
- `batch_size`
- `max_length`
- `k_max`
- `rebalance_train`
- `target_k_dist`

### LossConfig

- `lambda_ans`
- `lambda_sft`
- `lambda_cf`
- `lambda_batch`
- `lambda_consistency`
- `digit_temperature`
- `counterfactual_schedule`

### TrainConfig

- `lr, weight_decay`
- `steps, grad_accum`
- `print_every, save_every`
- `seed, output_dir`

## Training

Example:

```python
from z_pipeline.phase23_gs.conf import Config
from z_pipeline.phase23_gs.train import train

cfg = Config()
cfg.model.phase1_dir = "/path/to/phase1_checkpoint"
cfg.data.data_path = "/path/to/train.jsonl"
cfg.train.steps = 1000

train(cfg)
```

Training loop behavior:

- Loads tokenizer/model from `from_phase1(...)`
- Builds dataset with `<|latent|>` slots
- Optional K-bucket rebalanced sampling
- Applies GS-ST latent execution
- Computes losses and weighted sum
- Saves checkpoints under `output_dir/step_<N>/`

## Evaluation

```python
from z_pipeline.phase23_gs.conf import Config
from z_pipeline.phase23_gs.eval import evaluate

cfg = Config()
cfg.model.phase1_dir = "/path/to/phase1_checkpoint"
cfg.data.data_path = "/path/to/eval.jsonl"

metrics = evaluate(cfg)
print(metrics)
```

Eval computes the same loss components as training.

## Notes and assumptions

- `<|latent|>` and `<ANSWER>` must exist in the tokenizer.
- Phase23-GS does not require Z labels.
- The package does not implement PPO or value heads.
- After Phase 2.5-GS, the same model can be trained with PPO without architectural changes.

## Quick troubleshooting

### Tokenizer missing special tokens

Ensure Phase1 tokenizer includes `<|latent|>` and `<ANSWER>`.

### Z collapse

- Increase `lambda_batch` or temperature
- Verify Gumbel noise is enabled

### Unstable Z switching

- Increase `lambda_consistency`
- Slow down temperature annealing
