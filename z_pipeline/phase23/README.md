
# Phase23 (Merged Mode): Soft Latent Execution + Self-Teacher Distillation

This package implements **Phase2.5 / Merged Mode** latent reasoning for decoder-only LLMs.

“Merged Mode” is an internal nickname referring to the merge of Phase-2 soft latent execution
with Phase-3-style LM-head supervision. In code and checkpoints, this phase is referred to
as **Phase23**.

It keeps a mostly standard Hugging Face CausalLM, preserves `.generate()`, and adds:

- Soft latent execution at `<|latent|>` positions using expected Z embeddings.
- Optional self-teacher distillation (same pre-injection logits, different temperature).
- Digit answer heads at `<ANSWER>` for supervised numeric prediction.
- Counterfactual answer loss that perturbs latent Z mixtures and maximizes output divergence.

## Why Phase 2.5 exists

Phase 2.5 is a deliberate transition stage between soft latent reasoning (Phase 2)
and discrete latent policy optimization (Phase 3 / PPO).

It allows the LM head itself to learn Z-token semantics under full supervision and
differentiability, so that PPO later operates on a well-shaped policy rather than
random logits.

## Core idea

For each latent slot position `p_i` in the sequence (corresponding to a `<|latent|>` token):

1. Run prefix forward to get pre-injection hidden state `u_i`.
2. Compute student logits over Z tokens using LM head rows:
   - `s_i = W_Z @ u_i`
   - `p_i = softmax(s_i / tau_student)`
3. Compute expected Z embedding and inject:
   - `e_i = p_i @ E_Z`
4. Replace embedding at `p_i` with `e_i`.

Optional self-teacher target:

- `q_i = softmax(s_i / tau_teacher)`
- Same `s_i` as student, pre-injection, stop-grad.
- The teacher exists to prevent early collapse of Z distributions and to stabilize training before discrete sampling is introduced.
- The teacher distribution is computed from the **same pre-injection hidden state `u_i`**
- as the student and never observes the injected mixture embedding `e_i`.

Losses:

- `AnswerDigitLoss` (CE over 5 digit heads)
- `self_distill_z_kl_loss = KL(q || p)`
- `CounterfactualAnswerLoss = -JS(p_ref_digits, p_cf_digits)`
- `usage_shaping_loss_stub` (disabled by default)

This loss explicitly enforces that latent Z structure is causally relevant to the final answer, not merely an auxiliary representation.

## Package layout

- `conf.py`: dataclass configs for model/data/loss/train
- `dataset.py`: dataset building, collate, and K-bucket rebalanced sampler
- `model.py`: `UnifiedZSoftModel` and `from_phase1(...)` loader
- `loss.py`: canonical loss definitions
- `train.py`: training loop + checkpoint saving
- `eval.py`: eval loop with same losses/forward API
- `utils.py`: shared helpers
- `__init__.py`: package exports

## Data contract

Each example must contain:

```json
{
  "question": "...",
  "K": 1,
  "final_answer": 12345
}
```

Where:

- `K` is in `[1, k_max]` (default `k_max=20`)
- `final_answer` is an integer in `[0, 99999]`

The dataset builds per-sample tokens as:

`[question_tokens..., <|latent|> x K, <ANSWER>]`

Padding is suffix-only in `collate_fn`.

## K-bucket balancing

Training can rebalance by K buckets with target distribution:

- `K1`, `K2`, `K3`, `K4_7`, `K8_12`, `K13_20`

Buckets are used **only for sampling balance**, not model conditioning.

## Model API

### `UnifiedZSoftModel.forward(...)`

Arguments:

- `input_ids`, `attention_mask`
- `use_self_teacher: bool`
- `tau_student: float`
- `tau_teacher: float`
- `return_distributions: bool`

Returns:

- always: `digit_logits` with shape `[B, 5, 10]`
- auxiliary: `slot_mask` with shape `[B, Kmax]` (internal training mask, not required as external serving API)
- when `return_distributions=True`:
  - `p_student` shape `[B, Kmax, Vz]`
  - `q_teacher` shape `[B, Kmax, Vz]` (or `None` if self-teacher disabled)

### `UnifiedZSoftModel.forward_with_fixed_z_distributions(...)`

Used by counterfactual loss.

- Inputs: `input_ids`, `attention_mask`, `p_z` with shape `[B, Kmax, Vz]`
- Behavior: inject provided per-slot mixtures instead of recomputing student `p`
- Output: `digit_logits` `[B, 5, 10]`

### `generate()` and `generate_with_digits()`

- `generate()` uses HF `.generate()` with logits masking (`AllowedTokensOnly`) and does not replace LM head.
- `generate_with_digits()` runs generation, then computes digit-head logits at `<ANSWER>` (or fallback final token position).

## Configuration reference (`conf.py`)

### `ModelConfig`

- `phase1_dir: str`
- `v_z: int`
- `tau_student: float` (student mixture exploration/sharpness)
- `tau_teacher: float` (self-distillation target sharpness)
- `use_self_teacher: bool`

### `DataConfig`

- `dataset_name: Optional[str]`
- `data_path: Optional[str]` (JSONL/local)
- `train_split: str`
- `eval_split: str`
- `batch_size: int`
- `max_length: Optional[int]`
- `k_max: int`
- `rebalance_train: bool`
- `target_k_dist: Dict[str, float]`

### `LossConfig`

- `lambda_ans: float`
- `lambda_softz: float`
- `lambda_cf: float`
- `lambda_usage: float` (default `0.0`)
- `digit_temperature: float`
- `counterfactual_schedule: Dict[int, float]`
- `freeze_softz_after_steps: Optional[int]`

### `TrainConfig`

- `lr`, `weight_decay`
- `steps`, `grad_accum`
- `print_every`, `save_every`
- `seed`, `output_dir`

## Training

Example (from repo root):

```python
from z_pipeline.phase23.conf import Config
from z_pipeline.phase23.train import train

cfg = Config()
cfg.model.phase1_dir = "/path/to/phase1_checkpoint"
cfg.data.data_path = "/path/to/train.jsonl"  # or set dataset_name + train_split
cfg.train.steps = 1000

train(cfg)
```

Training loop behavior:

- Loads tokenizer/model from `from_phase1(...)`
- Builds `UnifiedDataset`
- Optional K-bucket rebalanced sampler
- Computes losses and weighted sum
- Logs periodic metrics (including effective vocab)
- Saves checkpoints under `output_dir/step_<N>/`

Checkpoint contents:

- `phase23_state.pt`
- tokenizer files
- `config.json`

## Evaluation

```python
from z_pipeline.phase23.conf import Config
from z_pipeline.phase23.eval import evaluate

cfg = Config()
cfg.model.phase1_dir = "/path/to/phase1_checkpoint"
cfg.data.data_path = "/path/to/eval.jsonl"  # or set dataset_name + eval_split

metrics = evaluate(cfg)
print(metrics)
```

Eval computes the same loss components as training and reports dataset averages.

## Dependencies

Typical required packages:

- `torch`
- `transformers`
- `datasets`

Use a Python environment where these are installed and compatible with your CUDA/CPU runtime.

## Notes and assumptions

- Phase23 relies on `<|latent|>` and `<ANSWER>` existing in tokenizer/phase checkpoints.
- `from_phase1(...)` includes pragmatic key mapping from Phase1 weights; adjust if your checkpoint schema differs.
- The package is intentionally focused on merged mode and does not implement PPO/value heads.
- After Phase 2.5, the same model can be trained with PPO by replacing soft mixtures with sampled Z tokens, without changing architecture.

## Quick troubleshooting

- **`Tokenizer missing special tokens`**
  - Ensure Phase1 tokenizer artifacts include `<|latent|>` and `<ANSWER>`.
- **`p_z shape` runtime errors in CF loss**
  - Confirm `p_student`/`p_z` shape is `[B, Kmax, Vz]` and `Vz == len(z_token_ids)`.
- **Bucket errors in rebalancing**
  - Ensure the dataset contains examples across all configured target buckets, or disable rebalancing.
