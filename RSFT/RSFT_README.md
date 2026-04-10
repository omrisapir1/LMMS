# LMMS + RSFT Deep-Dive README (QA / Debug / Analysis Guide)

This document is a full onboarding guide for the **RSFT phase** inside this repository, written for contributors who are new to LMMS and need to quickly become effective at:
- QA
- debugging
- experiment analysis
- proposing model/training improvements

It is based on the current code in:
- `RSFT/train.py`
- `RSFT/logic.py`
- `RSFT/dataset.py`
- `RSFT/eval_vllm.py`
- `RSFT/config.py`
- shared rollout/token utilities in `PPO/*`

---

## 1) What LMMS Is (Repo-Level Context)

In this repository, LMMS is a multi-phase math reasoning pipeline that evolves from supervised latent reasoning to stronger post-training policies.

High-level phase intent in this codebase:
1. `thought_embedding/` ("Phase X"):
   - builds thought-state vectors from solved examples.
2. `phase1/`:
   - supervised latent reasoning with `<|latent|>` and `<ANSWER>`, digit-targeted loss.
3. `codebook/`:
   - quantizes latent vectors into discrete `<z_i>` IDs.
4. `SFT/`:
   - supervised training over discrete Z-programs and 5-digit answers.
5. `RSFT/` (this doc):
   - multi-round rejection-sampling-style supervised fine-tuning with verify actions.
6. `PPO/`:
   - reinforcement-learning stage using PPO contracts/rewarding.

So RSFT is not an isolated script. It is the bridge between supervised Z-program training and RL policy optimization.

---

## 2) What RSFT Optimizes

RSFT trains a model to operate in **rounds** for each question:
1. Generate Z reasoning tokens until `<ANSWER>`.
2. Generate exactly 5 digits.
3. Generate one verify action token:
   - `<FINALIZE>`
   - `<RETRY>`

A rollout is a sequence of rounds up to `max_rounds`.

The key supervision rule in current code:
- **Verify token is supervised in every round**.
- **Z + `<ANSWER>` + digits are supervised only on the final successful round**.
- If a sequence never succeeds (max rounds reached), it is a **verify-only** training example.

This is implemented in `RSFT/logic.py` via `build_training_example` and validated by `RSFT/tests/test_rsft_logic.py`.

---

## 3) RSFT Data + Token Contracts

### Dataset contract
Configured in `RSFT/config.py` `DataConfig`:
- `dataset_name` default: `omrisap/numina_openmath`
- split names: `train`, `eval`
- fields:
  - `question`
  - `answer_digits` (preferred)
  - `final_answer` (fallback parser)

### Prompt building
`RSFT/dataset.py` uses `phase1.dataset.SYSTEM_PROMPT` and tokenizer chat template:
- system message = shared LMMS system prompt
- user message = question
- generation prompt added for assistant

### True-digit parsing
`parse_true_digits` accepts:
- strict `answer_digits` lists of length 5
- fallback parsing from `final_answer`, including relaxed cases like boxed/commas/signed integer text

### Tokenization contracts (hard-fail)
Using shared PPO utilities:
- `<ANSWER>` must be exactly one token.
- digits `"0".."9"` must each be one token.
- `<FINALIZE>` and `<RETRY>` must each be one token and distinct.
- Z tokens are discovered from vocab names matching `<z_i>` or `<Z_i>`.

Any violation raises a runtime error early in startup.

---

## 4) RSFT Architecture

RSFT training loop has two model execution roles:
1. **Torch model** (`transformers`) on `rollout.torch_device`:
   - forward/backward/optimizer updates.
2. **Rollout engine** (`vllm` or HF fallback):
   - generates Z/digits/verify rounds under token masks.

`rollout.backend` options:
- `vllm` (default)
- `hf`

For vLLM, `PPO/vllm_rollout.py` is used with weight synchronization from torch model every `sync_every_n_steps` updates.

---

## 5) Exact Training Loop (Step-by-Step)

Implemented in `RSFT/train.py`.

For each training step:
1. Sample unique prompt batch (`_next_unique_batch`).
2. Expand into `rollouts_per_prompt` trajectories.
3. Run multi-round rollout:
   - Z generation (stops when `<ANSWER>` found).
   - digit generation (must be exactly 5 tokens).
   - choose verify action:
     - target = `<FINALIZE>` if digits exact else `<RETRY>`
     - optional forced retry on correct round via `retry_on_correct_prob` logic.
4. Terminal status per sequence:
   - `success` if executed verify is `<FINALIZE>` on correct round.
   - `failed` for `no_answer_before_max_tokens` or max-round exhaustion.
5. Build candidate training examples only for:
   - success
   - full failure (`max_rounds_reached_without_success`)
6. Two-stage selection filter:
   - Stage A: keep prompts only if they had more than one wrong rollout.
   - Stage B: keep all correct examples; keep full-failure examples up to 50% of correct count.
7. Assign per-example weight = inverse of accepted examples per prompt (equal prompt contribution).
8. Compute losses and update model.
9. Sync rollout engine as needed.
10. Log metrics + rollout traces + optional eval.
11. Save periodic and last checkpoints (with trainer state).

Important: optimizer step is skipped if no accepted rows were produced.

---

## 6) Losses and Masking

`RSFT/logic.py::compute_rsft_losses` computes four restricted masked CE terms:
- `l_z` over class `{TARGET_Z}`, allowed IDs = `z_token_ids`
- `l_answer` over class `{TARGET_ANSWER}`, allowed IDs = `[answer_token_id]`
- `l_digits` over `TARGET_DIGIT`, allowed IDs = digit token IDs
- `l_verify` over `TARGET_VERIFY`, allowed IDs = `[finalize_token_id, retry_token_id]`

Total:
- `loss = w_z * l_z + w_answer * l_answer + w_digits * l_digits + w_verify * l_verify`

Defaults (`LossConfig`):
- `w_z = 1.0`
- `w_answer = 1.0`
- `w_digits = 1.0`
- `w_verify = 2.0`

---

## 7) Verify Warmup Mode

Early steps can run a safety warmup (`train.warmup_steps > 0`):
- freeze all params
- unfreeze only input embedding rows (and lm_head rows if untied) for verify token IDs
- gradient hooks zero out all non-verify rows
- optimizer weight decay forced to 0

After warmup, full RSFT mode restores all parameters trainable.

This is one of the most important implementation details for stability and is easy to miss when interpreting results.

---

## 8) Evaluation Modes

`RSFT/eval_vllm.py::evaluate_with_rollout_engine` supports:
- `standard` (always on)
- `retry_bias` (optional): adds logit bias toward `<RETRY>`
- `oracle_auto_retry` (optional): auto-retries wrong rounds until forced finalize/max rounds

Primary metrics:
- `greedy_exact`
- `pass_at_n`
- `mean_z_length`
- `no_answer_before_kmax_rate`

Per-mode sequence logs are saved under:
- `runs/.../logs/eval_modes/step_xxxxxx/{standard,retry_bias,oracle_retry}.jsonl`

Note: eval questions are selected deterministically from `RSFT/sample_questions_for_eval.py` when matching rows are available.

---

## 9) Configuration Map (What to Tune First)

All in `RSFT/config.py`.

### ModelConfig
- `init_ckpt`
- token names for `<ANSWER>`, `<FINALIZE>`, `<RETRY>`

### RolloutConfig
- throughput and exploration: `vllm_batch_size`, `rollouts_per_prompt`, `temperature`, `top_p`, `min_p`
- rollout horizon: `max_rounds`, `max_new_tokens`
- retry behavior: `retry_on_correct_prob`, `retry_on_correct_only_first_round`
- backend/device/w-sync knobs

### TrainConfig
- optimization + runtime length: `lr`, `max_steps`, `max_grad_norm`, `max_length`
- warmup controls: `warmup_steps`, `warmup_lr`
- resume path: `resume_from`

### EvalConfig
- cadence and size: `eval_every_steps`, `max_eval_questions`, `pass_at_n`
- mode toggles and retry/oracle settings

### LoggingConfig
- `output_dir`, `log_every`, `save_every`, `keep_last`

CLI overrides are passed via repeated `--set key=value`.

---

## 10) Running RSFT

### Minimal launch
```bash
python -m RSFT.train
```

### Example with overrides
```bash
python -m RSFT.train \
  --set model.init_ckpt="omrisap/RSFT_250_8" \
  --set rollout.backend="vllm" \
  --set rollout.vllm_batch_size=64 \
  --set rollout.rollouts_per_prompt=8 \
  --set train.max_steps=1000 \
  --set logging.output_dir="./runs/rsft"
```

### Resume
```bash
python -m RSFT.train --set train.resume_from="/abs/path/to/run_or_last_or_step_dir"
```

Valid `resume_from` targets:
- run dir
- `run/last`
- `run/checkpoints/step_xxxxx`
- `.../trainer_state.pt`

---

## 11) Output Artifacts

Per run directory (timestamped under `logging.output_dir`):
- `logs/run.log`
- `logs/metrics.csv`
- `logs/metrics.jsonl`
- `logs/rollouts_rank0_stepXXXX.jsonl`
- `logs/eval_modes/...` (if evaluation runs)
- `checkpoints/step_xxxxx/`
- `last/`
  - model + tokenizer
  - `meta.json`
  - `trainer_state.pt`

`trainer_state.pt` includes optimizer state, RNG states, ordered sampler state, and cursor for deterministic resume.

---

## 12) QA Checklist (What to Validate)

1. Token contracts pass at startup:
   - single-token digits
   - single-token answer/verify tokens
   - non-empty Z token set
2. Rollout invariants:
   - exactly 5 digit tokens per round
   - verify token always in `{<FINALIZE>, <RETRY>}`
3. Masking invariants:
   - verify supervision count equals number of rounds
   - non-final rounds do not supervise z/answer/digits
4. Selection invariants:
   - Stage A prompt filter behavior (must have >1 wrong rollout)
   - Stage B failure cap is 0.5 * correct count
5. Warmup invariants (if enabled):
   - no non-verify-row gradients in embedding/lm_head
6. Resume invariants:
   - dataset length unchanged
   - ordered indices + cursor restored
7. Metrics sanity:
   - accepted rate not silently collapsing
   - `no_answer_before_kmax_rate` stable
   - mode-specific eval metrics are written and coherent

---

## 13) Debugging Guide by Symptom

### Symptom: frequent `no_answer_before_max_tokens`
Check:
- Z/answer mask correctness
- `rollout.max_new_tokens`
- sampling temperature/top_p/min_p too aggressive
- base checkpoint quality for `<ANSWER>` emission

### Symptom: "Digits phase must emit exactly 5"
Check:
- rollout engine token restriction for digits
- tokenizer contract for digit single-token IDs
- engine API regressions in generation length control

### Symptom: verify token outside allowed set
Check:
- verify token IDs resolved and distinct
- verify logits masking in rollout backend
- accidental tokenizer mismatch between torch model and rollout engine

### Symptom: accepted rows always zero
Check:
- over-strict filtering (`wrong_count <= 1` prompt exclusion)
- no successful or max-round-failure trajectories
- max length clipping (`build_training_example` returns `None`)
- prompt dataset parse failures reducing usable rows

### Symptom: unstable/flat losses in warmup
Check:
- warmup mode enabled intentionally
- verify row IDs correct
- grad masking hooks installed and removed at phase switch
- optimizer LR during warmup (`warmup_lr`)

### Symptom: vLLM sync issues
Check:
- `PPO/vllm_rollout.py` weight transfer init logs
- CUDA_VISIBLE_DEVICES and tensor-parallel setup
- sync cadence (`sync_every_n_steps`)
- fallback test with `rollout.backend="hf"`

---

## 14) Brainstorming Levers for Better RSFT

When proposing experiments, these usually have highest impact:
1. Data selection:
   - Stage A/B filtering policy and failure-cap ratio.
2. Retry policy:
   - `retry_on_correct_prob`, first-round-only behavior.
3. Loss balancing:
   - especially `w_verify` vs (`w_z`, `w_answer`, `w_digits`).
4. Warmup strategy:
   - warmup length and verify-only initialization behavior.
5. Decoding policy:
   - digit greedy vs sampled, temperature/top_p/min_p.
6. Round budget:
   - `max_rounds` tradeoff between compute and recoverability.
7. Eval policy realism:
   - standard vs retry-biased vs oracle mode interpretation.

---

## 15) Important Differences vs Older RSFT Descriptions

If you read old notes/docs, align to current code:
- RSFT is now **multi-round with explicit verify actions**.
- Loss is **three-part** (`z_ans`, `digits`, `verify`) not just two-part.
- Training can include **full-failure verify-only examples**.
- Selection is not "shortest successful only"; it uses two-stage prompt/sequence filtering.
- Warmup mode can train verify rows only before full RSFT.

---

## 16) Key Files to Read First

1. `RSFT/train.py`
2. `RSFT/logic.py`
3. `RSFT/eval_vllm.py`
4. `RSFT/dataset.py`
5. `RSFT/config.py`
6. `RSFT/tests/test_rsft_logic.py`
7. `PPO/vllm_rollout.py` and `PPO/hf_rollout.py`
8. `PPO/token_contract.py` and `PPO/masking.py`

---

## 17) Practical Handoff Summary

If you are joining to help with QA/debugging:
- start from one short run with small `max_steps`
- inspect rollout logs + metrics every step
- verify masking contracts before trusting aggregate metrics
- treat eval mode outputs separately (`standard` vs retry/oracle)
- verify resume determinism before long jobs

If you are joining to help analyze results:
- compare acceptance dynamics, round distributions, and no-answer rates
- separate issues caused by generation policy from issues caused by training signal
- always annotate findings with config snapshot from `run.log`
