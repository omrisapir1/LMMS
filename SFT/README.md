# SFT (Harmony + GPT-OSS-20B PEFT)

This stage trains a Harmony-formatted SFT policy over discrete Z programs.

Current implementation uses:
- Harmony chat-template serialization (tokenizer-native defaults)
- Restricted supervised CE objective over analysis and final-digit regions
- GPT-OSS-20B base loading with MXFP4 dequantized training path
- PEFT/LoRA + row-local Z-token training (embedding + LM-head)

This stage is supervised learning (not PPO).

---

## 1) Data Contract

Each training row must contain:
- `question: str`
- `z_ids: List[int]` (indices into `<z_0> ... <z_{vocab_size-1}>`)
- `answer_digits: List[int]` of length 5 (or `answer_int` in `[0, 99999]`)

`answer_digits` are always interpreted as exactly 5 digits.

---

## 2) Harmony Sequence Shape

Examples are built through `tokenizer.apply_chat_template(...)` with assistant `thinking` + `content` structure, conceptually:

- system (tokenizer default)
- user question
- assistant `analysis` channel containing Z tokens, closed by one analysis `<|end|>`
- assistant `final` channel containing exactly 5 digit tokens

Supervision is token-class based:
- `TARGET_IGNORE`
- `TARGET_ANALYSIS`
- `TARGET_ANALYSIS_END`
- `TARGET_DIGIT`

Only one specific analysis-closing `<|end|>` is supervised (not all `<|end|>` tokens globally).

---

## 3) Supervision / Loss Regions

No CE loss on:
- structural prompt/header tokens
- final structural tail tokens (for example `<|return|>`)

Restricted CE on analysis:
- all `TARGET_ANALYSIS` positions
- the single `TARGET_ANALYSIS_END` position
- allowed ids: `{z_token_ids} ∪ {analysis_end_token_id}`

Restricted CE on final answer:
- exactly 5 `TARGET_DIGIT` positions
- allowed ids: `{digit_token_ids for 0..9}`

Total weighted loss:
- `L_total = w_z * L_analysis + w_answer * L_analysis_end + w_digits * L_digits`

Counterfactual regularizer is also applied per config (truncate/reverse/random variants).

---

## 4) Digit Tokenization Contract

The pipeline enforces:
- each digit `"0" ... "9"` maps to exactly one token id
- final answer supervision always uses exactly 5 digit-token positions

Important: the code does not rely on tokenizing a raw string like `"00006"` as one chunk.

---

## 5) Clipping Safety (Fail-Closed)

After max-length clipping, an example is kept only if it still has:
- exactly one supervised analysis-end position
- exactly five supervised digit positions

Otherwise it is dropped.

If an entire collated batch becomes invalid after clipping, training raises an error instead of continuing with corrupted supervision.

Collator emits clipping-drop stats (batch and cumulative), which are logged in training metrics.

---

## 6) Model/Training Setup

### 6.1 Base Model Loading

Current defaults target GPT-OSS-20B:
- `base_model_or_checkpoint = "openai/gpt-oss-20b"`
- `Mxfp4Config(dequantize=True)`
- `torch_dtype=torch.bfloat16` (configurable)
- `attn_implementation="eager"`
- `use_cache=False`

### 6.2 PEFT/LoRA

LoRA is attached via PEFT with:
- `target_modules="all-linear"`
- optional MoE-aware `target_parameters` discovery from expert parameter names

### 6.3 Row-Local Z Training (True Row-Only Optimizer State)

In addition to LoRA, the code trains compact row-local parameters for Z rows only:
- `embedding_row_deltas: [num_z, hidden_dim]`
- `lm_head_row_deltas: [num_z, hidden_dim]`

`num_z = cfg.vocab_size`, corresponding to `<z_0> ... <z_{num_z-1}>`.

Base embedding/lm_head full tensors remain frozen and are not optimizer parameters.

Forward path applies row deltas by:
- adding embedding delta at input positions whose token id is in Z ids
- adding Z-column logits correction using final LM-head input activations

Optimizer includes only:
- LoRA trainables
- `embedding_row_deltas`
- `lm_head_row_deltas`

---

## 7) Startup Diagnostics

At startup, training prints once:
- one regular example from a real collated batch
- one counterfactual example (deterministic truncate variant)

Each section prints decoded text, token strings, ids, and supervision metadata to verify sequence construction.

Training logs also report:
- total/trainable parameter counts and percentage
- LoRA trainable count
- row-local embedding/lm-head parameter counts
- clipping-drop counters

---

## 8) Checkpointing / Export

Saved artifacts include:
- tokenizer
- PEFT adapter
- row state file: `row_state.pt`

`row_state.pt` stores **full effective trained Z rows** (not just deltas):
- `embedding_rows_effective`
- `lm_head_rows_effective`
- `z_token_ids`

Merged export path:
1. load base model
2. resize embeddings to tokenizer size
3. load + merge LoRA adapter
4. overwrite Z rows directly with saved effective rows

This is designed to make Z-row reconstruction faithful.

---

## 9) Key Config Fields

Main knobs in `SFT/config.py`:
- data/training: `vocab_size`, `batch_size`, `max_steps`, `max_length`, `learning_rate`
- objective: `w_z`, `w_start_answer`, `w_end_answer`, `w_start_digits`, `w_end_digits`, `z_label_smoothing`
- counterfactual: `cf_enabled`, `cf_every_n_steps`, `cf_lambda`, `cf_prob_tuple`, `cf_trunc_range`
- GPT-OSS loading: `dequantize_mxfp4`, `force_bfloat16`, `attn_implementation`
- LoRA: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, MoE targeting fields
- export: `save_merged_for_eval`

---

## 10) Notes

- This README reflects the current SFT code path in `SFT/train.py`, `SFT/dataset.py`, and `SFT/losses.py`.
- PPO integration is out of scope for this stage.
