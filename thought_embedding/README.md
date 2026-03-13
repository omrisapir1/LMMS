# Phase X — Thought-State Embedding Dataset Builder

This phase converts a dataset of **math questions with full worked solutions** into a new dataset where each textual solution is replaced by a **sequence of dense vectors**.

The core idea is:

- Input: `question + solution text`
- Split solution into ordered `thoughts`
- For each thought index `t`, build a text representing the reasoning **up to and including** thought `t`
- Embed that text with a pretrained embedding model
- Output: one vector per thought

So instead of:

```text
question + full_solution_text
```

we produce:

```text
question + [z_1, z_2, z_3, ..., z_T]
```

Where:

- `T = number of thoughts`
- `z_t` is the embedding of the reasoning state after consuming thought `t`

This phase is not training.
It is a dataset transformation / representation extraction phase.

## 0) What This Phase Must Do

Given a record with:

- `question: str`
- `solution: str`
- optionally `answer` / `final_answer`

and a function:

- `split_thoughts(solution) -> List[str]`

this phase must:

- Split the solution into ordered thoughts.
- For each thought step `t`, construct a prefix-style reasoning input that contains:
  - the question
  - the previous reasoning
  - the current thought
- Encode that input with a pretrained embedding model.
- Store the resulting vector as `z_t`.
- Return the same dataset semantics, but replace textual solution reasoning with:
  - `thoughts`
  - `state_vectors`

The final output is a dataset where each sample contains a chain of vectors, not a chain of textual thoughts as the main reasoning representation.

## 1) High-Level Design

### 1.1 Target representation

For a solution split into:

- `thought_1`
- `thought_2`
- `thought_3`

we want:

```python
z_1 = embed(question, thought_1)
z_2 = embed(question, thought_1, thought_2)
z_3 = embed(question, thought_1, thought_2, thought_3)
```

More precisely, `z_t` should represent:

- the reasoning state after consuming the question and all thoughts up to step `t`

This is not the same as embedding `thought_t` alone.

### 1.2 Main modeling choice

The recommended default is:

- Embedding model: `Qwen/Qwen3-Embedding-0.6B` or `Qwen/Qwen3-Embedding-4B`
- Backend: preferably `vLLM` embedding mode
- Readout: model-native embedding output
- Input format: instruction-aware query text

Reasoning:

- Qwen3 Embedding is a real embedding model.
- It supports long context.
- It is instruction-aware.
- It is compatible with `vLLM` pooling / embedding workflows.
- It avoids ad hoc custom `[STATE]` token tricks that would require training.

### 1.3 Core principle

The main vector for step `t` should come from the entire reasoning prefix so far, while still making the current thought explicit in the prompt structure.

Recommended structure:

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
{question}

Previous reasoning:
{thought_1}
...
{thought_{t-1}}

Current step:
{thought_t}
```

This preserves:

- cumulative state
- explicit focus on the new thought
- no future leakage

## 2) Configuration Contract (Single Source of Truth)

The phase should be driven by a config object such as:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThoughtEmbeddingConfig:
    # Data
    dataset_name: str = "your_dataset_name"
    train_split: str = "train"
    eval_split: Optional[str] = None
    input_question_field: str = "question"
    input_solution_field: str = "solution"
    input_answer_field: Optional[str] = "answer"

    # Model
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    backend: str = "vllm"   # "vllm" | "transformers"
    dtype: str = "bfloat16"
    max_model_len: int = 32768

    # Embedding behavior
    use_instruction: bool = True
    instruction_text: str = (
        "Represent the current reasoning state of a math solution prefix, "
        "given the problem, prior reasoning, and the current step."
    )
    include_question: bool = True
    include_previous_reasoning_header: bool = True
    include_current_step_header: bool = True

    # Optional extra embedding
    emit_step_vectors: bool = False
    step_instruction_text: str = (
        "Represent the meaning of this single reasoning step in a math solution."
    )

    # Batching / performance
    batch_size: int = 64
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 128

    # Output
    output_dir: str = "runs/thought_embedding"
    output_format: str = "parquet"   # "parquet" | "jsonl"
    shard_size: int = 5000
    save_float_dtype: str = "float16"  # "float16" | "float32"

    # Reliability / resuming
    save_every_n_examples: int = 1000
    resume: bool = True
    overwrite: bool = False

    # Filtering / truncation
    drop_empty_thoughts: bool = True
    min_thoughts: int = 1
    max_thoughts_per_example: Optional[int] = None
    skip_overlong_examples: bool = False
    truncate_overlong_examples: bool = True

    # Logging
    log_every_n_examples: int = 100
    seed: int = 42
```

Everything in the pipeline should map cleanly onto this config.

## 3) Data Contract

### 3.1 Input dataset contract

Each input row is expected to contain at least:

- `question: str`
- `solution: str`

Optional but recommended:

- `answer`
- `id` / `qid`
- metadata fields from upstream generation

If the configured question or solution field is missing, the pipeline must raise a clear error.

### 3.2 Output dataset contract

Each output row should contain:

- `question: str`
- `thoughts: List[str]`
- `num_thoughts: int`
- `state_vectors: List[List[float]]`

Optional fields:

- `answer`
- `solution` (kept only if explicitly requested)
- `step_vectors: List[List[float]]` if `emit_step_vectors=True`
- `id` / `qid`
- token-length metadata
- truncation flags

Recommended output schema:

```python
{
    "id": str | None,
    "question": str,
    "answer": str | int | None,
    "thoughts": List[str],
    "num_thoughts": int,
    "state_vectors": List[List[float]],
    "embedding_dim": int,
    "model_name": str,
    "prompt_version": str,
    "was_truncated": bool,
}
```

If optional step vectors are enabled:

```python
{
    ...
    "step_vectors": List[List[float]],
}
```

### 3.3 Semantics of state_vectors

`state_vectors[t]` must correspond to:

- the reasoning state after thought `t+1`

So:

- `state_vectors[0] = after first thought`
- `state_vectors[1] = after second thought`
- etc.

Length invariant:

- `len(state_vectors) == len(thoughts) == num_thoughts`

This invariant must always hold.

## 4) Thought Splitting Contract

This phase depends on a function:

- `split_thoughts(solution: str) -> List[str]`

The splitter is assumed to exist upstream or in this phase.

### 4.1 Requirements

`split_thoughts` must return:

- ordered thoughts
- each thought as text
- no future knowledge leakage across ordering
- deterministic output for the same input string

### 4.2 Post-processing rules

After splitting:

- Strip leading/trailing whitespace from each thought.
- If `drop_empty_thoughts=True`, remove empty thoughts.
- Recompute `num_thoughts`.
- If `num_thoughts < min_thoughts`, skip the example.
- If `max_thoughts_per_example` is set, truncate or skip according to config.

Recommended cleaning:

```python
thoughts = [t.strip() for t in split_thoughts(solution)]
if cfg.drop_empty_thoughts:
    thoughts = [t for t in thoughts if t]
```

## 5) Prompt / Text Construction

### 5.1 Why prompt design matters

We do not want to encode only `thought_t` by itself.
We want the full prefix state.

But we also do not want the current thought to be lost inside a long block of earlier text.

So the prompt must do both:

- preserve the reasoning prefix
- explicitly highlight the current step

### 5.2 Default query builder

For thought index `t`, build:

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
{question}

Previous reasoning:
{thought_1}
...
{thought_{t-1}}

Current step:
{thought_t}
```

Special handling:

- for `t = 1`, `Previous reasoning:` may be omitted or left empty
- `Current step:` should always be present

This is the default and recommended main representation.

### 5.3 First thought special case

For the first thought:

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
{question}

Current step:
{thought_1}
```

Do not insert an empty `Previous reasoning:` block unless formatting logic prefers consistency.

### 5.4 Optional auxiliary step-only prompt

If `emit_step_vectors=True`, also build:

```text
Instruct: Represent the meaning of this single reasoning step in a math solution.
Query: Question:
{question}

Current step:
{thought_t}
```

Important:

- this auxiliary vector is not the main state vector
- it should be stored separately
- it should not be averaged into the state vector by default

### 5.5 Why not average prefix vector and thought-only vector

Do not use:

```text
z_t = 0.5 * state_vector_t + 0.5 * step_vector_t
```

as the default.

Reason:

- they represent different semantic objects
- one is cumulative state
- one is local meaning

If both are needed, store both separately or concatenate later in downstream experiments.

## 6) Embedding Model Contract

### 6.1 Default model choices

Recommended order:

1. `Qwen/Qwen3-Embedding-0.6B`
2. `Qwen/Qwen3-Embedding-4B`

Use `0.6B` when:

- throughput matters most
- very large-scale dataset conversion is the priority

Use `4B` when:

- quality matters more
- hardware budget allows slower processing

### 6.2 Backend choices

Supported backends should be abstracted behind one interface:

- `vllm`
- `transformers`

Recommended default:

- `vllm`

because it is operationally simpler for high-throughput serving/inference workflows.

### 6.3 Encoder abstraction

The rest of the pipeline must not know backend details.

Define a single interface such as:

```python
class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
```

Possible implementations:

- `VLLMEmbedder`
- `TransformersEmbedder`

The pipeline should call only:

```python
vectors = embedder.embed_texts(texts)
```

## 7) Pipeline Logic

### 7.1 Per-example flow

For each dataset row:

- Read question
- Read solution
- Split into thoughts
- Build one main query text per thought
- Optionally build one auxiliary step-only text per thought
- Batch these texts
- Embed them
- Assemble output row
- Write to shard / file

### 7.2 Per-thought flow

For each thought index `t`:

- Build prefix-aware prompt text
- Send to embedding model
- Receive vector
- Cast vector to configured save dtype
- Append to `state_vectors`

### 7.3 Batching strategy

Batch texts across examples when possible.

Do not insist on embedding all thoughts for one example before moving on to the next if cross-example batching improves throughput.

A practical strategy:

- accumulate prompt texts in a queue
- embed in batches
- map outputs back to `(example_id, thought_idx, vector_type)`

Suggested internal record:

```python
{
    "row_id": int,
    "thought_idx": int,
    "kind": "state" | "step",
    "text": str,
}
```

This decouples text construction from embedding execution.

### 7.4 Output assembly

Maintain a temporary per-row structure like:

```python
{
    "question": ...,
    "answer": ...,
    "thoughts": [...],
    "state_vectors": [None] * T,
    "step_vectors": [None] * T,   # optional
}
```

After embedding results return, fill by index.

Only write a row when:

- all required vectors are present
- vector count matches thought count

## 8) Handling Long Examples

Some examples may be very long.

### 8.1 Context limit reality

Even with a long-context model, cumulative prefix prompts can become large.

The pipeline must explicitly handle overlong inputs.

### 8.2 Policy options

Config options:

- `skip_overlong_examples=True`
- or `truncate_overlong_examples=True`

Recommended default:

- `truncate_overlong_examples=True`

but do it carefully.

### 8.3 Truncation policy

If a prompt exceeds max input length:

Preferred truncation order:

- preserve:
  - question
  - current step
- keep as much of recent previous reasoning as possible
- drop the oldest previous thoughts first

So truncation should be left-truncation over previous reasoning, not truncation of the current step.

This preserves the latest reasoning transition.

Pseudo-policy:

- keep full question
- keep full current step
- include recent previous thoughts from newest to oldest until token budget is full

This is better than naïvely truncating the end.

### 8.4 Metadata for truncation

If truncation occurs, record:

- `was_truncated=True`
- `num_previous_thoughts_kept`
- optional token counts

This makes later analysis possible.

## 9) Saving Format

### 9.1 Preferred formats

Recommended:

- `parquet` for structured storage
- optionally Arrow / Hugging Face Dataset export after processing

Use:

- `float16` for vectors by default to reduce storage
- `float32` only if needed for analysis fidelity

### 9.2 Row layout recommendation

Recommended one-row-per-example output:

```python
{
    "id": ...,
    "question": ...,
    "answer": ...,
    "thoughts": [...],
    "num_thoughts": ...,
    "state_vectors": [[...], [...], ...],
    "embedding_dim": ...,
    "model_name": ...,
    "prompt_version": ...,
    "was_truncated": ...,
}
```

Optional alternative:

- one row per thought step

Example:

```python
{
    "id": ...,
    "question": ...,
    "thought_idx": 7,
    "thought_text": ...,
    "state_vector": [...],
}
```

Default recommendation:

- one row per example

because it matches downstream “chain of vectors” semantics better.

### 9.3 Sharding

For large datasets, write shards:

```text
part-00000.parquet
part-00001.parquet
...
```

A shard should contain only complete rows.

Do not leave partially written rows in final output shards.

## 10) Reliability, Resume, and Idempotency

### 10.1 Resume support

The pipeline should support interruption and resume.

Recommended behavior:

- after every `save_every_n_examples`, flush complete rows
- maintain a progress file / manifest of completed example IDs or row indices
- on resume, skip already processed examples

### 10.2 Overwrite behavior

If output exists:

- if `overwrite=True`, delete and rebuild
- else if `resume=True`, continue
- otherwise raise a clear error

### 10.3 Determinism

This phase should be deterministic up to embedding backend nondeterminism.

Deterministic parts include:

- thought splitting
- text formatting
- output ordering
- sharding policy

## 11) Recommended Software Architecture

Use a small folder package, not a single large script.

Recommended structure:

```text
thought_embedding/
    __init__.py
    config.py
    prompts.py
    splitter.py
    encoder.py
    pipeline.py
    io_utils.py
    main.py
```

### 11.1 Responsibilities

`config.py`

- config dataclass
- config loading / validation

`prompts.py`

- build main state prompt
- build optional step prompt
- token-budget-aware truncation helpers

`splitter.py`

- wrapper around `split_thoughts`
- thought cleanup / filtering

`encoder.py`

- `Embedder` interface
- `VLLMEmbedder`
- `TransformersEmbedder`

`pipeline.py`

- main orchestration
- queue batching
- output assembly
- progress management

`io_utils.py`

- save/load parquet/jsonl
- shard writing
- manifest handling

`main.py`

- CLI entrypoint

## 12) CLI Expectations

The phase should be runnable from CLI, for example:

```bash
python -m thought_embedding.main \
  --dataset_name omrisap/my_math_dataset \
  --train_split train \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --backend vllm \
  --output_dir runs/thought_embedding_run1
```

Optional arguments:

- batch size
- shard size
- max model len
- save dtype
- emit step vectors
- resume / overwrite
- prompt version

## 13) Validation Rules

Before writing any row, validate:

- question is non-empty
- thoughts is non-empty
- `len(state_vectors) == len(thoughts)`
- all vectors have same dimensionality
- no vector is missing
- `num_thoughts == len(thoughts)`

If any check fails:

- log the issue
- skip the row or raise depending on severity/config

## 14) Non-Goals

This phase does not do:

- model training
- fine-tuning
- RL
- vector quantization
- latent token learning
- evaluation of reasoning quality
- next-thought prediction

It is strictly:

- convert textual worked solutions into sequential vector states

## 15) Downstream Role in LMMS

This phase is intended to support later stages where reasoning may be modeled as:

- continuous latent trajectories
- clustered states
- vector-quantized reasoning states
- discrete code sequences
- compressed solution representations

This phase therefore must produce outputs that are:

- sequential
- fixed-size per thought
- easy to save/load
- easy to feed into future modeling stages

## 16) Canonical Example

Input:

```python
{
    "question": "If 3 apples cost 12 dollars, how much do 5 apples cost?",
    "solution": (
        "First find the price of one apple. "
        "12 divided by 3 is 4. "
        "Then multiply by 5. "
        "5 times 4 is 20."
    ),
    "answer": 20,
}
```

Assume:

```python
thoughts = [
    "First find the price of one apple.",
    "12 divided by 3 is 4.",
    "Then multiply by 5.",
    "5 times 4 is 20.",
]
```

Constructed prompts:

`z_1`

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
If 3 apples cost 12 dollars, how much do 5 apples cost?

Current step:
First find the price of one apple.
```

`z_2`

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
If 3 apples cost 12 dollars, how much do 5 apples cost?

Previous reasoning:
First find the price of one apple.

Current step:
12 divided by 3 is 4.
```

`z_3`

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
If 3 apples cost 12 dollars, how much do 5 apples cost?

Previous reasoning:
First find the price of one apple.
12 divided by 3 is 4.

Current step:
Then multiply by 5.
```

`z_4`

```text
Instruct: Represent the current reasoning state of a math solution prefix, given the problem, prior reasoning, and the current step.
Query: Question:
If 3 apples cost 12 dollars, how much do 5 apples cost?

Previous reasoning:
First find the price of one apple.
12 divided by 3 is 4.
Then multiply by 5.

Current step:
5 times 4 is 20.
```

Output:

```python
{
    "question": "If 3 apples cost 12 dollars, how much do 5 apples cost?",
    "answer": 20,
    "thoughts": [
        "First find the price of one apple.",
        "12 divided by 3 is 4.",
        "Then multiply by 5.",
        "5 times 4 is 20.",
    ],
    "num_thoughts": 4,
    "state_vectors": [
        z_1,
        z_2,
        z_3,
        z_4,
    ],
    "embedding_dim": D,
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "prompt_version": "v1_prefix_prev_current",
    "was_truncated": False,
}
```

## 17) Final Implementation Guidance for the Code Agent

The implementation should optimize for:

- correctness
- clarity
- resumability
- easy backend swapping
- future compatibility with later LMMS phases

The code agent should follow these priorities:

- build a clean config object
- isolate prompt-building logic
- isolate embedding backend behind one interface
- make the pipeline resumable
- enforce output invariants strictly
- keep the output dataset simple and stable

Most important semantic rule:

> The main vector for thought `t` must represent the reasoning state after consuming the prefix up to `t`, while making the current thought explicit in the input formatting.

That is the core contract of this phase.
