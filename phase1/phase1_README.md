# Phase 1 – Latent Distillation (Coconut-Style)

Phase 1 trains a model to **replace explicit textual reasoning (“thoughts”) with latent internal computation**, while predicting answers **only via an `<ANSWER>` action**, without language supervision.

This phase implements **Coconut-style latent execution**:  
latent tokens do **not** correspond to decoded text. Instead, they trigger **internal hidden-state feedback**, allowing the model to perform multi-step reasoning **entirely in latent space**.

Phase 1 is a **fully supervised curriculum** that distills reasoning structure from model-generated thoughts into latent computation, preparing the model for later action-based and RL training.

---

## Motivation

Most reasoning-capable language models rely on generating explicit text (Chain-of-Thought).  
This introduces several problems:

- Reasoning is entangled with language modeling
- Intermediate text can leak answers
- Supervision pressure encourages imitation rather than computation
- Later reinforcement learning becomes unstable

**Phase 1 – Latent Distillation** addresses these issues by:

- Treating reasoning steps as **latent actions**
- Executing reasoning through **hidden-state feedback**, not text
- Training **only on final answer correctness**
- Enforcing delayed answer emission
- Removing all language supervision on reasoning steps

This phase intentionally **does not evaluate or reward reasoning text**.

---

## Input Data Assumptions

Each example consists of:

- A **question**
- A model-generated **answer**, which is split into a list of textual thoughts:
  ```python
  thoughts: List[str]
  ```
- The same base model used in Phase 1 is used to generate the answers and extract the initial thoughts.

---

## Core Training Objective

- Loss is applied only to the answer prediction.
- Question text → masked
- Thought text → masked
- Latent tokens → masked
- <ANSWER> prediction → only source of loss

This ensures:

- No supervision pressure on language
- No reward for copying thoughts
- Reasoning is learned only insofar as it improves the answer

---

## Answer Representation

Phase 1 loads a Phase-0 model that already includes:

- An <ANSWER> token
- A 5-digit classification head (one head per digit)

Example loading code:

```python
from phase_0.model import Phase0Model
from transformers import AutoTokenizer

model = Phase0Model.from_pretrained(
    "your-username/lmms-phase0",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "your-username/lmms-phase0",
)
```

Phase 1 does not modify the answer head or loss definition. Only the reasoning pathway changes.

---

## Latent Execution Mechanism (Coconut)

<latent> tokens are not normal language tokens and are never decoded.

During the model forward pass:

- The sequence is processed up to a <latent> position.
- The model extracts the last hidden state immediately preceding that position.
- That hidden state is fed back directly into the embedding stream, replacing the <latent> token embedding.
- The forward pass continues using this continuous internal state.
- This process repeats for each <latent> token.

As a result:

- Latent tokens act as internal reasoning steps
- No language is generated or decoded
- Reasoning unfolds through iterative hidden-state transitions
- Computation depth is controlled by the number of latent steps

This is the core Coconut mechanism.

This mechanism is implemented entirely inside the model forward pass; the training loop treats <latent> positions as ordinary inputs.

---

## Latent Distillation Curriculum

Let:

- K_i = number of thoughts for example i (capped at 8)
- Global final stage = Stage 8

### Participation Rule

An example with K_i thoughts:

- Appears in Stages 1 … (K_i − 1)
- Skips Stage K_i
- Reappears only in Stage 8

This prevents examples from finishing early and creates a global synchronization point where all examples complete reasoning together.

### Right-to-Left Latent Replacement

Latent reasoning replaces thoughts from right to left to prevent answer leakage.

Stage x (1 ≤ x < 8)

```
Thought₁
Thought₂
...
Thought_{K_i − x}
<latent> <latent> ... <latent>   (x times)
<ANSWER>
```

Stage 8 (Final)

```
<latent> <latent> ... <latent>   (K_i times)
<ANSWER>
```

Properties:

- No <|start-latent|> or <|end-latent|> tokens
- No thought text in Stage 8
- Latents always immediately precede <ANSWER>
- Each <latent> triggers one internal reasoning step

---

## Toy Example

Question

What is 327 × 45?

Extracted thoughts

- Thought₁: 300 × 45 = 13500
- Thought₂: 27 × 45 = 1215
- Thought₃: 13500 + 1215 = 14715

Stage 2 input

```
Thought₁
Thought₂
<latent> <latent>
<ANSWER>
```

Stage 8 input

```
<latent> <latent> <latent>
<ANSWER>
```

Only the <ANSWER> prediction contributes to loss.

---

## Stage Transition Rule

Stage advancement is accuracy-gated using the validation set.

- Evaluation is run once every X batches

Stage → Exit Condition:

- Stage 1 → Validation accuracy ≥ 10%
- Stage 2 → Validation accuracy ≥ 20%
- Stage 3 → Validation accuracy ≥ 30%
- Stage 4 → Validation accuracy ≥ 40%
- Stage 5 → Validation accuracy ≥ 50%
- Stage 6 → Validation accuracy ≥ 60%
- Stage 7 → Validation accuracy ≥ 70%
- Stage 8 → Validation accuracy ≥ 80%

Notes:

- Early-stage accuracy is expected to be low
- Accuracy is used only as a curriculum gate
- Meaningful evaluation occurs primarily at Stage 8

---

## Phase-1 Completion Criterion

Phase 1 is complete when:

- Stage 8 reaches ≥ 80% validation accuracy

No additional conditions are required.

---

## Class Imbalance: Trailing-Zero Downsampling

Answer digits are often imbalanced (e.g., leading zeros).

To prevent gradient domination by zeros:

For each digit position d, compute:

- p0[d] = P(digit_d == 0)

Define:

- keep_prob[d] = min(1.0, target_p0 / p0[d])

During training:

- If digit ≠ 0 → always include loss
- If digit == 0 → include loss with probability keep_prob[d]

This is subsampling, not reweighting:

- Gradients remain unbiased in expectation
- Rare digits are never suppressed
- Evaluation uses the full distribution

---

## Summary

Phase 1 – Latent Distillation (Coconut):

- Replaces textual reasoning with latent internal computation
- Executes reasoning via hidden-state feedback
- Trains exclusively on answer correctness
- Prevents early completion and answer leakage
- Synchronizes reasoning completion across the dataset
- Establishes a clean substrate for later RL and action-based training

This phase intentionally prioritizes structural alignment over performance.

Later phases will build on this latent execution pathway with action-based objectives and reinforcement learning.
