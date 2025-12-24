# Latent Markov Math Solver (LMMS)

A research project exploring **non-linguistic, latent, reinforcement-learned reasoning**
for mathematical problem solving.

---

## Motivation

Most current approaches to mathematical reasoning with large language models rely on
natural language generation (e.g. Chain-of-Thought, Tree-of-Thought).
While effective, these approaches impose constraints that are unnecessary in settings
where **only the final answer matters**:

- The model must generate linguistically coherent text
- Reasoning is entangled with grammar, formatting, and decoding artifacts
- Training and inference are sensitive to prompt and tokenization details

This project explores an alternative paradigm:

> **Latent, iterative reasoning without language decoding**, where a model reasons through
> internal state transitions and is trained directly on final answer correctness using
> reinforcement learning.

---

## Core Design Principles

This repository is built around three non-negotiable principles:

1. **No language decoding**
   - The model does not generate natural language explanations.
   - Tokens are treated as *latent actions*, not as communicative text.

2. **Iterative Markov reasoning**
   - Reasoning is performed through a sequence of latent steps.
   - Each step updates an internal state based only on the current state and the input.

3. **Strong pretrained language understanding**
   - Natural language questions are encoded using a large pretrained math-focused LLM.
   - Language understanding is inherited, not re-learned.

These principles apply throughout the entire lifetime of the project, even as training
protocols evolve.

---

## High-Level Architecture

The system consists of three conceptual components:

- **Policy Model**
  - A pretrained math LLM (Qwen-Math-1.5B) augmented with:
    - latent action tokens (`<z*>`)
    - a value head for PPO
  - Language understanding is retained internally, but language is never decoded.

- **Environment**
  - A Gym-like environment that:
    - enforces interaction protocols and structural constraints during training
    - applies step-dependent action masking
    - inserts scaffold tokens during early phases
    - computes rewards based solely on final answer correctness

- **PPO Trainer**
  - Optimizes the policy using Proximal Policy Optimization (PPO)
  - Uses sparse, delayed rewards aligned with evaluation metrics

---

## Interaction Protocol (Training vs. End State)

This project explicitly distinguishes between **training scaffolds** and the **intended
end-state behavior**.

### Phase 1: Enforced Protocol (Training Scaffold)

In the initial training phase, the environment enforces a structured interaction
protocol designed to bootstrap latent reasoning and policy control.

For each training episode:

1. The environment provides a natural language math question.
2. A latent step count `K` is sampled uniformly from a predefined range
   (e.g. `K ∈ [K_min, K_max]`).
3. The policy emits exactly `K` latent action tokens, where:
   - only `<z*>` tokens are allowed
   - all other tokens are masked out
4. The environment inserts an answer scaffold beginning with:
```</answer>```
5. Based on the known length of the labeled answer, the environment inserts a number
   of leading `0` tokens such that the **total numeric answer width is fixed to 5 digits**.
6. The policy emits the remaining digit tokens required to complete the 5-digit answer
   representation, with:
   - the action space restricted to `{0..9}`
7. A reward is assigned:
   - `+1` if the predicted number exactly matches the labeled answer
   - `0` otherwise

Examples of Phase-1 answer formats:

```</answer>``` 0 0 0 0 d1
```</answer>``` 0 0 0 d1 d2
```</answer>``` 0 0 d1 d2 d3

Only policy-sampled tokens (latent actions and digit tokens) participate in PPO
optimization.

---

### Phase-1 Scaffolding Notes

The use of labeled answer length to determine the number of leading `0` tokens is a
deliberate Phase-1 training scaffold. It simplifies early learning by decoupling
latent reasoning from output-length prediction.

Similarly, the emission of the `</answer>` token and the termination of latent steps
are controlled by the environment during this phase to enforce a minimum reasoning
depth.

These constraints are **temporary** and exist solely to stabilize early learning.

---

### End State: Fully Policy-Controlled Protocol

The intended final behavior of the system is:

- The policy decides when to emit `</answer>`.
- The policy decides how many latent steps to perform, as in a standard autoregressive
  language model.
- The policy generates all answer digits, without environment-provided padding.
- The environment no longer inserts scaffold tokens.

In later phases, answers are generated directly in fixed-width form:
</answer> d1 d2 d3 d4 d5

In the end state:

> **All control over reasoning depth and answer emission belongs to the policy.**

---

## Latent Action Tokens (`<z*>`)

- `<z*>` tokens represent **latent micro-actions**.
- They have no predefined semantics and are not natural language.
- Their meaning emerges entirely through reinforcement learning.

At early stages, only micro tokens are used. Hierarchical or macro-action structures
may be introduced later, but are not yet committed to.

---

## Learning Algorithm

Training uses **Proximal Policy Optimization (PPO)**:

- Tokens are treated as actions.
- Transformer hidden states are treated as environment states.
- Rewards are sparse and delayed.
- A shared-backbone value head estimates expected final reward.

Early phases train only a subset of model parameters to preserve pretrained math
capabilities. Additional layers are gradually unfrozen as task difficulty increases.

---

## Curriculum Learning Strategy

Training proceeds through increasingly difficult stages:

### Phase 1

- Very simple arithmetic (answers in the range `0–999`)
- One- to three-digit answers
- Fixed-width (5-digit) output with environment-provided padding
- Latent step count externally controlled and randomized
- Goal: learn the latent action protocol and deferred answering

### Later Phases (Planned)

- Larger numeric ranges
- Fully policy-controlled answer formatting
- Longer and variable reasoning horizons
- Harder word problems
- Learned termination (`</answer>`)
- Optional hierarchical latent actions

---

## Output Representation

All answers are represented in **fixed-width numeric form**:

- Total width: **5 digits**
- Right-aligned
- Left-padded with `0` tokens when necessary

Padding is environment-controlled in early phases and policy-controlled in later
phases.

---

## Non-Goals

This project intentionally does **not** aim to:

- Generate human-readable reasoning traces
- Optimize for interpretability of intermediate steps
- Use chain-of-thought supervision
- Perform standard supervised fine-tuning (SFT)

Any of the above would violate the core design principles.

---

## Repository Philosophy

This repository is intended to be used for the **entire lifetime of the project**.

The README is written to:

- clearly separate temporary training scaffolds from permanent design principles
- remain correct as training protocols evolve
- be unambiguous for automated agents modifying the code

When in doubt:

> **Favor latent state, policy control, and minimal supervision.**

---

## Future Directions (Non-Exhaustive)

- Learned termination policies
- Hierarchical latent action systems
- Value shaping and verifier-guided rewards
- Scaling to competition-level math benchmarks
