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
   - The model never generates natural language answers or explanations.
   - Tokens are treated as *latent control actions*, not as communicative text.

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

### **Policy Model**
- A pretrained math-focused LLM (e.g. Qwen-Math-1.5B)
- Augmented with:
  - **Latent action tokens** (`<z*>`) used exclusively during reasoning
  - **Answer classification head** producing discrete numeric answers
  - **Value head** for PPO

The language model backbone is used only for **encoding and latent state transitions**.
No language is ever decoded as output.

---

### **Environment**
A Gym-like environment that:

- Enforces interaction protocols during training
- Applies phase-dependent action masking
- Inserts training scaffolds where needed
- Computes rewards **only from final answer correctness**

The environment owns protocol correctness but never performs reasoning itself.

---

### **PPO Trainer**
- Optimizes the policy using **Proximal Policy Optimization (PPO)**
- Treats:
  - latent tokens as discrete actions
  - the answer as a discrete classification action
- Uses sparse, delayed rewards aligned with evaluation metrics

---

## Interaction Protocol (Training vs. End State)

This project explicitly distinguishes between **training scaffolds** and the **intended
end-state behavior**.

---

## Phase 1: Enforced Protocol (Training Scaffold)

In the initial training phase, the environment enforces a strict interaction protocol
designed to bootstrap latent reasoning and policy control.

For each training episode:

1. The environment provides a natural language math question.
2. A latent step count `K` is sampled uniformly from a predefined range
   (e.g. `K ∈ [K_min, K_max]`).
3. The policy emits exactly `K` latent actions, where:
   - only `<z*>` tokens are allowed
   - all other tokens are masked out
4. After the final latent step, the environment transitions to the **answer phase**.
5. The policy emits **one discrete answer action**:
   - sampled from a **classification head**
   - action space: `{0, 1, 2, ..., 9}` (or larger ranges in later phases)
6. The episode terminates immediately after the answer action.
7. A reward is assigned:
   - `+1` if the predicted answer equals the labeled answer
   - `0` otherwise

### Key properties of Phase 1

- The answer is **not a token**
- The answer does **not** affect the input sequence
- The answer is a **pure control action**
- The reward depends only on the final answer

Only policy-sampled actions (latent actions + answer classification)
participate in PPO optimization.

---

## Training Scaffolding Notes

Several constraints in Phase 1 are **deliberate training scaffolds**:

- The number of latent steps `K` is externally controlled
- The answer is emitted via a dedicated classification head
- The episode terminates immediately after answering

These constraints exist solely to stabilize early learning and ensure
clear credit assignment between latent reasoning and final answers.

They are **not** part of the intended end-state behavior.

---

## End State: Fully Policy-Controlled Reasoning

The intended final behavior of the system is:

- The policy decides **when** to stop reasoning
- The policy decides **how many** latent steps to take
- The policy decides **when** to answer
- The policy produces the answer directly, without environment-imposed structure

In the end state:

> **All control over reasoning depth and answer emission belongs to the policy.**

The environment becomes a passive evaluator rather than a protocol enforcer.

---

## Latent Action Tokens (`<z*>`)

- `<z*>` tokens represent **latent micro-actions**
- They have no predefined semantics and are not natural language
- Their meaning emerges entirely through reinforcement learning

They are never decoded, logged, or interpreted linguistically.

---

## Learning Algorithm

Training uses **Proximal Policy Optimization (PPO)**:

- Latent tokens and answer classifications are treated as actions
- Transformer hidden states are treated as environment states
- Rewards are sparse and delayed
- A shared-backbone value head estimates expected final reward

Early phases train only a subset of model parameters to preserve pretrained
language understanding. Additional layers may be unfrozen gradually.

---

## Curriculum Learning Strategy

Training proceeds through increasingly difficult stages:

### Phase 1
- Very simple arithmetic
- Small answer spaces (e.g. digits `0–9`)
- Fixed latent step count range
- Environment-controlled termination
- Goal: learn latent action protocol and deferred answering

### Later Phases (Planned)
- Larger answer spaces
- Learned termination policies
- Variable and longer reasoning horizons
- Harder word problems
- Removal of all environment-enforced structure
- Optional hierarchical latent actions

---

## Output Representation

The system **does not generate textual output**.

Answers are represented as **discrete actions** sampled from a classification head.
No decoding, formatting, or token-level output is required.

---

## Non-Goals

This project intentionally does **not** aim to:

- Generate human-readable reasoning traces
- Produce chain-of-thought explanations
- Optimize interpretability of intermediate steps
- Perform standard supervised fine-tuning (SFT)

Any of the above would violate the core design principles.

---

## Repository Philosophy

This repository is intended to be used for the **entire lifetime of the project**.

The README is written to:

- Clearly separate temporary training scaffolds from permanent principles
- Remain correct as training protocols evolve
- Be unambiguous for automated agents modifying the code

When in doubt:

> **Favor latent state, policy control, and minimal supervision.**

---

## Future Directions (Non-Exhaustive)

- Learned termination policies
- Larger answer spaces
- Hierarchical latent action systems
- Verifier-guided rewards
- Scaling to competition-level math benchmarks
