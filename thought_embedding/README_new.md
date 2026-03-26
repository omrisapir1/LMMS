# Phase X — Thought-State Embedding Dataset Builder (Single-Pass Decoder)

This phase converts a dataset of **math questions with step-by-step solutions** into a dataset where each solution is represented as a **sequence of dense vectors (one per thought)**.

Unlike previous designs, this phase uses:

- **one forward pass per example**
- a **causal decoder model**
- **no prefix re-encoding**
- **no embedding model**

---

## 🔥 Core Idea

Given:

- `problem: str`
- `splitted_solution: List[str]`  (already precomputed thoughts)

We construct a single sequence:

```
user: Solve the following math problem...

assistant: \n\nthought_1\n\nthought_2\n\n...\n\nthought_T
```

Then:

1. Tokenize full conversation (using model chat template)
2. Run **one forward pass**
3. Extract hidden states at **end of each thought**
4. Return:

```
[z_1, z_2, ..., z_T]
```

Where:

- `z_t` = hidden state after thought `t`
- represents **reasoning state after consuming thoughts 1..t**

---

## ✅ Key Properties

- **O(1) forward pass per example**
- Fully **causal** (no future leakage)
- Much faster than prefix embedding (~50–100x)
- Produces true **latent reasoning trajectory**

---

## 1) Model Contract

### Default model

- `nvidia/OpenMath-Nemotron-1.5B`

### Loading

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
```

### Requirements

- Use `output_hidden_states=True`
- Use `torch.bfloat16`
- Use `device_map="auto"`

---

## 2) Input Dataset Contract

Each row must contain:

```python
{
    "problem": str,
    "splitted_solution": List[str],
    "answer": Optional[str | int],
    "id": Optional[str],
}
```

---

## 3) Prompt Construction

### User Message

```
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

{problem}
```

### Assistant Content

```
\n\n{thought_1}\n\n{thought_2}\n\n...\n\n{thought_T}
```

Rules:

- MUST start with `\n\n`
- separator is always `\n\n`
- NO trailing separator after last thought
- exactly **T thoughts → T vectors**

---

## 4) Token Boundary Tracking (CRITICAL)

Track token positions during construction.

Output:

```
thought_token_end_positions: List[int]
```

---

## 5) Forward Pass

```python
outputs = model(
    input_ids,
    output_hidden_states=True,
    use_cache=False,
)

hidden = outputs.hidden_states[-1]
```

---

## 6) Vector Extraction

```
z_t = hidden[batch_idx, thought_token_end_positions[t]]
```

---

## 7) Output Format

```python
{
    "question": str,
    "thoughts": List[str],
    "state_vectors": List[List[float]],
    "embedding_dim": int,
}
```

---

## 🚀 Summary

Single forward pass → extract hidden states → get reasoning trajectory.
