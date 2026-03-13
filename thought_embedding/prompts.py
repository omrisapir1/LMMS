from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from thought_embedding.config import ThoughtEmbeddingConfig


class PromptError(RuntimeError):
    pass


@dataclass
class PromptBuildResult:
    text: str
    was_truncated: bool
    num_previous_thoughts_kept: int
    token_count: int


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_state_prompt(
    cfg: ThoughtEmbeddingConfig,
    question: str,
    previous_thoughts: Sequence[str],
    current_thought: str,
) -> str:
    blocks: list[str] = []

    if cfg.use_instruction:
        blocks.append(f"Instruct: {cfg.instruction_text}")

    blocks.append("Query:")

    if cfg.include_question:
        blocks.append("Question:")
        blocks.append(question)

    if cfg.include_previous_reasoning_header and previous_thoughts:
        blocks.append("Previous reasoning:")
        blocks.extend(previous_thoughts)

    if cfg.include_current_step_header:
        blocks.append("Current step:")
    blocks.append(current_thought)

    return "\n\n".join(blocks)


def build_state_prompt_for_thought(
    cfg: ThoughtEmbeddingConfig,
    question: str,
    thoughts: Sequence[str],
    thought_idx: int,
    tokenizer: Any,
) -> PromptBuildResult:
    if thought_idx < 0 or thought_idx >= len(thoughts):
        raise PromptError(f"Invalid thought index {thought_idx} for {len(thoughts)} thoughts.")

    previous = list(thoughts[:thought_idx])
    current = thoughts[thought_idx]

    prompt = render_state_prompt(cfg, question, previous, current)
    tokens = _token_count(tokenizer, prompt)
    if tokens <= cfg.max_model_len:
        return PromptBuildResult(
            text=prompt,
            was_truncated=False,
            num_previous_thoughts_kept=len(previous),
            token_count=tokens,
        )

    if not cfg.truncate_overlong_examples:
        raise PromptError(
            f"Prompt for thought index {thought_idx} exceeds max_model_len={cfg.max_model_len}."
        )

    # Preserve full question and current step; pack as many recent previous thoughts as possible.
    base_prompt = render_state_prompt(cfg, question, [], current)
    base_tokens = _token_count(tokenizer, base_prompt)
    if base_tokens > cfg.max_model_len:
        raise PromptError(
            "Question + current thought exceed max_model_len; cannot satisfy truncation policy "
            "without violating the preserve-question/current-step rule."
        )

    kept_suffix: list[str] = []
    for thought in reversed(previous):
        candidate = [thought] + kept_suffix
        candidate_prompt = render_state_prompt(cfg, question, candidate, current)
        candidate_tokens = _token_count(tokenizer, candidate_prompt)
        if candidate_tokens <= cfg.max_model_len:
            kept_suffix = candidate
        else:
            break

    truncated_prompt = render_state_prompt(cfg, question, kept_suffix, current)
    truncated_tokens = _token_count(tokenizer, truncated_prompt)
    return PromptBuildResult(
        text=truncated_prompt,
        was_truncated=True,
        num_previous_thoughts_kept=len(kept_suffix),
        token_count=truncated_tokens,
    )
