from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThoughtEmbeddingConfig:
    # Data
    dataset_name: str = "your_dataset_name"
    train_split: str = "train"
    eval_split: Optional[str] = None
    input_question_field: str = "problem"
    input_solution_field: str = "generated_solution"
    input_answer_field: Optional[str] = "answer"
    input_expected_answer_field: Optional[str] = "expected_answer"
    input_id_field: Optional[str] = "id"
    input_qid_field: Optional[str] = "qid"

    # Model
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    backend: str = "vllm"  # "vllm"
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

    # Optional extra embedding (kept for forward compatibility, not implemented in v1)
    emit_step_vectors: bool = False
    step_instruction_text: str = (
        "Represent the meaning of this single reasoning step in a math solution."
    )

    # Batching / performance
    batch_size: int = 64 * 200
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 128 * 200

    # Output
    output_dir: str = "runs/thought_embedding"
    output_format: str = "parquet"  # "parquet"
    shard_size: int = 5000
    save_float_dtype: str = "float16"  # "float16" | "float32"
    prompt_version: str = "v1_prefix_prev_current"
    keep_solution: bool = False

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

    # Splitter controls
    splitter_min_chars: int = 100
    splitter_max_chars: int = 300

    # Logging
    log_every_n_examples: int = 100
    seed: int = 42


class ConfigError(ValueError):
    pass


def validate_config(cfg: ThoughtEmbeddingConfig) -> None:
    if cfg.backend != "vllm":
        raise ConfigError(f"Unsupported backend '{cfg.backend}'. Only 'vllm' is implemented in v1.")
    if cfg.output_format != "parquet":
        raise ConfigError(
            f"Unsupported output_format '{cfg.output_format}'. Only 'parquet' is implemented in v1."
        )
    if cfg.save_float_dtype not in {"float16", "float32"}:
        raise ConfigError("save_float_dtype must be 'float16' or 'float32'.")
    if cfg.batch_size <= 0:
        raise ConfigError("batch_size must be positive.")
    if cfg.shard_size <= 0:
        raise ConfigError("shard_size must be positive.")
    if cfg.max_model_len <= 0:
        raise ConfigError("max_model_len must be positive.")
    if cfg.emit_step_vectors:
        raise ConfigError("emit_step_vectors=True is not implemented in v1.")
    if cfg.skip_overlong_examples and cfg.truncate_overlong_examples:
        raise ConfigError(
            "skip_overlong_examples and truncate_overlong_examples cannot both be True."
        )
