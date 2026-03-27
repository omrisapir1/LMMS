from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThoughtEmbeddingConfig:
    # Data
    dataset_name: str = "omrisap/nvidia_math_generated_solution_750K"
    train_split: str = "train"
    eval_split: Optional[str] = None
    input_problem_field: str = "problem"
    input_thoughts_field: str = "splitted_solution"
    input_answer_field: Optional[str] = "answer"
    input_expected_answer_field: Optional[str] = "expected_answer"
    input_solution_field: Optional[str] = "generated_solution"
    input_source_field: Optional[str] = "source"
    input_id_field: Optional[str] = "id"
    input_qid_field: Optional[str] = "qid"
    max_samples: int = 0
    source_filter: Optional[str] = None  # None means use all sources

    # Model
    model_name: str = "nvidia/OpenMath-Nemotron-1.5B"
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    max_model_len: int = 32768

    # Prompt / construction
    user_prompt_template: str = (
        "Solve the following math problem. Make sure to put the answer "
        "(and only answer) inside \\boxed{{}}.\n\n{problem}"
    )
    separator_text: str = "\n\n"

    # Batching
    max_tokens_per_batch: int = 16_384
    max_examples_per_batch: int = 8
    sort_by_length: bool = True

    # Parallel preprocessing
    pretokenize_num_proc: int = 15
    pretokenize_batch_size: int = 128*10

    # Output
    output_dir: str = "runs/thought_embedding"
    output_format: str = "parquet"  # only parquet in v1
    shard_size: int = 5000
    save_float_dtype: str = "float16"  # "float16" | "float32"
    keep_solution: bool = False

    # Reliability / resume
    save_every_n_examples: int = 1000
    resume: bool = True
    overwrite: bool = False

    # Filtering
    drop_empty_thoughts: bool = True
    min_thoughts: int = 1

    # Logging
    log_every_n_examples: int = 100
    seed: int = 42


class ConfigError(ValueError):
    pass


def validate_config(cfg: ThoughtEmbeddingConfig) -> None:
    if cfg.output_format != "parquet":
        raise ConfigError(
            f"Unsupported output_format '{cfg.output_format}'. Only 'parquet' is implemented in v1."
        )
    if cfg.save_float_dtype not in {"float16", "float32"}:
        raise ConfigError("save_float_dtype must be 'float16' or 'float32'.")
    if cfg.dtype not in {"bfloat16", "float16", "float32"}:
        raise ConfigError("dtype must be one of: bfloat16, float16, float32.")
    if cfg.max_samples < 0:
        raise ConfigError("max_samples must be >= 0.")
    if cfg.max_tokens_per_batch <= 0:
        raise ConfigError("max_tokens_per_batch must be positive.")
    if cfg.max_examples_per_batch <= 0:
        raise ConfigError("max_examples_per_batch must be positive.")
    if cfg.pretokenize_num_proc <= 0:
        raise ConfigError("pretokenize_num_proc must be positive.")
    if cfg.pretokenize_batch_size <= 0:
        raise ConfigError("pretokenize_batch_size must be positive.")
    if cfg.shard_size <= 0:
        raise ConfigError("shard_size must be positive.")
    if cfg.save_every_n_examples <= 0:
        raise ConfigError("save_every_n_examples must be positive.")
    if cfg.log_every_n_examples <= 0:
        raise ConfigError("log_every_n_examples must be positive.")
    if cfg.max_model_len <= 0:
        raise ConfigError("max_model_len must be positive.")
    if cfg.min_thoughts <= 0:
        raise ConfigError("min_thoughts must be positive.")
