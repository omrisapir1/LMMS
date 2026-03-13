from __future__ import annotations

import argparse
import json

from thought_embedding.config import ThoughtEmbeddingConfig
from thought_embedding.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build thought-state embedding datasets")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--train_split", default="train")
    p.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--backend", default="vllm")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=32768)

    p.add_argument("--input_question_field", default="question")
    p.add_argument("--input_solution_field", default="solution")
    p.add_argument("--input_answer_field", default="answer")
    p.add_argument("--input_expected_answer_field", default="expected_answer")
    p.add_argument("--input_id_field", default="id")
    p.add_argument("--input_qid_field", default="qid")
    p.add_argument("--max_samples", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_num_seqs", type=int, default=128)

    p.add_argument("--output_dir", default="runs/thought_embedding")
    p.add_argument("--output_format", default="parquet")
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--save_float_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--prompt_version", default="v1_prefix_prev_current")
    p.add_argument("--keep_solution", action="store_true")

    p.add_argument("--save_every_n_examples", type=int, default=1000)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--drop_empty_thoughts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_thoughts", type=int, default=1)
    p.add_argument("--max_thoughts_per_example", type=int)
    p.add_argument("--skip_overlong_examples", action="store_true")
    p.add_argument("--truncate_overlong_examples", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--splitter_min_chars", type=int, default=100)
    p.add_argument("--splitter_max_chars", type=int, default=300)

    p.add_argument("--log_every_n_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_args() -> ThoughtEmbeddingConfig:
    args = build_parser().parse_args()
    return ThoughtEmbeddingConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
