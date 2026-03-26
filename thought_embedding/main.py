from __future__ import annotations

import argparse
import json

from thought_embedding.config import ThoughtEmbeddingConfig
from thought_embedding.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build thought-state embeddings from single-pass decoder states")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--train_split", default="train")

    p.add_argument("--input_problem_field", default="problem")
    p.add_argument("--input_thoughts_field", default="splitted_solution")
    p.add_argument("--input_answer_field", default="answer")
    p.add_argument("--input_expected_answer_field", default="expected_answer")
    p.add_argument("--input_solution_field", default="generated_solution")
    p.add_argument("--input_id_field", default="id")
    p.add_argument("--input_qid_field", default="qid")
    p.add_argument("--max_samples", type=int, default=0)

    p.add_argument("--model_name", default="nvidia/OpenMath-Nemotron-1.5B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_model_len", type=int, default=32768)

    p.add_argument("--user_prompt_template", default=(
        "Solve the following math problem. Make sure to put the answer "
        "(and only answer) inside \\boxed{}.\\n\\n{problem}"
    ))
    p.add_argument("--separator_text", default="\\n\\n")

    p.add_argument("--max_tokens_per_batch", type=int, default=16384)
    p.add_argument("--max_examples_per_batch", type=int, default=8)

    p.add_argument("--output_dir", default="runs/thought_embedding")
    p.add_argument("--output_format", default="parquet")
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--save_float_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--keep_solution", action="store_true")

    p.add_argument("--save_every_n_examples", type=int, default=1000)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--drop_empty_thoughts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_thoughts", type=int, default=1)

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
