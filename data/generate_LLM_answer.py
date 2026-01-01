#!/usr/bin/env python3

import json
import argparse
import pandas as pd
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # ---- Load dataset ----
    rows = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # We only need the question, but keep answer for comparison
    df = df[["id", "question", "answer"]].copy()

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Build prompts ----
    df["prompt"] = df["question"].apply(
        lambda q: build_prompt(tokenizer, q)
    )

    # ---- vLLM ----
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,   # greedy
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    # ---- Batched generation ----
    generated = []

    prompts = df["prompt"].tolist()

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for out in outputs:
            text = out.outputs[0].text
            generated.append(text)

    df["generated_answer"] = generated

    # Optional: drop prompt column
    df.drop(columns=["prompt"], inplace=True)

    # ---- Save ----
    df.to_parquet(args.output_parquet, index=False)
    print(f"Saved results to {args.output_parquet}")


if __name__ == "__main__":
    main()
