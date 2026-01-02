import re
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

ANSWER_TOKEN = "<ANSWER>"

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

BOXED_ANY_REGEX = re.compile(
    r"\\boxed\{([^}]*)}"
)


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def extract_boxed_int(text: str) -> Optional[int]:
    match = BOXED_ANY_REGEX.search(text)
    if match is None:
        return None

    content = match.group(1).strip()

    # Allow spaces, but digits only
    if not content.isdigit():
        return None

    value = int(content)
    if 0 <= value <= 99999:
        return value

    return None


def replace_box_or_int_with_answer_token(text: str, gen_final_answer: str) -> Optional[str]:
    match = BOXED_ANY_REGEX.search(text)
    if match is None:
        return None
    final_answer_index = text.find(gen_final_answer)
    if final_answer_index == -1 or match.start() < final_answer_index:
        text = text[:match.start()] + ANSWER_TOKEN
    else:
        text = text[:final_answer_index] + ANSWER_TOKEN

    # Remove any remaining boxed expressions
    text = BOXED_ANY_REGEX.sub("", text)

    return text


def build_prompt(tokenizer, question: str, answer: str, gen_final_answer: str) -> str:
    # If the answer contains a boxed final value, replace it with the ANSWER token
    processed_answer = replace_box_or_int_with_answer_token(answer, gen_final_answer) or answer

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": processed_answer},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )



def int_to_digit_labels(x: int) -> Dict[str, int]:
    """
    Map integer to 5 digit labels with zero-padding.

    Example:
      42 -> d4=0, d3=0, d2=0, d1=4, d0=2
    """
    s = f"{x:05d}"
    return {
        "d4": int(s[0]),
        "d3": int(s[1]),
        "d2": int(s[2]),
        "d1": int(s[3]),
        "d0": int(s[4]),
    }


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class Phase0Dataset(Dataset):
    """
    Phase 0 dataset:
      - Loads HF dataset
      - Filters rows with invalid boxed answers
      - Builds chat prompt from question and inserts <ANSWER> where model should output final answer
      - Produces digit classification labels
    """

    def __init__(
        self,
        hf_name: str,
        split: str,
        tokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw_ds = load_dataset(hf_name, split=split)

        self.samples = []
        dropped = 0

        for row in raw_ds:
            question = row.get("question")
            gen_answer = row.get("generated_answer")
            gen_final_answer = str(int(row.get("generated_final_answer")))

            if not isinstance(question, str) or not isinstance(gen_answer, str):
                dropped += 1
                continue

            answer = extract_boxed_int(gen_answer)
            if answer is None:
                dropped += 1
                continue

            # Build chat-style prompt, with assistant content containing reasoning
            # and <ANSWER> token where the final answer should be generated
            prompt = build_prompt(self.tokenizer, question, gen_answer, gen_final_answer)

            labels = int_to_digit_labels(gen_final_answer)

            self.samples.append({
                "text": prompt,
                "answer": answer,
                "labels": labels,
                "question": question,
                "generated_final_answer": gen_final_answer,
            })

        if len(self.samples) == 0:
            raise RuntimeError("Phase0Dataset: no valid samples after filtering.")

        print(
            f"[Phase0Dataset] Loaded {len(self.samples)} samples "
            f"(dropped {dropped})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        sample = self.samples[idx]
        if ANSWER_TOKEN not in sample["text"]:
            print(f"sample text: {sample['text']}")
            raise RuntimeError("Dataset sample missing <ANSWER> before tokenization")

        encoded = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Flatten batch dimension
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        # Digit labels as tensor (order fixed!)
        digit_labels = torch.tensor(
            [
                sample["labels"]["d4"],
                sample["labels"]["d3"],
                sample["labels"]["d2"],
                sample["labels"]["d1"],
                sample["labels"]["d0"],
            ],
            dtype=torch.long,
        )

        encoded["digit_labels"] = digit_labels
        encoded["answer"] = sample["answer"]
        # Add metadata for evaluation export
        encoded["question"] = sample["question"]
        encoded["prompt"] = sample["text"]
        encoded["generated_final_answer"] = sample["generated_final_answer"]

        return encoded
