"""
Phase 1 dataset.
Consumes Hugging Face records with fields:
- question
- answer
- generated_answer

Derives:
- thoughts via split_thoughts(generated_answer)
- K = number of thoughts
- digit labels from answer

Dataset is stage-agnostic; stage logic is injected via num_latent_fn.
"""
from typing import List, Optional, Callable
import re

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

LATENT_TOKEN = "<|latent|>"
# ANSWER_TOKEN is documented-only; actual token is provided via dataset constructor
ANSWER_TOKEN = "<ANSWER>"


def build_prompt(question: str, answer: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    with_chat = tokenizer.apply_chat_template(messages,
                                              tokenize=False,
                                              add_generation_prompt=False)
    return tokenizer(with_chat,  add_special_tokens=True, padding=False, return_attention_mask=True)


def format_answer(thoughts: List[str], K: int, num_latent: int, answer_token: str) -> str:
    """
    Build formatted text with:
    - first (K - num_latent) thoughts (each on its own line)
    - num_latent latent tokens appended on a single line
    - final <ANSWER> token on the last line

    Assumptions:
    - 0 <= num_latent <= K
    - No stage logic
    - No validation or exclusion

    Newline policy:
    - sections separated by newlines
    - thoughts are newline-separated
    - latent tokens are space-separated on their line
    - <ANSWER> is always last and on its own line
    """
    # Defensive clamp of num_latent to [0, K]
    num_latent = max(0, min(int(num_latent), int(K)))
    keep_n = max(0, K - num_latent)
    lines: List[str] = []
    # question first

    # kept thoughts
    for t in thoughts[:keep_n]:
        lines.append(t)
    # latent tokens line, if any

    if num_latent > 0:
        lines.append(" ".join([LATENT_TOKEN] * num_latent + [answer_token]))

    return "\n".join(lines)




# Single dataset class with no stage logic
import torch

class Phase1Dataset(torch.utils.data.Dataset):
    """
    Dataset that returns tokenized inputs and digit labels.
    No stage policy. Uses num_latent_fn(K) to determine formatting.
    - If num_latent_fn(K) returns None, dataset does not raise; it defaults to zero latent tokens.
    - Otherwise, formats with format_text(question, thoughts, K, num_latent, answer_token).
    Returns dict with input_ids, attention_mask, digit_labels (torch.LongTensor of shape [5]), and optional K, id.
    """
    def __init__(
        self,
        items: List[dict],
        tokenizer,
        max_length: int,
        num_latent_fn: Callable[[int], Optional[int]],
        max_thoughts: int,
        answer_token: str,
        debug: bool = True,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_latent_fn = num_latent_fn
        self.max_thoughts = int(max_thoughts)
        # Store answer token on the instance; do not mutate module globals
        self.answer_token = answer_token

        self.debug = debug
        self.already_been_called_to_print = False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        # Thoughts and K are provided upstream (preprocessed once)
        if "thoughts" not in rec:
            raise KeyError("thoughts missing from record; preprocessing is mandatory")
        if "K" not in rec:
            raise KeyError("K missing from record; preprocessing is mandatory")
        thoughts = rec["thoughts"]
        K = rec["K"]
        # K == 0 is allowed (no-thought examples); handled by stage logic upstream
        # Enforce global thought cap
        assert K <= self.max_thoughts, "K > max_thoughts must be filtered upstream"
        # Stage-agnostic latent count
        num_latent_opt = self.num_latent_fn(K)
        num_latent = 0 if num_latent_opt is None else int(num_latent_opt)
        # Validate non-negative and clamp to K before formatting and checks
        assert num_latent >= 0, "num_latent must be >= 0"
        num_latent = max(0, min(num_latent, K))
        # Include question in formatting
        question = rec.get("question")
        if question is None:
            raise KeyError("question missing from record")
        answer_text = format_answer(thoughts, K, num_latent, self.answer_token)
        enc = build_prompt(answer_text, question, self.tokenizer)
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        # Tokenization safety checks
        answer_id = self.tokenizer.convert_tokens_to_ids(self.answer_token)
        latent_id = self.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        # Debug-only: ensure special tokens are in vocab (not mapped to UNK)
        assert latent_id != self.tokenizer.unk_token_id
        assert answer_id != self.tokenizer.unk_token_id
        assert enc["input_ids"].count(answer_id) == 1, "<ANSWER> must appear exactly once"
        assert enc["input_ids"].count(latent_id) == num_latent, "Latent tokens count mismatch"
        # Derive digit labels from answer
        ans = rec.get("answer")
        print(f'This is ans: {ans}')
        if ans is None:
            raise KeyError("answer missing from record")
        answer_int = int(ans)
        if not (0 <= answer_int <= 99999):
            raise ValueError(f"answer out of range [0,99999]: {answer_int}")
        digits = [int(c) for c in f"{answer_int:05d}"]
        digit_labels = torch.tensor(digits, dtype=torch.long)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "digit_labels": digit_labels,
            "K": K,
        }
        rid = rec.get("id")
        if rid is not None:
            out["id"] = rid

        if self.debug and not self.already_been_called_to_print:
            print(f"[Phase1Dataset] Sample {idx}:")
            print(f"  question: {question}")
            print(f"  answer: {ans}")
            print(f"  thoughts (K={K})")
            print(f"  num_latent: {num_latent}")
            print(f"  formatted answer text:\n{answer_text}")
            print(f"  input_ids tokenized: {self.tokenizer.decode(input_ids.tolist())}")
            print(f"  digit_labels: {digit_labels.tolist()}")
            self.already_been_called_to_print = True

        return out


# New: collate function for DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: List[dict], pad_token_id: int):
    """
    Pad input_ids and attention_mask; stack digit_labels to [B,5].
    - pad_token_id: use tokenizer.pad_token_id from caller.
    Returns dict with batched tensors and passes through optional K and id lists.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    digit_labels_list = [item["digit_labels"] for item in batch]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    digit_labels = torch.stack(digit_labels_list, dim=0)  # [B,5]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "digit_labels": digit_labels,
    }
    # Optional debug fields
    if "K" in batch[0]:
        out["K"] = torch.tensor([item["K"] for item in batch], dtype=torch.long)
    if "id" in batch[0]:
        out["id"] = [item.get("id", "") for item in batch]
    return out
