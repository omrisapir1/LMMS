from __future__ import annotations

from typing import Dict, List

from phase1.dataset import ANSWER_TOKEN, LATENT_TOKEN, Phase1Dataset, format_answer
from phase1.eval import make_stage_k_filter, make_stage_num_latent_fn


class TinyTokenizer:
    def __init__(self) -> None:
        self._tok_to_id: Dict[str, int] = {
            LATENT_TOKEN: 1000,
            ANSWER_TOKEN: 1001,
        }
        for d in "0123456789":
            self._tok_to_id[d] = int(d)
        self._next_id = 2000
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize
        text = ""
        for msg in messages:
            text += f"{msg['role']}:{msg['content']}\n"
        if add_generation_prompt:
            text += "assistant:\n"
        return text

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._tok_to_id.get(token, -1))

    def _alloc(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            self._tok_to_id[tok] = self._next_id
            self._next_id += 1
        return int(self._tok_to_id[tok])

    def _tokenize(self, text: str) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(text):
            if text.startswith(LATENT_TOKEN, i):
                out.append(LATENT_TOKEN)
                i += len(LATENT_TOKEN)
                continue
            if text.startswith(ANSWER_TOKEN, i):
                out.append(ANSWER_TOKEN)
                i += len(ANSWER_TOKEN)
                continue
            ch = text[i]
            if ch.isdigit():
                out.append(ch)
                i += 1
                continue
            if ch.isspace():
                i += 1
                continue
            out.append(ch)
            i += 1
        return out

    def encode(self, text: str, add_special_tokens=False) -> List[int]:
        del add_special_tokens
        toks = self._tokenize(text)
        return [self._alloc(t) for t in toks]

    def __call__(
        self,
        text: str,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=True,
    ):
        del add_special_tokens, padding, return_attention_mask
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _records() -> List[Dict[str, str]]:
    return [
        {"question": "q1", "answer": 7, "generated_answer": "a1"},
        {"question": "q2", "answer": 42, "generated_answer": "a1\n\na2"},
        {"question": "q3", "answer": 314, "generated_answer": "a1\n\na2\n\na3"},
    ]


def test_stage_num_latent_mapping_transition() -> None:
    s1 = make_stage_num_latent_fn(1)
    s2 = make_stage_num_latent_fn(2)
    s3 = make_stage_num_latent_fn(3)
    s8 = make_stage_num_latent_fn(8)

    assert [s1(k) for k in (1, 2, 3)] == [0, 1, 1]
    assert [s2(k) for k in (1, 2, 3)] == [1, 2, 2]
    assert [s3(k) for k in (1, 2, 3)] == [1, 2, 3]
    assert [s8(k) for k in (1, 2, 3)] == [1, 2, 3]

    assert make_stage_k_filter(1)(1) is False
    assert make_stage_k_filter(2)(1) is False
    assert make_stage_k_filter(3)(1) is True
    assert make_stage_k_filter(8)(1) is True


def test_format_answer_partial_then_full_latentization() -> None:
    thoughts = ["t0", "t1", "t2"]
    k = len(thoughts)

    s1_out = format_answer(thoughts, k, make_stage_num_latent_fn(1)(k), ANSWER_TOKEN)
    s2_out = format_answer(thoughts, k, make_stage_num_latent_fn(2)(k), ANSWER_TOKEN)
    s3_out = format_answer(thoughts, k, make_stage_num_latent_fn(3)(k), ANSWER_TOKEN)

    assert "t0" in s1_out or "t1" in s1_out or "t2" in s1_out
    assert "t0" in s2_out or "t1" in s2_out or "t2" in s2_out
    assert s3_out == (LATENT_TOKEN * k) + ANSWER_TOKEN
    assert s3_out.count(LATENT_TOKEN) == k
    assert s3_out.count(ANSWER_TOKEN) == 1


def test_stage3_dataset_full_latent_and_digit_supervision_for_small_k() -> None:
    tok = TinyTokenizer()
    ds = Phase1Dataset(
        records=_records(),
        tokenizer=tok,
        num_latent_fn=make_stage_num_latent_fn(3),
        k_filter=make_stage_k_filter(3),
        max_thoughts=8,
        answer_token=ANSWER_TOKEN,
        min_chars=1,
        max_chars=10_000,
    )

    assert len(ds) == 3
    for i in range(len(ds)):
        sample = ds[i]
        ids = sample["input_ids"]
        answer_id = int(sample["answer_token_id"])

        assert ids.count(answer_id) == 1
        answer_pos = ids.index(answer_id)
        assert len(ids) >= (answer_pos + 6)

        digit_ids_after_answer = ids[answer_pos + 1: answer_pos + 6]
        assert len(digit_ids_after_answer) == 5
        assert all(int(x) in set(ds.digit_token_ids) for x in digit_ids_after_answer)

        assert int(sample["K"]) in (1, 2, 3)
        assert int(sample["latent_count"]) == int(sample["K"])
        assert sum(int(x) for x in sample["digit_mask"]) == 5
        assert int(sample["labels"][answer_pos]) == int(ids[answer_pos + 1])


def test_stage5_budget_behavior_examples() -> None:
    stage = 5
    fn = make_stage_num_latent_fn(stage)

    thoughts_k3 = [f"t{i}" for i in range(3)]
    thoughts_k6 = [f"t{i}" for i in range(6)]
    thoughts_k7 = [f"t{i}" for i in range(7)]

    out_k3 = format_answer(thoughts_k3, 3, fn(3), ANSWER_TOKEN)
    out_k6 = format_answer(thoughts_k6, 6, fn(6), ANSWER_TOKEN)
    out_k7 = format_answer(thoughts_k7, 7, fn(7), ANSWER_TOKEN)

    assert fn(3) == 3
    assert fn(6) == 5
    assert fn(7) == 5

    assert out_k3 == (LATENT_TOKEN * 3) + ANSWER_TOKEN
    assert out_k6.count(LATENT_TOKEN) == 5
    assert out_k7.count(LATENT_TOKEN) == 5
    assert "t4" in out_k6
    assert "t4" in out_k7
