import sys
from pathlib import Path
# Ensure project root is on sys.path for imports like `phase_0.*`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer

from phase_0.model_config import Phase0Config
from phase_0.model import Phase0Model


def main():
    # Mirror train.py setup minimally
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # adjust if needed
    answer_token = "<ANSWER>"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({"additional_special_tokens": [answer_token]})
    answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
    assert answer_token_id != tokenizer.unk_token_id

    # Config + Model
    config = Phase0Config(
        base_model_name=base_model_name,
        answer_token=answer_token,
        answer_token_id=answer_token_id,
        unfrozen_layer_pct=0.25,
        num_digits=5,
        num_classes=10,
        vocab_size=len(tokenizer),
    )

    model = Phase0Model(config)
    model = model.to(dtype=torch.bfloat16)

    # Save
    out_dir = Path("/tmp/phase0_save_load_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Load
    loaded = Phase0Model.from_pretrained(out_dir, torch_dtype=torch.bfloat16)
    loaded.eval()

    # Reload tokenizer from the saved dir to ensure consistent length
    re_tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)

    # Quick shape checks: embedding and heads
    base_emb = loaded.model.get_input_embeddings().weight
    print("Loaded embedding shape:", tuple(base_emb.shape))
    print("Reloaded tokenizer length:", len(re_tok))
    assert base_emb.shape[0] == len(re_tok)

    print("OK: model saved and loaded successfully.")


if __name__ == "__main__":
    main()
