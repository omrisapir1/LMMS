from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .conf import Config
from .eval import evaluate_generate_table
from .model import UnifiedZSoftModel


def _extract_five_digits(final_answer: Any) -> List[int]:
    if isinstance(final_answer, (list, tuple)):
        vals = [int(x) for x in final_answer]
        if len(vals) != 5 or any(v < 0 or v > 9 for v in vals):
            raise ValueError("final_answer list/tuple must contain exactly 5 digits in [0,9].")
        return vals

    s = str(final_answer)
    digits = re.findall(r"\d", s)
    if len(digits) != 5:
        raise ValueError(f"Could not parse exactly 5 digits from final_answer: {final_answer!r}")
    return [int(x) for x in digits]


class HFEvalGenerationDataset(Dataset):
    def __init__(self, hf_ds) -> None:
        self.ds = hf_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        if "problem" not in ex:
            raise KeyError("Dataset samples must contain 'problem' (str).")
        if "final_answer" in ex:
            ans_raw = ex["final_answer"]
        elif "final_asnwer" in ex:
            ans_raw = ex["final_asnwer"]
        else:
            raise KeyError("Dataset samples must contain 'final_answer' (or legacy 'final_asnwer').")

        digits = _extract_five_digits(ans_raw)
        return {
            "question": str(ex["problem"]),
            "digit_labels": torch.tensor(digits, dtype=torch.long),
        }


def _collate_hf_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "question": [str(x["question"]) for x in batch],
        "digit_labels": torch.stack([x["digit_labels"] for x in batch], dim=0),
    }


def _load_cfg_from_checkpoint(ckpt_dir: str) -> Config:
    cfg = Config()
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        return cfg
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for section in ("model", "data", "loss", "train"):
        src = raw.get(section, {})
        dst = getattr(cfg, section)
        for k, v in src.items():
            if hasattr(dst, k):
                setattr(dst, k, v)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Phase23 evaluate_generate_table on an HF dataset with columns problem/final_answer."
    )
    p.add_argument("--checkpoint_dir", required=True, help="Phase23 checkpoint dir (contains phase23_state.pt).")
    p.add_argument("--dataset_name", required=True, help="HF dataset name or repo id.")
    p.add_argument("--split", default="train", help="HF split name (default: train).")
    p.add_argument("--batch_size", type=int, default=16, help="Eval batch size.")
    p.add_argument("--step", type=int, default=0, help="Step label used in output filename.")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output dir for eval_generations_step_*.jsonl (default: checkpoint_dir).",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = _load_cfg_from_checkpoint(args.checkpoint_dir)
    cfg.train.output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    model = UnifiedZSoftModel.from_pretrained(
        args.checkpoint_dir,
        device=device,
        v_z=cfg.model.v_z,
        torch_dtype=torch.bfloat16,
        z_prefix=cfg.model.z_prefix,
        latent_token=cfg.model.latent_token,
        answer_token=cfg.model.answer_token,
    )
    model.eval()

    hf_ds = load_dataset(args.dataset_name, split=args.split)
    ds = HFEvalGenerationDataset(hf_ds)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_hf_eval,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("Failed to resolve pad_token_id from tokenizer.")

    out_path = evaluate_generate_table(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        step=args.step,
        pad_token_id=pad_token_id,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
