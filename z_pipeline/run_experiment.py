# z_pipeline/run_experiment.py
#
# Orchestrates:
#   Phase-2 -> Phase-3 dataset generation -> Phase-3 training
#
# Contract:
# - phase2_ckpt is in-memory dict returned by run_phase2
# - generate_phase3_dataset returns a HF Dataset (per split)
# - we build DatasetDict({"train": ..., "eval": ...}), save_to_disk, then pass into run_phase3
# - optionally save a "start checkpoint" (step 0) before training begins
#
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import DatasetDict
from torch.optim import AdamW

from phase2.train import run_phase2
from phase3.generate_dataset import generate_phase3_dataset
from phase3.train import run_phase3
from phase3.model import Phase3ZModel


# -----------------------------
# Top-level config wrapper
# -----------------------------

@dataclass
class ExperimentConfig:
    phase2: object  # Phase2Config
    phase3: object  # Phase3Config

    # Dataset generation inputs (for phase3.generate_dataset)
    # We intentionally keep these here (pipeline-level), not inside phase3.data,
    # because phase3.data is about already-generated sequences.
    dataset_name: str
    train_split: str = "train"
    eval_split: str = "eval"

    # Generation mode for dataset creation (using Phase-2 model)
    gen_batch_size: int = 16
    gen_z_mode: str = "hard_argmax"   # "hard_argmax" | "hard_sample"
    gen_temperature: float = 1.0


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # no autocast on CPU
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def _save_start_ckpt_if_needed(
    *,
    cfg_phase3,
    phase2_ckpt: dict,
    dataset_dict: DatasetDict,
    device: torch.device,
) -> Optional[str]:
    """
    Saves ckpt_step_0.pt if cfg.phase3.ckpt.save_at_start == True.

    We create Phase3ZModel.from_phase2, initialize optimizer, and save:
      - model_state
      - optim_state
      - cfg snapshot (inside run_phase3 anyway; but we keep parity)
      - step=0
    """
    ckpt_cfg = getattr(cfg_phase3, "ckpt", None)
    if ckpt_cfg is None:
        return None

    save_at_start = bool(getattr(ckpt_cfg, "save_at_start", False))
    if not save_at_start:
        return None

    save_dir = str(getattr(ckpt_cfg, "save_dir", "./phase3_ckpts"))
    _ensure_dir(save_dir)
    ckpt_path = os.path.join(save_dir, "ckpt_step_0.pt")

    # Build phase3 model initialized from phase2
    phase3_model = Phase3ZModel.from_phase2(phase2_model=phase2_ckpt["model"])
    phase3_model.to(device)

    # Optimizer must match phase3/train.py optimizer settings
    optim_cfg = getattr(cfg_phase3, "optim", None)
    if optim_cfg is None:
        raise RuntimeError("cfg.phase3.optim missing")

    optimizer = AdamW(
        phase3_model.parameters(),
        lr=float(getattr(optim_cfg, "lr")),
        betas=tuple(getattr(optim_cfg, "betas")),
        eps=float(getattr(optim_cfg, "eps")),
        weight_decay=float(getattr(optim_cfg, "weight_decay")),
    )

    payload = {
        "step": 0,
        "model_state": phase3_model.state_dict(),
        "optim_state": optimizer.state_dict(),
        # cfg is optional; run_phase3 will also embed cfg in later ckpts
        "cfg": getattr(cfg_phase3, "__dict__", None),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


# -----------------------------
# Main entry point
# -----------------------------

def run_experiment(cfg: ExperimentConfig) -> None:
    """
    Runs Phase-2 then Phase-3 end-to-end.

    Side effects:
      - Saves generated DatasetDict to: <cfg.phase3.ckpt.save_dir>/dataset
      - Saves optional start checkpoint: <cfg.phase3.ckpt.save_dir>/ckpt_step_0.pt
      - Saves training checkpoints per cfg.phase3.ckpt.save_every_steps (inside run_phase3)

    Returns:
      None (per your requirement).
    """
    device = _default_device()
    print(f"[run_experiment] device={device}")

    # -------------------------
    # Phase 2
    # -------------------------
    print("[run_experiment] Running Phase-2...")
    phase2_ckpt = run_phase2(cfg.phase2)  # in-memory handoff dict

    # -------------------------
    # Phase 3 dataset generation
    # -------------------------
    print("[run_experiment] Generating Phase-3 dataset (train split)...")
    train_ds = generate_phase3_dataset(
        phase2_ckpt=phase2_ckpt,
        dataset_name=cfg.dataset_name,
        split=cfg.train_split,
        batch_size=int(cfg.gen_batch_size),
        z_mode=str(cfg.gen_z_mode),
        temperature=float(cfg.gen_temperature),
        device=device,
    )

    print("[run_experiment] Generating Phase-3 dataset (eval split)...")
    eval_ds = generate_phase3_dataset(
        phase2_ckpt=phase2_ckpt,
        dataset_name=cfg.dataset_name,
        split=cfg.eval_split,
        batch_size=int(cfg.gen_batch_size),
        z_mode=str(cfg.gen_z_mode),
        temperature=float(cfg.gen_temperature),
        device=device,
    )

    ds_dict = DatasetDict({"train": train_ds, "eval": eval_ds})

    # Persist dataset to disk (always)
    save_dir = str(getattr(cfg.phase3.ckpt, "save_dir", "./phase3_ckpts"))
    ds_path = os.path.join(save_dir, "dataset")
    _ensure_dir(save_dir)
    print(f"[run_experiment] Saving Phase-3 DatasetDict to disk: {ds_path}")
    ds_dict.save_to_disk(ds_path)

    # Optionally save start checkpoint (step 0) before training begins
    ckpt_path = _save_start_ckpt_if_needed(
        cfg_phase3=cfg.phase3,
        phase2_ckpt=phase2_ckpt,
        dataset_dict=ds_dict,
        device=device,
    )
    if ckpt_path is not None:
        print(f"[run_experiment] Saved Phase-3 start checkpoint: {ckpt_path}")

    # -------------------------
    # Phase 3 training
    # -------------------------
    print("[run_experiment] Running Phase-3 training...")
    with _autocast_ctx(device):
        # If start ckpt exists, run_phase3 will resume from it.
        run_phase3(
            cfg.phase3,
            phase2_ckpt=phase2_ckpt,
            ds_dict=ds_dict,
            ckpt_path=ckpt_path,
            # Also pass path for redundancy (run_phase3 can persist again if desired)
            save_dataset_to_disk=None,
        )

    print("[run_experiment] Done.")


__all__ = ["ExperimentConfig", "run_experiment"]
