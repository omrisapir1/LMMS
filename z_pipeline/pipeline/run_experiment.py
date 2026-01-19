# z_pipeline/pipeline/run_experiment.py

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch


def run_phase2(cfg: Dict[str, Any]):
    """
    Runs Phase-2 training in-memory.
    Returns the trained model object (to be reused by Phase-3 later).
    """

    # Import locally to avoid circular deps
    from phase2.train import train_phase2

    print("========== Phase 2: START ==========")

    model = train_phase2(cfg)

    print("========== Phase 2: END ==========")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment JSON config",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Global reproducibility (important for Phase-3 later)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -----------------------
    # Phase 2
    # -----------------------
    model = run_phase2(cfg["phase2"])

    # -----------------------
    # Phase 3 (future)
    # -----------------------
    # Placeholder – intentionally explicit
    #
    # model = run_phase3(
    #     model=model,
    #     cfg=cfg["phase3"],
    # )

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
