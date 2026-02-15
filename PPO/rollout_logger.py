from __future__ import annotations

import json
import os
from typing import Dict, Iterable


class RolloutLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_step(self, step: int, rows: Iterable[Dict[str, object]]) -> str:
        path = os.path.join(self.output_dir, f"rollouts_rank0_step{step:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return path
