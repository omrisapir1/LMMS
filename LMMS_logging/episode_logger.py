import json
import os
from typing import List


def log_episode(
    *,
    run_id: str,
    global_step: int,
    batch_idx: int,
    episode_idx: int,
    question: str,
    K: int,
    full_sequence_tokens: List[str],
    pred_int: int,
    label_int: int,
    reward: float,
) -> None:
    """
    Append one episode log record as a JSON line to logs/{run_id}/episodes.jsonl.

    The record is pandas-friendly and contains only strings, ints, and floats.
    """
    log_dir = os.path.join("logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "episodes.jsonl")

    record = {
        "run_id": run_id,
        "global_step": int(global_step),
        "batch_idx": int(batch_idx),
        "episode_idx": int(episode_idx),
        "question": str(question),
        "K": int(K),
        "full_sequence_tokens": list(map(str, full_sequence_tokens)),
        "pred_int": int(pred_int),
        "label_int": int(label_int),
        "reward": float(reward),
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")

