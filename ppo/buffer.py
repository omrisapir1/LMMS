from typing import Any, Dict, List

import torch


class TrajectoryBuffer:
    """
    Pure storage for PPO rollouts.

    Stores per-episode trajectories and exposes flattened, PPO-ready tensors.

    Per-step stored: actions, logprob_old, value_old, entropy, phases
    Per-episode stored: reward
    """

    def __init__(self):
        self._episodes: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return sum(len(ep.get("actions", [])) for ep in self._episodes)

    def clear(self) -> None:
        self._episodes.clear()

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """
        Add a single episode returned by collect_rollout.

        Expected keys:
        - actions: List[int]
        - logprob_old: List[float]
        - value_old: List[float]
        - entropy: List[float]
        - phases: List[str]
        - reward: float

        Shape checks:
        - All per-step lists must be the same non-zero length
        - Reward must be scalar
        """
        required_step_keys = [
            "actions", "logprob_old", "value_old", "entropy", "phases",
            "input_ids_steps", "attention_mask_steps", "inserted_token_ids_steps"
        ]
        for k in required_step_keys + ["reward"]:
            if k not in episode:
                raise KeyError(f"Episode missing key: {k}")

        actions = episode["actions"]
        logprob_old = episode["logprob_old"]
        value_old = episode["value_old"]
        entropy = episode["entropy"]
        phases = episode["phases"]
        reward = episode["reward"]
        input_ids_steps = episode["input_ids_steps"]
        attention_mask_steps = episode["attention_mask_steps"]
        inserted_token_ids_steps = episode["inserted_token_ids_steps"]

        # Optional metadata
        question = episode.get("question")
        label_int = episode.get("label_int")
        K = episode.get("K")
        length_ans = episode.get("length_ans")

        if not isinstance(reward, (float, int)):
            raise TypeError("reward must be a scalar float/int")

        lengths = list(map(len, [actions, logprob_old, value_old, entropy, phases, input_ids_steps, attention_mask_steps, inserted_token_ids_steps]))
        if any(L == 0 for L in lengths):
            raise ValueError("Episode length must be > 0 for all per-step lists")
        if len(set(lengths)) != 1:
            raise ValueError(f"Per-step lists have mismatched lengths: {lengths}")

        # Basic type sanity
        if not all(isinstance(a, int) for a in actions):
            raise TypeError("actions must be List[int]")
        if not all(isinstance(p, (float, int)) for p in logprob_old):
            raise TypeError("logprob_old must be List[float]")
        if not all(isinstance(v, (float, int)) for v in value_old):
            raise TypeError("value_old must be List[float]")
        if not all(isinstance(e, (float, int)) for e in entropy):
            raise TypeError("entropy must be List[float]")
        if not all(isinstance(s, str) for s in phases):
            raise TypeError("phases must be List[str]")
        if not all(isinstance(seq, list) for seq in input_ids_steps):
            raise TypeError("input_ids_steps must be List[List[int]]")
        if not all(isinstance(seq, list) for seq in attention_mask_steps):
            raise TypeError("attention_mask_steps must be List[List[int]]")
        if not all(isinstance(seq, list) for seq in inserted_token_ids_steps):
            raise TypeError("inserted_token_ids_steps must be List[List[int]]")
        if any(len(a) != len(m) for a, m in zip(input_ids_steps, attention_mask_steps)):
            raise ValueError("input_ids_steps and attention_mask_steps length mismatch at a step")

        # New: ensure inner contents are integers
        for step_idx, seq in enumerate(input_ids_steps):
            if not all(isinstance(x, int) for x in seq):
                raise TypeError(f"input_ids_steps[{step_idx}] must contain int elements")
        for step_idx, seq in enumerate(attention_mask_steps):
            if not all(isinstance(x, int) for x in seq):
                raise TypeError(f"attention_mask_steps[{step_idx}] must contain int elements")
        for step_idx, ids in enumerate(inserted_token_ids_steps):
            if not isinstance(ids, list):
                raise TypeError(f"inserted_token_ids_steps[{step_idx}] must be a list")
            if not all(isinstance(x, int) for x in ids):
                raise TypeError(f"inserted_token_ids_steps[{step_idx}] must contain int elements")

        # Store episode (keep original structure)
        stored = {
            "actions": list(actions),
            "logprob_old": list(float(x) for x in logprob_old),
            "value_old": list(float(x) for x in value_old),
            "entropy": list(float(x) for x in entropy),
            "phases": list(phases),
            "reward": float(reward),
            "input_ids_steps": [list(ids) for ids in input_ids_steps],
            "attention_mask_steps": [list(msk) for msk in attention_mask_steps],
            "inserted_token_ids_steps": [list(ids) for ids in inserted_token_ids_steps],
        }
        if question is not None:
            stored["question"] = str(question)
        if label_int is not None:
            stored["label_int"] = int(label_int)
        if K is not None:
            stored["K"] = int(K)
        if length_ans is not None:
            stored["length_ans"] = int(length_ans)
        self._episodes.append(stored)

    def get_batch(self, device: torch.device, pad_token_id: int = 0) -> Dict[str, Any]:
        """
        Flatten all episodes into PPO-ready tensors.

        Returns dict with:
        - actions: LongTensor [N]
        - logprob_old: FloatTensor [N]
        - value_old: FloatTensor [N]
        - entropy: FloatTensor [N]
        - rewards: FloatTensor [N] (broadcast per step)
        - phases: List[str] length N
        - episode_index: LongTensor [N]
        - input_ids: LongTensor [N, T] (padded to max T)
        - attention_mask: LongTensor [N, T]
        """
        total_steps = len(self)
        if total_steps == 0:
            raise ValueError("Buffer is empty; add episodes before batching.")

        actions_all: List[int] = []
        logprob_all: List[float] = []
        value_all: List[float] = []
        entropy_all: List[float] = []
        rewards_all: List[float] = []
        phases_all: List[str] = []
        episode_index_all: List[int] = []
        seqs_all: List[List[int]] = []
        masks_all: List[List[int]] = []

        for epi, ep in enumerate(self._episodes):
            L = len(ep["actions"])
            actions_all.extend(ep["actions"])
            logprob_all.extend(ep["logprob_old"])
            value_all.extend(ep["value_old"])
            entropy_all.extend(ep["entropy"])
            phases_all.extend(ep["phases"])
            rewards_all.extend([ep["reward"]] * L)
            episode_index_all.extend([epi] * L)
            seqs_all.extend(ep["input_ids_steps"])
            masks_all.extend(ep["attention_mask_steps"])

        # NOTE:
        # inserted_token_ids_steps is intentionally NOT included in the PPO batch.
        # It is episode-structured metadata used only for logging / reconstruction,
        # not for policy or value computation.

        # Minor robustness: ensure we collected exactly one sequence per step
        if len(seqs_all) != total_steps:
            raise RuntimeError("Mismatch between total steps and collected sequences")

        # Pad sequences to max length (LEFT padding to align last index with last real token)
        max_T = max(len(s) for s in seqs_all)
        input_ids_padded = []
        attention_mask_padded = []
        for s, m in zip(seqs_all, masks_all):
            pad_len = max_T - len(s)
            # Left-pad input_ids with pad_token_id and attention_mask with zeros
            input_ids_padded.append(([pad_token_id] * pad_len) + s)
            attention_mask_padded.append(([0] * pad_len) + m)

        # Convert to tensors
        actions_t = torch.tensor(actions_all, dtype=torch.long, device=device)
        logprob_t = torch.tensor(logprob_all, dtype=torch.float32, device=device)
        value_t = torch.tensor(value_all, dtype=torch.float32, device=device)
        entropy_t = torch.tensor(entropy_all, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards_all, dtype=torch.float32, device=device)
        episode_index_t = torch.tensor(episode_index_all, dtype=torch.long, device=device)
        input_ids_t = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(attention_mask_padded, dtype=torch.long, device=device)

        # Final shape checks
        N = total_steps
        for name, t in {
            "actions": actions_t,
            "logprob_old": logprob_t,
            "value_old": value_t,
            "entropy": entropy_t,
            "rewards": rewards_t,
            "episode_index": episode_index_t,
        }.items():
            if t.numel() != N:
                raise RuntimeError(f"Batched tensor '{name}' has size {t.numel()} but expected {N}")
        if input_ids_t.shape[0] != N or attention_mask_t.shape[0] != N:
            raise RuntimeError("Sequence batch first dimension mismatch")

        return {
            "actions": actions_t,
            "logprob_old": logprob_t,
            "value_old": value_t,
            "entropy": entropy_t,
            "rewards": rewards_t,
            "phases": phases_all,  # keep as strings per spec
            "episode_index": episode_index_t,
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
        }

    def reconstruct_full_sequence(self, ep: Dict[str, Any]) -> List[int]:
        """
        Reconstruct the full generated token ID sequence for an episode,
        including policy actions and environment-inserted tokens.
        """
        seq: List[int] = []
        for prefix, action, inserted in zip(
            ep["input_ids_steps"],
            ep["actions"],
            ep["inserted_token_ids_steps"],
        ):
            seq = prefix[:]  # prefix already includes everything before this step
            seq.append(action)
            if inserted:
                seq.extend(inserted)
        return seq
