"""
ActionSpace module

Token masks apply only to LM (vocab) logits.
Class masks apply only to answer-head (classification) logits.
Rollout code must route logits explicitly by ActionSpace.kind.

Routing contract:
- kind="token" → LM vocab logits → use logit_mask()
- kind="answer" → answer head logits → use class_mask()

This file must fail loudly if the wrong mask is used for the given kind.
Do not auto-route inside ActionSpace.
"""
from dataclasses import dataclass
from typing import List, Literal, Optional, Set

import torch

@dataclass(init=False)
class ActionSpace:
    """
    ActionSpace invariants:

    - kind="token":
        * allowed_ids are vocab token ids
        * use logit_mask()
        * logits shape [..., vocab_size]

    - kind="answer":
        * allowed_ids are class indices
        * use class_mask()
        * logits shape [..., num_classes]

    Using the wrong mask for a given kind is a fatal error.
    """
    allowed_ids: List[int]
    kind: Literal["token", "answer"] = "token"  # explicit kinds

    def __init__(self, allowed_ids: Optional[List[int]] = None, kind: Literal["token", "answer"] = "token", allowed_token_ids: Optional[List[int]] = None):
        # Support both allowed_ids and allowed_token_ids for compatibility, but not both.
        if allowed_ids is not None and allowed_token_ids is not None:
            raise ValueError("Provide either allowed_ids or allowed_token_ids, not both.")
        if allowed_ids is None and allowed_token_ids is not None:
            allowed_ids = allowed_token_ids
        # Normalize to a concrete list internally
        self.allowed_ids = list(allowed_ids) if allowed_ids is not None else []
        self.kind = kind
        self.__post_init__()

    def __post_init__(self):
        # Validate allowed_ids basic invariants (allow empty at construction for terminal states)
        if self.allowed_ids is None:
            self.allowed_ids = []
        # Ensure all ids are ints
        for tid in self.allowed_ids:
            if not isinstance(tid, int):
                raise TypeError(f"Token/class id {tid} is not an int.")
        # Additional validation based on kind
        if self.kind == "answer":
            for cid in self.allowed_ids:
                if cid < 0:
                    raise ValueError("Answer action ids must be non-negative class indices")
        elif self.kind == "token":
            # For tokens, we cannot know vocab_size here; range checks are performed in logit_mask
            pass
        else:
            raise ValueError(f"Invalid ActionSpace kind: {self.kind}")

        # Maintain a set for O(1) membership checks during rollout
        self._allowed_set: Set[int] = set(self.allowed_ids)
        # Do NOT fail on empty here; environments may construct empty spaces for done states.

    def contains(self, action: int) -> bool:
        """
        Check whether an action is allowed in this action space.

        Semantics:
        - kind="token": action is a vocab token id
        - kind="answer": action is a class index
        """
        if not isinstance(action, int):
            raise TypeError("Action must be an int")
        return action in self._allowed_set

    def _build_mask(self, size: int, allowed: Set[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Dtype-aware large negative to suppress logits robustly across dtypes
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            raise TypeError(f"Mask dtype must be floating point, got {dtype}")
        neg_large = torch.finfo(dtype).min
        mask = torch.full((size,), fill_value=neg_large, device=device, dtype=dtype)
        if allowed:
            idx = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
            mask.index_fill_(0, idx, 0.0)
        return mask

    def logit_mask(self, vocab_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build an additive mask tensor for masking LM token logits prior to sampling.

        Contract:
        - Valid ONLY for kind == "token".
        - Returns a vector of shape [V] where V=vocab_size.
        - Values are 0.0 for allowed ids and a large negative value for disallowed ids.
        - Use like: masked_logits = logits + mask (e.g., logits[:, -1, :] + mask)

        Safety checks:
        - allowed_ids must be non-empty (empty spaces are invalid for PPO masking; environment should not mask after done)
        - all ids must be within [0, V-1]
        """
        if self.kind != "token":
            raise RuntimeError(
                f"logit_mask() called for ActionSpace kind='{self.kind}'. Token masking is only valid for token action spaces."
            )
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not self._allowed_set:
            raise RuntimeError("Empty ActionSpace is invalid for PPO masking (token). Environment must prevent masking after done.")
        # Validate ids within range
        for tid in self._allowed_set:
            if tid < 0 or tid >= vocab_size:
                raise ValueError(f"Token id {tid} out of range [0,{vocab_size-1}].")
        return self._build_mask(vocab_size, self._allowed_set, device, dtype)

    def class_mask(self, num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build an additive mask tensor for masking classification logits (answer head).

        Contract:
        - Valid ONLY for kind == "answer".
        - Returns a vector of shape [C] where C=num_classes.
        - Values are 0.0 for allowed class indices and a large negative value for disallowed indices.

        Safety checks:
        - allowed_ids must be non-empty (empty spaces are invalid for PPO masking; environment should not mask after done)
        - all ids must be within [0, C-1]
        """
        if self.kind != "answer":
            raise RuntimeError(
                f"class_mask() called for ActionSpace kind='{self.kind}'. Classification masking is only valid for answer action spaces."
            )
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not self._allowed_set:
            raise RuntimeError("Empty ActionSpace is invalid for PPO masking (answer). Environment must prevent masking after done.")
        # Validate ids within range
        for cid in self._allowed_set:
            if cid < 0 or cid >= num_classes:
                raise ValueError(f"Class id {cid} out of range [0,{num_classes-1}].")
        return self._build_mask(num_classes, self._allowed_set, device, dtype)


def _self_test() -> None:
    # Tiny self-check for token mask semantics
    torch.manual_seed(0)
    space = ActionSpace([3, 7], kind="token")
    V = 20
    mask = space.logit_mask(vocab_size=V, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (V,)
    assert float(mask[3].item()) == 0.0 and float(mask[7].item()) == 0.0
    # Most others should be negative
    assert (mask.sum().item() < 0)

    # Apply to random logits and check probabilities ~0 for disallowed
    logits = torch.randn(1, 1, V)
    masked_logits = logits + mask.view(1, 1, V)
    probs = torch.softmax(masked_logits, dim=-1)
    disallowed_idx = [i for i in range(V) if i not in {3, 7}]
    # Numerically robust upper-bound check for disallowed probabilities (bf16-safe)
    assert probs[0, 0, disallowed_idx].max().item() < 1e-5

    # Classification action space test
    C = 10
    answer_space = ActionSpace([0, 5, 9], kind="answer")
    class_mask = answer_space.class_mask(num_classes=C, device=torch.device("cpu"), dtype=torch.float32)
    assert class_mask.shape == (C,)
    assert float(class_mask[0].item()) == 0.0 and float(class_mask[5].item()) == 0.0 and float(class_mask[9].item()) == 0.0
    assert (class_mask.sum().item() < 0)

    # Apply to random class logits and check probabilities ~0 for disallowed classes
    class_logits = torch.randn(1, C)
    masked_class_logits = class_logits + class_mask.view(1, C)
    class_probs = torch.softmax(masked_class_logits, dim=-1)
    disallowed_classes = [i for i in range(C) if i not in {0, 5, 9}]
    # Numerically robust upper-bound check (bf16-safe)
    assert class_probs[0, disallowed_classes].max().item() < 1e-5

    # Invalid usage checks (head-specific failures)
    try:
        _ = answer_space.logit_mask(vocab_size=V, device=torch.device("cpu"), dtype=torch.float32)
        raise AssertionError("Expected RuntimeError for logit_mask on answer kind")
    except RuntimeError:
        pass
    try:
        token_space = ActionSpace([1, 2], kind="token")
        _ = token_space.class_mask(num_classes=C, device=torch.device("cpu"), dtype=torch.float32)
        raise AssertionError("Expected RuntimeError for class_mask on token kind")
    except RuntimeError:
        pass

    # Empty ActionSpace construction and mask errors
    empty_token = ActionSpace([], kind="token")
    try:
        _ = empty_token.logit_mask(vocab_size=V, device=torch.device("cpu"), dtype=torch.float32)
        raise AssertionError("Expected RuntimeError for token mask on empty ActionSpace")
    except RuntimeError:
        pass
    empty_answer = ActionSpace([], kind="answer")
    try:
        _ = empty_answer.class_mask(num_classes=C, device=torch.device("cpu"), dtype=torch.float32)
        raise AssertionError("Expected RuntimeError for answer mask on empty ActionSpace")
    except RuntimeError:
        pass


if __name__ == "__main__":
    _self_test()
