from dataclasses import dataclass
from typing import List

import torch

@dataclass
class ActionSpace:
    # Allowed token IDs for the current step
    allowed_token_ids: List[int]

    def __post_init__(self):
        # Maintain a set for O(1) membership checks during rollout
        self._allowed_set = set(self.allowed_token_ids)

    def contains(self, token_id: int) -> bool:
        return token_id in self._allowed_set

    def logit_mask(self, vocab_size: int, device: torch.device, dtype: torch.dtype, allow_empty: bool = False) -> torch.Tensor:
        """
        Build an additive mask tensor for masking logits prior to sampling.

        Contract:
        - Returns a vector of shape [V] where V=vocab_size.
        - Values are 0.0 for allowed ids and a large negative value for disallowed ids.
        - Use like: masked_logits = logits + mask (e.g., logits[:, -1, :] + mask)

        Safety checks:
        - allowed_token_ids must be non-empty unless allow_empty=True (e.g., when env is done)
        - all ids must be within [0, V-1]
        - mask will not block all tokens unless allow_empty=True

        Notes:
        - This mask MUST be applied before sampling. Sampling MUST NOT be done on unmasked logits.
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not self._allowed_set and not allow_empty:
            raise ValueError("No allowed tokens for current action space (did you call during a done state?).")

        # Validate ids within range
        for tid in self._allowed_set:
            if not isinstance(tid, int):
                raise TypeError(f"Token id {tid} is not an int.")
            if tid < 0 or tid >= vocab_size:
                raise ValueError(f"Token id {tid} out of range [0,{vocab_size-1}].")

        # Build mask
        neg_large = -1e9
        mask = torch.full((vocab_size,), fill_value=neg_large, device=device, dtype=dtype)
        if self._allowed_set:
            idx = torch.tensor(sorted(self._allowed_set), device=device)
            mask.index_fill_(0, idx, 0.0)
        return mask


def _self_test() -> None:
    # Tiny self-check for mask semantics
    torch.manual_seed(0)
    space = ActionSpace([3, 7])
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
    assert torch.allclose(probs[0, 0, disallowed_idx], torch.zeros(len(disallowed_idx)), atol=1e-7)


if __name__ == "__main__":
    _self_test()
