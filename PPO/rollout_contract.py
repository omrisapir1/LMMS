from __future__ import annotations


def validate_action_scope(action_scope: str) -> str:
    if action_scope not in ("ppo_only_z_tokens", "ppo_full"):
        raise ValueError(
            f"Unsupported rollout.action_scope={action_scope!r}; expected 'ppo_only_z_tokens' or 'ppo_full'"
        )
    return action_scope


def is_ppo_action(action_scope: str, phase: str) -> bool:
    if phase == "z":
        return True
    if phase == "digits":
        return action_scope == "ppo_full"
    raise ValueError(f"Unsupported phase {phase!r}")
