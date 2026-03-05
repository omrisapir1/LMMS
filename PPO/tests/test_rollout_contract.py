from __future__ import annotations

import pytest

from PPO.conf import Config
from PPO.rollout_contract import is_ppo_action, validate_action_scope


def test_default_rollout_contract_values() -> None:
    cfg = Config()
    assert cfg.rollout.action_scope == "ppo_only_z_tokens"
    assert abs(cfg.rollout.top_p - 0.95) < 1e-12
    assert cfg.rollout.vllm_enabled is True
    assert cfg.rollout.vllm_sync_every == 4


def test_action_scope_validation() -> None:
    assert validate_action_scope("ppo_only_z_tokens") == "ppo_only_z_tokens"
    assert validate_action_scope("ppo_full") == "ppo_full"
    with pytest.raises(ValueError):
        validate_action_scope("bad_scope")


def test_scope_controls_digit_ppo_actions() -> None:
    assert is_ppo_action("ppo_only_z_tokens", "z") is True
    assert is_ppo_action("ppo_only_z_tokens", "digits") is False
    assert is_ppo_action("ppo_full", "z") is True
    assert is_ppo_action("ppo_full", "digits") is True
