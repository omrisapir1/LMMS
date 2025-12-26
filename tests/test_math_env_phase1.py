import random
import math
import os
from typing import Dict, Any, List

import pytest

from env.math_env import MathEnv

# Minimal fake tokenizer that mimics the needed methods and IDs
class FakeTokenizer:
    def __init__(self, z_vocab_size: int = 64, answer_token: str = "</answer>"):
        # Build token->id map
        self.token_to_id = {}
        # Digits '0'..'9'
        for i in range(10):
            self.token_to_id[str(i)] = i
        # z tokens
        base = len(self.token_to_id)
        for i in range(z_vocab_size):
            self.token_to_id[f"<z{i}>"] = base + i
        # answer token
        self.token_to_id[answer_token] = base + z_vocab_size

    def convert_tokens_to_ids(self, tokens):
        # Accept either single token string or a list
        if isinstance(tokens, list):
            out = []
            for t in tokens:
                out.append(self.token_to_id.get(t))
            return out
        else:
            return self.token_to_id.get(tokens)


def load_cfg() -> Dict[str, Any]:
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "phase1.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


@pytest.fixture
def env_fixture():
    # Deterministic randomness for tests
    random.seed(1234)
    cfg = load_cfg()
    tok = FakeTokenizer(z_vocab_size=int(cfg["model"]["z_tokens"]["vocab_size"]), answer_token=str(cfg["model"]["special_tokens"]["answer_token"]))
    env = MathEnv(cfg, tok, debug=False)
    return env, tok, cfg


def test_reset_initializes_state(env_fixture):
    env, tok, cfg = env_fixture
    question = "2 + 3"
    label = {"label_int": 5, "length_ans": 1}
    env.reset(question, label)

    st = env._require_state()
    assert st.phase == "latent"
    assert cfg["environment"]["latent_steps"]["min"] <= st.K <= cfg["environment"]["latent_steps"]["max"]
    assert st.latent_step_idx == 0



def test_latent_phase_allows_only_z_tokens(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})

    space = env.get_action_space()
    # Collect z ids and digit ids
    z_ids = [tok.convert_tokens_to_ids(f"<z{i}>") for i in range(int(cfg["model"]["z_tokens"]["vocab_size"]))]
    digit_ids = [tok.convert_tokens_to_ids(str(i)) for i in range(10)]

    # All z ids should be allowed
    for zid in z_ids:
        assert space.contains(zid)
    # Any digit should be rejected
    for did in digit_ids:
        assert not space.contains(did)
        with pytest.raises(ValueError):
            env.step(did)


def test_exactly_K_latent_steps(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})
    st = env._require_state()
    K = st.K

    z0 = tok.convert_tokens_to_ids("<z0>")
    # Perform exactly K latent steps
    for _ in range(K):
        env.step(z0)
    # Phase should be answer
    assert env._require_state().phase == "answer"
    # Next latent action should error
    with pytest.raises(ValueError):
        env.step(z0)


def test_scaffold_insertion_once(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})
    st = env._require_state()
    K = st.K
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(K):
        env.step(z0)
    # First call should return answer + padding zeros
    inserted1 = env.get_inserted_token_ids()
    ans_id = tok.convert_tokens_to_ids(str(cfg["model"]["special_tokens"]["answer_token"]))
    total_width = int(cfg["environment"]["answer"]["total_width"])
    zeros_needed = total_width - st.length_ans
    zero_id = tok.convert_tokens_to_ids("0")
    assert inserted1 == [ans_id] + [zero_id] * zeros_needed

    # Second call should be empty
    inserted2 = env.get_inserted_token_ids()
    assert inserted2 == []


def test_answer_phase_allows_only_digits(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(st.K):
        env.step(z0)
    space = env.get_action_space()
    # Digits allowed
    for i in range(10):
        did = tok.convert_tokens_to_ids(str(i))
        assert space.contains(did)
    # z-token rejected
    assert not space.contains(z0)
    with pytest.raises(ValueError):
        env.step(z0)


def test_episode_terminates_after_length_ans_digits(env_fixture):
    env, tok, cfg = env_fixture
    label = {"label_int": 5, "length_ans": 1}
    env.reset("2 + 3", label)
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(st.K):
        env.step(z0)
    # Consume exactly length_ans digits
    did5 = tok.convert_tokens_to_ids("5")
    env.step(did5)
    assert env.is_done() is True
    # Further step should error
    with pytest.raises(RuntimeError):
        env.step(did5)


def test_reward_correct_positive(env_fixture):
    env, tok, cfg = env_fixture
    label = {"label_int": 5, "length_ans": 1}
    env.reset("2 + 3", label)
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(st.K):
        env.step(z0)
    # Emit correct digit
    env.step(tok.convert_tokens_to_ids("5"))
    assert env.is_done() is True
    assert env.get_reward() == 1.0


def test_reward_correct_negative(env_fixture):
    env, tok, cfg = env_fixture
    label = {"label_int": 5, "length_ans": 1}
    env.reset("2 + 3", label)
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(st.K):
        env.step(z0)
    # Emit wrong digit
    env.step(tok.convert_tokens_to_ids("4"))
    assert env.is_done() is True
    assert env.get_reward() == 0.0


def test_no_scaffold_during_latent_phase(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})

    # Before any latent steps
    assert env.get_inserted_token_ids() == []

    # After some but not all latent steps
    z0 = tok.convert_tokens_to_ids("<z0>")
    env.step(z0)
    assert env.get_inserted_token_ids() == []

def test_get_reward_before_done_raises(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})

    with pytest.raises(RuntimeError):
        env.get_reward()

def test_action_space_empty_when_done(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")
    for _ in range(st.K):
        env.step(z0)
    env.step(tok.convert_tokens_to_ids("5"))

    assert env.is_done()
    space = env.get_action_space()
    assert space.allowed_token_ids == []

def test_num_actions_taken_counts_correctly(env_fixture):
    env, tok, cfg = env_fixture
    env.reset("2 + 3", {"label_int": 5, "length_ans": 1})
    st = env._require_state()
    z0 = tok.convert_tokens_to_ids("<z0>")

    assert env.num_actions_taken() == 0
    env.step(z0)
    assert env.num_actions_taken() == 1

    for _ in range(st.K - 1):
        env.step(z0)
    assert env.num_actions_taken() == st.K

    env.step(tok.convert_tokens_to_ids("5"))
    assert env.num_actions_taken() == st.K + 1
