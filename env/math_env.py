from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

from .action_space import ActionSpace


@dataclass
class EpisodeState:
    question: str
    label_int: int
    length_ans: int
    K: int
    phase: str  # "latent" or "answer" or "done"
    latent_step_idx: int
    z_actions: List[int]
    answer_action: Optional[int]



class MathEnv:
    """Phase-1 math environment enforcing protocol and scaffolding.

    This environment owns the interaction structure. It does not sample actions or run PPO.
    """

    def __init__(self, cfg: Dict[str, Any], tokenizer, debug: bool = False):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.debug = debug

        # Config-driven protocol parameters
        self.K_min = int(cfg["environment"]["latent_steps"]["min"])
        self.K_max = int(cfg["environment"]["latent_steps"]["max"])
        # Derive token IDs from tokenizer and config
        z_vocab_size = int(cfg["model"]["z_tokens"]["vocab_size"])
        # CRITICAL: derive z-token ids directly from tokenizer; tokenizer is the source of truth
        self.z_token_ids = []
        for i in range(z_vocab_size):
            tok = f"<z{i}>"
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None:
                raise ValueError(f"Tokenizer missing id for z-token '{tok}'. Ensure tokenizer_ext added it.")
            self.z_token_ids.append(tid)

        answer_token = str(cfg["model"]["special_tokens"]["answer_token"])  # "</answer>"
        ans_id = self.tokenizer.convert_tokens_to_ids(answer_token)
        if ans_id is None:
            raise ValueError("Answer token id not found in tokenizer.")
        self.answer_token_id = ans_id

        # Episode state holder
        self.state: Optional[EpisodeState] = None

        # Pending environment-inserted tokens, to be returned once after transition
        self._pending_inserted_ids: List[int] = []

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if any(i is None for i in ids):
            missing = [t for t, i in zip(tokens, ids) if i is None]
            raise ValueError(f"Missing token ids for: {missing}")
        return ids

    def reset(self, question: str, label: Dict[str, Any]) -> None:
        # Read label fields
        label_int = int(label["label_int"])
        length_ans = int(label["length_ans"])


        # Sample K uniformly from [min, max]
        K = random.randint(self.K_min, self.K_max)

        # Initialize episode state

        self.state = EpisodeState(
            question=str(question),
            label_int=label_int,
            length_ans=length_ans,
            K=K,
            phase="latent",
            latent_step_idx=0,
            z_actions=[],
            answer_action=None,
        )
        # Clear any pending inserts
        self._pending_inserted_ids = []


    def get_action_space(self) -> ActionSpace:
        st = self._require_state()
        if st.phase == "latent":
            return ActionSpace(allowed_ids=self.z_token_ids, kind='token')
        elif st.phase == "answer":
            return ActionSpace(allowed_ids=list(range(10)), kind='answer')
        elif st.phase == "done":
            return ActionSpace(allowed_ids=[], kind="token")  # doesn't matter; never masked
        else:
            raise RuntimeError(f"Unknown phase: {st.phase}")

    def step(self, action: int) -> None:
        st = self._require_state()
        if st.phase == "done":
            raise RuntimeError("step() called after episode termination.")

        space = self.get_action_space()
        if not space.contains(action):
            raise ValueError(f"Action {action} not allowed in phase '{st.phase}'.")

        if st.phase == "latent":
            # Record z action and increment
            st.z_actions.append(action)
            st.latent_step_idx += 1
            # Transition to answer phase after exactly K latent steps
            if st.latent_step_idx == st.K:
                # NOTE (Phase-1):
                # </answer> is an environment-inserted scaffold token.
                # It is NOT a policy action and does not appear in trajectories.
                st.phase = "answer"
                # Prepare inserted scaffold tokens to be returned once on transition
                self._pending_inserted_ids = [self.answer_token_id]
                if self.debug:
                    print(f"[Env] Transition to answer phase after K={st.K} latent steps.")
            elif st.latent_step_idx > st.K:
                # No extra latent steps allowed
                raise RuntimeError("Exceeded K latent steps; protocol violation.")


        elif st.phase == "answer":
            assert st.latent_step_idx == st.K, "Illegal state: answer phase before completing K latent steps"
            if st.answer_action is not None:
                raise RuntimeError("Answer already provided; protocol violation.")

            st.answer_action = int(action)
            st.phase = "done"

    def is_done(self) -> bool:
        st = self._require_state()
        return st.phase == "done"

    def get_reward(self) -> float:
        st = self._require_state()
        if st.phase != "done":
            raise RuntimeError("get_reward() called before episode termination.")
        return 1.0 if st.answer_action == st.label_int else 0.0

    def get_inserted_token_ids(self) -> List[int]:
        """
        Returns environment-inserted tokens that must be appended
        to the input sequence before the next policy action.

        Phase-1 behavior:
        - returns [] during latent phase
        - returns [</answer>] when transitioning to answer phase
        - returns [] otherwise
        """
        st = self._require_state()
        if st.phase == "answer" and self._pending_inserted_ids:
            out = list(self._pending_inserted_ids)
            self._pending_inserted_ids = []
            return out
        return []

    def num_actions_taken(self) -> int:
        st = self._require_state()
        return int(len(st.z_actions) + (1 if st.answer_action is not None else 0))

    def get_z_actions(self) -> List[int]:
        st = self._require_state()
        return list(st.z_actions)

    def get_label(self) -> int:
        st = self._require_state()
        return int(st.label_int)

    def _require_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state
