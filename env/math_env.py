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
    digit_step_idx: int
    z_actions: List[int]
    digit_actions: List[int]
    padding_zeros: List[int]


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
        self.total_width = int(cfg["environment"]["answer"]["total_width"])

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

        # Digits 0..9
        self.digit_tokens = [str(i) for i in range(10)]
        self.digit_token_ids = self._tokens_to_ids(self.digit_tokens)

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
        # SAFE CHECKS
        assert length_ans <= self.total_width, "length_ans must be <= total_width"
        if length_ans < 1 or length_ans > self.total_width:
            raise ValueError(f"length_ans out of bounds: {length_ans}")

        # Sample K uniformly from [min, max]
        K = random.randint(self.K_min, self.K_max)

        # Initialize episode state
        padding_len = self.total_width - length_ans
        self.state = EpisodeState(
            question=str(question),
            label_int=label_int,
            length_ans=length_ans,
            K=K,
            phase="latent",
            latent_step_idx=0,
            digit_step_idx=0,
            z_actions=[],
            digit_actions=[],
            padding_zeros=[self.digit_token_ids[0]] * padding_len,  # IDs of '0'
        )
        # Clear any pending inserts
        self._pending_inserted_ids = []

        if self.debug:
            print(f"[Env] Reset: K={K} length_ans={length_ans} padding_zeros={padding_len}")

    def get_action_space(self) -> ActionSpace:
        st = self._require_state()
        if st.phase == "latent":
            return ActionSpace(allowed_token_ids=self.z_token_ids)
        elif st.phase == "answer":
            if st.digit_step_idx == 0:
                return ActionSpace(allowed_token_ids=self.z_token_ids[1:])
            return ActionSpace(allowed_token_ids=self.digit_token_ids)
        elif st.phase == "done":
            return ActionSpace(allowed_token_ids=[])  # no actions allowed
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
                self._pending_inserted_ids = [self.answer_token_id] + list(st.padding_zeros)
                if self.debug:
                    print(f"[Env] Transition to answer phase after K={st.K} latent steps.")
            elif st.latent_step_idx > st.K:
                # No extra latent steps allowed
                raise RuntimeError("Exceeded K latent steps; protocol violation.")

        elif st.phase == "answer":
            # In answer phase, ensure latent steps completed
            assert st.latent_step_idx == st.K, "Illegal state: answer phase before completing K latent steps"
            # Record digit and increment
            st.digit_actions.append(action)
            st.digit_step_idx += 1
            # Terminate after exactly length_ans digit actions
            if st.digit_step_idx == st.length_ans:
                st.phase = "done"
                if self.debug:
                    pred_int = self._compute_pred_int(st)
                    print(f"[Env] Episode done. Predicted={pred_int} Label={st.label_int}")
            elif st.digit_step_idx > st.length_ans:
                raise RuntimeError("Exceeded allowed digit steps; protocol violation.")
        else:
            raise RuntimeError(f"Unknown phase: {st.phase}")

    def is_done(self) -> bool:
        st = self._require_state()
        return st.phase == "done"

    def get_reward(self) -> float:
        st = self._require_state()
        if st.phase != "done":
            raise RuntimeError("get_reward() called before episode termination.")
        pred_int = self._compute_pred_int(st)
        return 1.0 if pred_int == st.label_int else 0.0

    def get_inserted_token_ids(self) -> List[int]:
        """
        Returns environment-inserted tokens that must be appended
        to the input sequence before the next policy action.

        Phase-1 behavior:
        - returns [] during latent phase
        - returns [</answer>] + padding zeros when transitioning to answer phase
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
        return int(len(st.z_actions) + len(st.digit_actions))

    def get_z_actions(self) -> List[int]:
        st = self._require_state()
        return list(st.z_actions)

    def get_digit_actions(self) -> List[int]:
        st = self._require_state()
        return list(st.digit_actions)

    def get_label(self) -> int:
        st = self._require_state()
        return int(st.label_int)

    def _compute_pred_int(self, st: EpisodeState) -> int:
        # Construct predicted fixed-width string from padding zeros + digits
        # Convert token ids back to string digits to build the integer
        # We assume digit_token_ids map to '0'..'9' in order
        id_to_digit = {tid: str(i) for i, tid in enumerate(self.digit_token_ids)}
        padding_digits = [id_to_digit[tid] for tid in st.padding_zeros]
        action_digits = []
        for tid in st.digit_actions:
            if tid not in id_to_digit:
                raise RuntimeError(f"Digit action id {tid} not recognized as a digit token.")
            action_digits.append(id_to_digit[tid])
        pred_str = "".join(padding_digits + action_digits)
        assert len(pred_str) == self.total_width, "Predicted string must match total_width"
        if len(pred_str) != self.total_width:
            raise RuntimeError(f"Predicted string length {len(pred_str)} != total_width {self.total_width}")
        return int(pred_str)

    def _require_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state
