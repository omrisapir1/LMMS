# phase_0/model_config.py

from typing import Optional
from transformers import PretrainedConfig


class Phase0Config(PretrainedConfig):
    model_type = "phase0"

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        answer_token: Optional[str] = None,
        answer_token_id: Optional[int] = None,
        unfrozen_layer_pct: float = 0.0,
        num_digits: int = 5,
        num_classes: int = 10,
        vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Base LM ---
        self.base_model_name = base_model_name

        # --- Answer token ---
        self.answer_token = answer_token
        self.answer_token_id = answer_token_id

        # --- Training knobs ---
        self.unfrozen_layer_pct = unfrozen_layer_pct

        # --- Classification head ---
        self.num_digits = num_digits
        self.num_classes = num_classes

        # --- Tokenizer / embeddings ---
        self.vocab_size = vocab_size

        # Always needed for Phase-0
        self.output_hidden_states = True
