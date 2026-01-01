from transformers import PretrainedConfig


class Phase0Config(PretrainedConfig):
    model_type = "phase0"

    def __init__(
        self,
        base_model_name: str,
        answer_token: str,
        answer_token_id: int,
        unfrozen_layer_pct: float = 0.0,
        num_digits: int = 5,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.answer_token = answer_token
        self.answer_token_id = answer_token_id
        self.unfrozen_layer_pct = unfrozen_layer_pct
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.output_hidden_states = True

