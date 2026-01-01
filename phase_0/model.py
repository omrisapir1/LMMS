import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

class Phase0Config(PretrainedConfig):
    model_type = "phase0-qwen"

    def __init__(
        self,
        base_model_name: str,
        answer_token: str = "<ANSWER>",
        answer_token_id: int | None = None,
        unfrozen_layer_pct: float = 0.0,
        hidden_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.answer_token = answer_token
        self.answer_token_id = answer_token_id
        self.unfrozen_layer_pct = unfrozen_layer_pct
        self.hidden_size = hidden_size


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class Phase0Model(PreTrainedModel):
    config_class = Phase0Config

    def __init__(self, config: Phase0Config):
        super().__init__(config)

        if config.answer_token_id is None:
            raise ValueError("answer_token_id must be set in Phase0Config")

        self.answer_token = config.answer_token
        self.answer_token_id = config.answer_token_id

        # ---- Base model ----
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Ensure hidden_size is available both locally and on self.config
        emb = self.base_model.get_input_embeddings()
        cfg_hidden = getattr(self.base_model.config, "hidden_size", None)
        self.hidden_size = (
            cfg_hidden
            if cfg_hidden is not None
            else getattr(emb, "embedding_dim", None)
            or (emb.weight.shape[1] if hasattr(emb, "weight") else None)
        )
        if self.hidden_size is None:
            raise RuntimeError("Could not determine hidden_size from base model")
        if getattr(self.config, "hidden_size", None) is None:
            self.config.hidden_size = self.hidden_size

        # ---- Digit heads ----
        self.digit_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 10) for _ in range(5)]
        )

        # ---- HF initialization hook ----
        self.post_init()

        # ---- Freezing strategy ----
        self._freeze_all()
        self._unfreeze_answer_embedding()
        self._unfreeze_last_layers(config.unfrozen_layer_pct)

    # ─────────────────────────────────────────────────────────
    # Freezing utilities
    # ─────────────────────────────────────────────────────────

    def _freeze_all(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

    def _unfreeze_answer_embedding(self):
        emb = self.base_model.get_input_embeddings()
        emb.weight.requires_grad = False
        emb.weight[self.answer_token_id].requires_grad = True

    def _unfreeze_last_layers(self, pct: float):
        """
        Unfreeze the top pct of transformer layers.
        pct ∈ [0.0, 1.0]
        """
        if pct <= 0.0:
            return

        if not (0.0 <= pct <= 1.0):
            raise ValueError("unfrozen_layer_pct must be in [0, 1]")

        layers = self.base_model.model.layers
        total_layers = len(layers)

        n_unfreeze = int(total_layers * pct)
        if n_unfreeze == 0:
            n_unfreeze = 1

        for layer in layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        hidden = outputs.hidden_states[-1]  # [B, T, H]

        # ---- Locate <ANSWER> token ----
        answer_mask = input_ids == self.answer_token_id
        if not torch.all(answer_mask.sum(dim=1) == 1):
            raise RuntimeError(
                "Each sample must contain exactly one <ANSWER> token"
            )

        idx = answer_mask.float().argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        answer_hidden = hidden[batch_idx, idx]  # [B, H]

        logits = torch.stack(
            [head(answer_hidden) for head in self.digit_heads],
            dim=1,
        )  # [B, 5, 10]

        loss = None
        if digit_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = sum(
                loss_fn(logits[:, i], digit_labels[:, i])
                for i in range(5)
            )

        return {
            "logits": logits,
            "loss": loss,
        }
