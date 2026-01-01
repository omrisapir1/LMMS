import torch
import torch.nn as nn
from typing import Optional, Dict

from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoTokenizer,
)

from .model_config import Phase0Config


class Phase0Model(PreTrainedModel):
    config_class = Phase0Config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Ensure config carries correct vocab_size from tokenizer before model init
        try:
            config = Phase0Config.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            config = kwargs.get("config")

        try:
            tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            if config is None:
                config = Phase0Config()
            config.vocab_size = len(tok)
            if getattr(config, "answer_token", None) and getattr(config, "answer_token_id", None) is None:
                config.answer_token_id = tok.convert_tokens_to_ids(config.answer_token)
            kwargs["config"] = config
        except Exception:
            if config is not None:
                kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(self, config: Phase0Config):
        super().__init__(config)

        self.answer_token_id = config.answer_token_id

        # ---- Base LM ----
        self.model = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Resize base embeddings to tokenizer vocab size saved in config
        if getattr(config, "vocab_size", None) is not None:
            try:
                self.model.resize_token_embeddings(config.vocab_size)
            except Exception:
                pass

        hidden_size = self.model.config.hidden_size

        # ---- Digit heads ----
        self.digit_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, config.num_classes)
                for _ in range(config.num_digits)
            ]
        )

        # ---- Freeze policy ----
        self._freeze_all()
        self._unfreeze_answer_embedding()
        self._unfreeze_last_layers(config.unfrozen_layer_pct)

        # Required by HF (ties weights, etc.)
        self.post_init()

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        digit_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        hidden = outputs.hidden_states[-1]  # [B, T, H]

        # ---- Locate <ANSWER> token ----
        answer_mask = input_ids == self.answer_token_id  # [B, T]


        if not torch.all(answer_mask.sum(dim=1) == 1):
            raise RuntimeError(
                "Each sample must contain exactly one <ANSWER> token"
            )

        idx = answer_mask.float().argmax(dim=1)  # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        answer_hidden = hidden[batch_idx, idx]  # [B, H]

        # ---- Digit logits ----
        logits = torch.stack(
            [head(answer_hidden) for head in self.digit_heads],
            dim=1,
        )  # [B, 5, 10]

        loss = None
        if digit_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = sum(
                loss_fn(logits[:, i], digit_labels[:, i])
                for i in range(logits.size(1))
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    # ─────────────────────────────────────────────────────────
    # Freezing utilities
    # ─────────────────────────────────────────────────────────

    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_answer_embedding(self):
        emb = self.model.get_input_embeddings()
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

        # This assumes a decoder-only LM (LLaMA / Qwen style)
        layers = self.model.layers
        total_layers = len(layers)

        n_unfreeze = max(1, int(total_layers * pct))

        for layer in layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
