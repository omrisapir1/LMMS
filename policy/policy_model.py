import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@dataclass
class PolicyForwardOutput:
    logits: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]
    values: torch.Tensor
    answer_logits: Optional[torch.Tensor] = None


class PolicyModel(nn.Module):
    """Policy model wrapper for LMMS Phase-1 and later phases.

    - Loads a pretrained causal LM from Hugging Face
    - Attaches a scalar value head operating on last hidden state per sequence
    - Applies freezing according to config (top fraction remains trainable)
    - Forward returns logits, hidden states, and values
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_name: str = cfg["model"]["base_model"]
        train_top_fraction: float = float(cfg["trainable"]["transformer"]["train_top_fraction"])  # e.g., 0.25

        # Load base model with required outputs. Some models (e.g., Qwen) may require trust_remote_code.
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Value head: projects hidden state to a scalar value.
        hidden_size = self._infer_hidden_size()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.answer_head = nn.Linear(hidden_size, 10)
        # Apply freezing rules according to Phase-1
        self._apply_freezing(train_top_fraction)

        # Log trainable parameter counts
        self._log_param_counts()



    def _infer_hidden_size(self) -> int:
        # Try common attributes
        if hasattr(self.lm, "config") and hasattr(self.lm.config, "hidden_size"):
            return int(self.lm.config.hidden_size)
        # Fallback: probe a small forward pass with dummy ids if tokenizer length is unknown is hard; use model internals
        # Many transformer models keep embed_dim on input embeddings
        emb = self.lm.get_input_embeddings()
        if emb is not None:
            return int(emb.embedding_dim)
        raise RuntimeError("Cannot infer hidden size from model config or embeddings.")

    def _get_transformer_blocks(self):
        """Return a list of transformer block modules and total count, handling different architectures.
        We avoid hardcoding architecture by checking common attributes.
        """
        m = self.lm
        blocks = None
        # Qwen and many GPT-like: model.transformer.h (list of blocks)
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            blocks = list(m.transformer.h)
        # Some architectures use model.model.layers
        elif hasattr(m, "model") and hasattr(m.model, "layers"):
            blocks = list(m.model.layers)
        # Another common: m.base_model.model.layers
        elif hasattr(m, "base_model") and hasattr(m.base_model, "model") and hasattr(m.base_model.model, "layers"):
            blocks = list(m.base_model.model.layers)
        if blocks is None:
            raise RuntimeError("Unsupported model architecture: cannot locate transformer blocks.")
        return blocks

    def _apply_freezing(self, train_top_fraction: float) -> None:
        # Determine blocks and which to keep trainable
        blocks = self._get_transformer_blocks()
        num_layers = len(blocks)
        keep = max(1, math.ceil(train_top_fraction * num_layers))
        # Indices for trainable blocks (top fraction): last 'keep' blocks
        trainable_idx = set(range(num_layers - keep, num_layers))

        # Freeze bottom layers
        for i, block in enumerate(blocks):
            for p in block.parameters():
                p.requires_grad = (i in trainable_idx)

        # Unfreeze LM head
        lm_head = getattr(self.lm, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad = True

        # Unfreeze answer head
        for p in self.answer_head.parameters():
            p.requires_grad = True

        # Ensure value head is trainable
        for p in self.value_head.parameters():
            p.requires_grad = True

        # Note: Embedding-level freezing for <z*> vs others is handled in tokenizer_ext via gradient masking.

        # Store for reporting
        self._num_layers = num_layers
        self._num_unfrozen_layers = len(trainable_idx)

    def _log_param_counts(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(1, total)
        print(f"PolicyModel params: total={total} trainable={trainable} ({pct:.2f}%)")
        print(f"Unfrozen transformer layers: {self._num_unfrozen_layers}/{self._num_layers}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = True,
    ) -> PolicyForwardOutput:
        """Forward that returns logits, hidden_states (last layer at least), and values.

        Values are computed for every token position from the final hidden layer.
        The rollout logic selects value estimates corresponding to action steps.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits  # [B, T, V]
        hidden_states = outputs.hidden_states  # tuple of layer outputs
        # Use last layer hidden state; shape [B, T, H]
        last_hidden = hidden_states[-1]

        values = self.value_head(last_hidden)  # [B, T, 1]
        values = values.squeeze(-1)   # [B, T]
        answer_logits = self.answer_head(last_hidden)  # [B, T, 10]
        return PolicyForwardOutput(
            logits=logits,
            hidden_states=hidden_states if return_hidden_states else (hidden_states[-1],),
            values=values,
            answer_logits=answer_logits,
        )


__all__ = ["PolicyModel", "PolicyForwardOutput"]

