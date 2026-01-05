"""
Phase 1 evaluation: compute validation accuracy over <ANSWER> predictions.
Builds an eval dataloader using Phase1Dataset and computes exact 5-digit accuracy.
Evaluation policy: dataset is stage-agnostic; stage logic is injected via num_latent_fn.
"""
from typing import Any, List
import torch

class Evaluator:
    def __init__(self, max_length: int = 2024, batch_size: int = 64, max_thoughts: int = 8):
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_thoughts = max_thoughts

    def compute_accuracy(self, model: Any, tokenizer, preprocessed_items: List[dict], stage: int) -> float:
        from torch.utils.data import DataLoader
        from .dataset import Phase1Dataset, collate_fn
        ANSWER_TOKEN = "<ANSWER>"
        if len(preprocessed_items) == 0:
            return 0.0

        # Evaluation stage policy injected via num_latent_fn
        def eval_num_latent(s: int, K: int) -> int:
            if s < 8:
                return min(s, max(0, K - 1))
            else:
                return K

        # eval_num_latent never returns None; all examples are evaluated at all stages
        num_latent_fn = lambda K: eval_num_latent(stage, K)

        # Stage-dependent row filtering by K
        # - Stages 1–7: include all rows except those with K == 1
        # - Stage 8: include all rows
        if 1 <= stage <= 7:
            filtered_items = [rec for rec in preprocessed_items if rec["K"] != 1]
        else:
            filtered_items = preprocessed_items

        print(
            f"[Stage {stage}] Training with latent={stage}. "
            f" Eval size={len(filtered_items)}. "
        )
        if len(filtered_items) == 0:
            return 0.0

        # Construct dataset with required args
        dataset = Phase1Dataset(
            items=filtered_items,
            tokenizer=tokenizer,
            max_length=self.max_length,
            num_latent_fn=num_latent_fn,
            max_thoughts=self.max_thoughts,
            answer_token=ANSWER_TOKEN,
            debug=True,
        )
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id))

        # Prepare model for evaluation
        model.eval()
        correct = 0
        total = 0
        answer_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        with torch.no_grad():
            for batch in loader:
                # Move tensors to model device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                # Safety check: ensure <ANSWER> is present exactly once per sample
                per_row_answer_count = (batch["input_ids"] == answer_id).sum(dim=1)
                assert per_row_answer_count.eq(1).all().item(), "<ANSWER> token truncated or duplicated in evaluation inputs"

                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])  # no digit_labels in eval forward
                logits = out.get("logits")  # [B,5,10]
                if logits is None:
                    continue
                preds = torch.argmax(logits, dim=-1)  # [B,5]
                labels = batch["digit_labels"]  # [B,5]
                # Exact 5-digit match
                match = (preds == labels).all(dim=1)  # [B]
                correct += int(match.sum().item())
                total += preds.shape[0]
        if total == 0:
            print("Warning: evaluation produced zero samples")
            return 0.0
        return correct / float(total)
