from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch

try:
    from .config import CodebookConfig
    from .dataset import iter_sequence_batches
    from .kmeans_init import init_codebook_with_kmeans, random_init_codebook
    from .losses import UsageKLLoss, compute_losses
    from .model import Codebook
except ImportError:
    from config import CodebookConfig  # type: ignore
    from dataset import iter_sequence_batches  # type: ignore
    from kmeans_init import init_codebook_with_kmeans, random_init_codebook  # type: ignore
    from losses import UsageKLLoss, compute_losses  # type: ignore
    from model import Codebook  # type: ignore


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CodebookTrainer:
    def __init__(self, config: CodebookConfig):
        self.config = config
        _set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Codebook(
            dim=config.dim,
            vocab_size=config.vocab_size,
            ema_decay=config.ema_decay,
        ).to(self.device)
        self.usage_kl = UsageKLLoss(
            vocab_size=config.vocab_size,
            alpha=config.usage_laplace_alpha,
            eta=config.usage_ema_eta,
        ).to(self.device)

    def train(self) -> None:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        print(
            f"[init] device={self.device.type} dim={cfg.dim} vocab={cfg.vocab_size} "
            f"batch_size(sequences)={cfg.batch_size} epochs={cfg.epochs}"
        )
        print("[target] healthy run usually shows perplexity > 100 and dead_fraction trending < 0.2")

        if cfg.no_kmeans:
            init_stats = random_init_codebook(self.model, seed=cfg.seed)
            print("[init] using random normal initialization (--no_kmeans)")
        else:
            init_stats = init_codebook_with_kmeans(
                model=self.model,
                input_dir=cfg.input_dir,
                dim=cfg.dim,
                sample_size=cfg.kmeans_sample_size,
                read_batch_size=cfg.read_batch_size,
                fit_batch_size=cfg.kmeans_fit_batch_size,
                seed=cfg.seed,
            )
            print(
                f"[init] kmeans complete sampled_vectors={init_stats['sampled_vectors']} "
                f"(target up to {cfg.kmeans_sample_size})"
            )

        global_step = 0
        total_sequences = 0
        total_vectors = 0

        self.model.train()
        for epoch in range(cfg.epochs):
            print(f"[epoch {epoch + 1}/{cfg.epochs}] starting")
            for batch in iter_sequence_batches(
                cfg.input_dir,
                batch_size=cfg.batch_size,
                dim=cfg.dim,
                read_batch_size=cfg.read_batch_size,
            ):
                if batch.sequence_count == 0:
                    continue
                global_step += 1
                total_sequences += batch.sequence_count
                total_vectors += batch.vector_count

                latents = batch.latents.to(self.device, dtype=torch.float32, non_blocking=True)

                with torch.no_grad():
                    z_ids, quantized = self.model(latents)
                    metrics = compute_losses(
                        latents=latents,
                        quantized_vectors=quantized,
                        z_ids=z_ids,
                        usage_kl=self.usage_kl,
                        beta=cfg.beta,
                        lambda_kl=cfg.lambda_kl,
                    )
                    self.model.ema_update(latents, z_ids, eps=cfg.eps)

                if global_step % cfg.log_interval == 0 or global_step == 1:
                    perplexity = float(metrics.perplexity.item())
                    print(
                        f"step={global_step} "
                        f"total_loss={float(metrics.total_loss.item()):.6f} "
                        f"vq_loss={float(metrics.vq_loss.item()):.6f} "
                        f"commit_loss={float(metrics.commit_loss.item()):.6f} "
                        f"kl_loss={float(metrics.kl_loss.item()):.6f} "
                        f"perplexity={perplexity:.2f} "
                        f"dead_fraction={float(metrics.dead_fraction.item()):.4f} "
                        f"effective_vocab={int(round(perplexity))}"
                    )

            print(f"[epoch {epoch + 1}/{cfg.epochs}] done")

        ckpt_path = os.path.join(cfg.output_dir, "codebook.pt")
        meta_path = os.path.join(cfg.output_dir, "meta.json")

        torch.save(
            {
                "dim": cfg.dim,
                "vocab_size": cfg.vocab_size,
                "ema_decay": cfg.ema_decay,
                "beta": cfg.beta,
                "lambda_kl": cfg.lambda_kl,
                "normalization": "cosine",
                "embeddings": self.model.embeddings.detach().cpu(),
                "ema_cluster_size": self.model.ema_cluster_size.detach().cpu(),
                "ema_embedding_sum": self.model.ema_embedding_sum.detach().cpu(),
                "usage_p_ema": self.usage_kl.p_ema.detach().cpu(),
                "config": asdict(cfg),
                "steps": global_step,
                "sequences_seen": total_sequences,
                "vectors_seen": total_vectors,
            },
            ckpt_path,
        )

        meta = {
            "dim": cfg.dim,
            "vocab_size": cfg.vocab_size,
            "ema_decay": cfg.ema_decay,
            "beta": cfg.beta,
            "lambda_kl": cfg.lambda_kl,
            "normalization": "cosine",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"[done] saved checkpoint: {ckpt_path}")
        print(f"[done] saved metadata:   {meta_path}")
