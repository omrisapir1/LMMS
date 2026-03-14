from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

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
            f"max_vectors_per_batch={cfg.max_vectors_per_batch} max_sequences_per_batch={cfg.batch_size} epochs={cfg.epochs}"
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
                kmeans_max_vectors_per_sequence=cfg.kmeans_max_vectors_per_sequence,
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
                max_vectors_per_batch=cfg.max_vectors_per_batch,
                max_sequences_per_batch=cfg.batch_size,
                dim=cfg.dim,
                read_batch_size=cfg.read_batch_size,
            ):
                if batch.sequence_count == 0:
                    continue
                global_step += 1
                total_sequences += batch.sequence_count
                total_vectors += batch.vector_count

                latents = batch.latents.to(self.device, dtype=torch.float32, non_blocking=True)
                should_log = global_step % cfg.log_interval == 0 or global_step == 1
                diagnostics: dict[str, float] = {}

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
                    if should_log:
                        x_norm = F.normalize(latents, p=2, dim=-1, eps=1e-12)
                        q_norm = F.normalize(quantized, p=2, dim=-1, eps=1e-12)
                        if x_norm.shape[0] == 0:
                            diagnostics = {
                                "cos_sim_mean": float("nan"),
                                "cos_sim_std": float("nan"),
                                "cos_sim_p50": float("nan"),
                                "cos_sim_p90": float("nan"),
                                "margin_mean": float("nan"),
                                "margin_std": float("nan"),
                                "margin_p50": float("nan"),
                                "margin_p90": float("nan"),
                                "latent_norm_mean": float("nan"),
                                "latent_norm_std": float("nan"),
                            }
                        else:
                            cos_sim = (x_norm * q_norm).sum(dim=-1)
                            cos_quantiles = torch.quantile(
                                cos_sim,
                                torch.tensor([0.5, 0.9], device=cos_sim.device, dtype=cos_sim.dtype),
                            )

                            e_norm = F.normalize(self.model.embeddings, p=2, dim=-1, eps=1e-12)
                            if e_norm.shape[0] < 2:
                                margin = torch.zeros_like(cos_sim)
                            else:
                                margin_chunks = []
                                margin_chunk_size = 16_384
                                for start in range(0, x_norm.shape[0], margin_chunk_size):
                                    sims = x_norm[start : start + margin_chunk_size] @ e_norm.t()
                                    top2_vals = torch.topk(sims, k=2, dim=-1).values
                                    margin_chunks.append(top2_vals[:, 0] - top2_vals[:, 1])
                                margin = torch.cat(margin_chunks, dim=0)
                            margin_quantiles = torch.quantile(
                                margin,
                                torch.tensor([0.5, 0.9], device=margin.device, dtype=margin.dtype),
                            )

                            latent_norm = latents.norm(dim=-1)
                            diagnostics = {
                                "cos_sim_mean": float(cos_sim.mean().item()),
                                "cos_sim_std": float(cos_sim.std(unbiased=False).item()),
                                "cos_sim_p50": float(cos_quantiles[0].item()),
                                "cos_sim_p90": float(cos_quantiles[1].item()),
                                "margin_mean": float(margin.mean().item()),
                                "margin_std": float(margin.std(unbiased=False).item()),
                                "margin_p50": float(margin_quantiles[0].item()),
                                "margin_p90": float(margin_quantiles[1].item()),
                                "latent_norm_mean": float(latent_norm.mean().item()),
                                "latent_norm_std": float(latent_norm.std(unbiased=False).item()),
                            }
                    self.model.ema_update(latents, z_ids, eps=cfg.eps)

                if should_log:
                    perplexity = float(metrics.perplexity.item())
                    print(
                        f"step={global_step} "
                        f"seqs={batch.sequence_count} "
                        f"vecs={batch.vector_count} "
                        f"avg_k={batch.avg_k_in_batch:.2f} "
                        f"total_loss={float(metrics.total_loss.item()):.6f} "
                        f"vq_loss={float(metrics.vq_loss.item()):.6f} "
                        f"commit_loss={float(metrics.commit_loss.item()):.6f} "
                        f"kl_loss={float(metrics.kl_loss.item()):.6f} "
                        f"perplexity={perplexity:.2f} "
                        f"dead_fraction={float(metrics.dead_fraction.item()):.4f} "
                        f"effective_vocab={int(round(perplexity))} "
                        f"cos_mean={diagnostics['cos_sim_mean']:.4f} "
                        f"cos_std={diagnostics['cos_sim_std']:.4f} "
                        f"cos_p50={diagnostics['cos_sim_p50']:.4f} "
                        f"cos_p90={diagnostics['cos_sim_p90']:.4f} "
                        f"margin_mean={diagnostics['margin_mean']:.4f} "
                        f"margin_std={diagnostics['margin_std']:.4f} "
                        f"margin_p50={diagnostics['margin_p50']:.4f} "
                        f"margin_p90={diagnostics['margin_p90']:.4f} "
                        f"latent_norm={diagnostics['latent_norm_mean']:.4f} "
                        f"latent_norm_std={diagnostics['latent_norm_std']:.4f}"
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
