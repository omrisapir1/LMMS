from __future__ import annotations

import argparse

try:
    from .config import CodebookConfig
    from .trainer import CodebookTrainer
except ImportError:
    from config import CodebookConfig  # type: ignore
    from trainer import CodebookTrainer  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase2 EMA-VQ codebook")
    p.add_argument("--input_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--vocab_size", default=512, type=int)
    p.add_argument("--dim", default=1536, type=int)
    p.add_argument("--batch_size", default=2048, type=int, help="Number of sequences per training step")
    p.add_argument("--epochs", default=3, type=int)
    p.add_argument("--ema_decay", default=0.995, type=float)
    p.add_argument("--beta", default=0.25, type=float)
    p.add_argument("--lambda_kl", default=0.01, type=float)
    p.add_argument("--kmeans_sample_size", default=500000, type=int)
    p.add_argument("--no_kmeans", action="store_true")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CodebookConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        dim=args.dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ema_decay=args.ema_decay,
        beta=args.beta,
        lambda_kl=args.lambda_kl,
        kmeans_sample_size=args.kmeans_sample_size,
        no_kmeans=bool(args.no_kmeans),
        seed=args.seed,
    )
    trainer = CodebookTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
