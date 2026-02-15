from __future__ import annotations

from PPO.conf import Config
from PPO.train import train


def main() -> None:
    cfg = Config()
    cfg.train.updates = 1
    cfg.rollout.episodes_per_batch = 10
    cfg.rollout.max_tokens_per_batch = 256
    cfg.rollout.max_new_tokens = 16
    cfg.ppo.ppo_epochs = 1
    cfg.ppo.minibatch_size = 4
    cfg.train.save_every = 1
    cfg.train.keep_last = 1
    cfg.train.output_dir = "./runs/ppo_smoke"
    train(cfg)


if __name__ == "__main__":
    main()
