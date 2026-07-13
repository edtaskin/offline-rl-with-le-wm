"""Train PPO on ``swm/PushT-v1``.

Run from the project root, e.g.::

    python -m src.ppo.train --total-timesteps 3000000 --num-envs 8
    python -m src.ppo.train --track --exp-name pusht_run1

Every field of :class:`~src.ppo.config.Config` is exposed as a CLI flag
(``snake_case`` -> ``--kebab-case``).
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from src.ppo.config import Config
from src.ppo.ppo import PPOTrainer


def _add_args(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    for f in fields(Config):
        if not f.init:
            continue
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if f.type == "bool" or isinstance(default, bool):
            # Support both --flag and --no-flag for booleans.
            parser.add_argument(
                flag, dest=f.name, action="store_true", default=default
            )
            parser.add_argument(
                "--no-" + f.name.replace("_", "-"), dest=f.name, action="store_false"
            )
        elif isinstance(default, tuple):
            parser.add_argument(
                flag, dest=f.name, type=int, nargs="+", default=list(default)
            )
        elif default is None:
            parser.add_argument(flag, dest=f.name, type=float, default=None)
        else:
            parser.add_argument(flag, dest=f.name, type=type(default), default=default)


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description="PPO on swm/PushT-v1")
    _add_args(parser)
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items()}
    if isinstance(kwargs.get("hidden_dims"), list):
        kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    return Config(**kwargs)


def main() -> None:
    cfg = parse_config()
    print("Config:")
    for f in fields(Config):
        print(f"  {f.name} = {getattr(cfg, f.name)}")

    trainer = PPOTrainer(cfg)

    if cfg.track:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.exp_name}__seed{cfg.seed}",
            config=cfg.__dict__,
            save_code=True,
        )
        trainer.writer = run

    try:
        trainer.train()
    finally:
        if cfg.track:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
