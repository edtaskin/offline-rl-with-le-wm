"""Chunk-level latent PPO fine-tuning of the Latent BC agent on ``swm/PushT-v1``.

Requires the official LeWM object checkpoint in the swm cache
(``<cache>/checkpoints/pusht/lewm_object.ckpt`` -- see
``scripts/download_lewm_checkpoint.py``) and a trained BC checkpoint from the
Hugging Face Hub. Runnable either way::

    python -m src.ppo.train --smoke
    python src/ppo/train.py --fixed_target --frame_stack 3 --action_chunk_size 5

Flags accept both dash and underscore spellings (``--fixed-target`` ==
``--fixed_target``). The agent *contract* (frame_stack/frame_stride/
action_chunk_size/latent_dim/hidden_dim/action_dim) is read from the BC
``_stats.pth`` so it matches the checkpoint; explicit CLI flags still win.
Precedence: CLI > stats > defaults.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

# Allow ``python src/ppo/train.py`` (not just ``-m``) by putting the repo
# root on sys.path before the ``src.*`` imports, mirroring src/bc/train_bc_latent.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001 - dotenv is optional
    pass

import torch

from src.ppo.config import LatentConfig
from src.utils.hf_hub import resolve_artifact

CONTRACT_FIELDS = (
    "frame_stack",
    "frame_stride",
    "action_chunk_size",
    "latent_dim",
    "hidden_dim",
    "action_dim",
)

# Tiny overrides for --smoke: a couple of quick iterations end-to-end.
SMOKE_OVERRIDES = dict(
    num_envs=2,
    num_chunks=4,
    total_timesteps=2 * 4 * 5 * 2,  # ~2 iterations
    update_epochs=2,
    num_minibatches=2,
    max_episode_steps=40,
    save_interval=1,
    log_interval=1,
)

# str fields whose dataclass default is None (so type can't be inferred).
_NULLABLE_STR_FIELDS = {
    "encoder_checkpoint",
    "wandb_entity",
    "hf_repo_id",
    "hf_token",
    "hf_revision",
    "hf_path_prefix",
}


def _flag_names(name: str, prefix: str = "--") -> list[str]:
    """Both dash and underscore spellings of a flag (deduplicated)."""
    dash = prefix + name.replace("_", "-")
    under = prefix + name
    return [dash] if dash == under else [dash, under]


def _add_args(parser: argparse.ArgumentParser) -> None:
    """Add one flag per init field, leaving unset flags absent (SUPPRESS).

    Each flag is registered with both ``--dash-case`` and ``--snake_case``
    spellings so the BC-style underscore flags work here too.
    """
    defaults = LatentConfig()
    for f in fields(LatentConfig):
        if not f.init:
            continue
        names = _flag_names(f.name)
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(
                *names, dest=f.name, action="store_true", default=argparse.SUPPRESS
            )
            parser.add_argument(
                *_flag_names(f.name, prefix="--no-"),
                dest=f.name,
                action="store_false",
                default=argparse.SUPPRESS,
            )
        elif f.name == "target_kl":
            parser.add_argument(*names, dest=f.name, type=float, default=argparse.SUPPRESS)
        elif f.name in _NULLABLE_STR_FIELDS:
            parser.add_argument(*names, dest=f.name, type=str, default=argparse.SUPPRESS)
        else:
            parser.add_argument(
                *names, dest=f.name, type=type(default), default=argparse.SUPPRESS
            )

def _stats_contract(stats_path: str) -> dict:
    resolved_stats_path = resolve_artifact(stats_path)
    stats = torch.load(resolved_stats_path, map_location="cpu")
    overrides = {k: int(stats[k]) for k in CONTRACT_FIELDS if k in stats}
    if overrides:
        print(f"Contract from {stats_path} ({resolved_stats_path}): {overrides}")
    return overrides


def parse_config() -> LatentConfig:
    parser = argparse.ArgumentParser(description="Chunk-level latent PPO on swm/PushT-v1")
    _add_args(parser)
    parser.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    args = vars(parser.parse_args())
    smoke = args.pop("smoke", False)

    defaults = {f.name: getattr(LatentConfig(), f.name) for f in fields(LatentConfig) if f.init}
    stats_path = args.get("bc_stats", defaults["bc_stats"])
    stats_overrides = _stats_contract(stats_path)

    # Precedence: CLI (args) > smoke > stats > defaults.
    merged = {**defaults, **stats_overrides}
    if smoke:
        merged.update(SMOKE_OVERRIDES)
    merged.update(args)
    return LatentConfig(**merged)


def main() -> None:
    cfg = parse_config()
    # Fail fast on a misconfigured HF push rather than after a full training run.
    if cfg.push_to_hf and not cfg.hf_repo_id:
        raise ValueError("--push_to_hf requires --hf_repo_id (e.g. your-username/pusht-latent-ppo)")
    print("Config:")
    for f in fields(LatentConfig):
        print(f"  {f.name} = {getattr(cfg, f.name)}")

    from src.ppo.ppo import LatentPPOTrainer

    trainer = LatentPPOTrainer(cfg)

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
        if cfg.push_to_hf:
            _push_run_artifacts(cfg, trainer.run_dir)
    finally:
        if cfg.track:
            import wandb

            wandb.finish()


def _push_run_artifacts(cfg: LatentConfig, run_dir: Path) -> None:
    """Publish what a run needs to be re-analysed, not only ``best.pt``.

    ``final.pt`` is the no-selection control, and a dream run's
    ``selection_log.jsonl`` carries the paired imagined/real measurements the
    optimism analysis reads. Without them a published run can only be re-scored,
    not re-examined.

    Everything lands under ``cfg.hf_path_prefix``. With no prefix the files go to
    the repository root, where ``best.pt``/``final.pt`` already exist from
    earlier runs and would be overwritten -- campaigns should always set one.
    """
    from src.utils.hf_hub import push_files_to_hub

    if not cfg.hf_repo_id:
        raise ValueError("--push_to_hf requires --hf_repo_id (e.g. your-username/pusht-latent-ppo)")
    candidates = ["best.pt", "final.pt", "selection_log.jsonl"]
    paths = [str(run_dir / name) for name in candidates if (run_dir / name).is_file()]
    if not paths:
        raise FileNotFoundError(f"No publishable artifacts in {run_dir}")
    if not cfg.hf_path_prefix:
        print(
            "warning: --push_to_hf without --hf_path_prefix writes to the repository "
            "root and can overwrite artifacts from earlier runs"
        )
    result = push_files_to_hub(
        repo_id=cfg.hf_repo_id,
        file_paths=paths,
        repo_type=cfg.hf_repo_type,
        private=cfg.hf_private,
        token=cfg.hf_token,
        revision=cfg.hf_revision,
        path_prefix=cfg.hf_path_prefix,
        commit_message=cfg.hf_commit_message,
    )
    destination = f"{cfg.hf_path_prefix.rstrip('/')}/" if cfg.hf_path_prefix else ""
    for path in paths:
        print(f"Pushed {destination}{Path(path).name} to {result.repo_url}")


def _push_checkpoint_to_hf(cfg: LatentConfig, checkpoint_path: Path) -> None:
    from src.utils.hf_hub import push_files_to_hub

    if not cfg.hf_repo_id:
        raise ValueError("--push_to_hf requires --hf_repo_id (e.g. your-username/pusht-latent-ppo)")
    result = push_files_to_hub(
        repo_id=cfg.hf_repo_id,
        file_paths=[str(checkpoint_path)],
        repo_type=cfg.hf_repo_type,
        private=cfg.hf_private,
        token=cfg.hf_token,
        revision=cfg.hf_revision,
        path_prefix=cfg.hf_path_prefix,
        commit_message=cfg.hf_commit_message,
    )
    print(f"Pushed checkpoint to Hugging Face Hub: {result.repo_url}")


if __name__ == "__main__":
    main()
