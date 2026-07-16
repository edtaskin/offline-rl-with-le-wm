import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


SENSITIVE_ARG_PATTERNS = ("token", "password", "secret")


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def args_for_config(args):
    config_args = vars(args).copy()
    for key, value in config_args.items():
        if value is not None and any(pattern in key.lower() for pattern in SENSITIVE_ARG_PATTERNS):
            config_args[key] = "***"
    return config_args


def sidecar_path(checkpoint_path, suffix):
    path = Path(checkpoint_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}"))
    return f"{checkpoint_path}{suffix}"


def write_run_config(args, dataset_stats=None, extra=None):
    payload = {
        "script": "src/bc/train_bc_latent.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": args_for_config(args),
    }
    if dataset_stats is not None:
        payload["dataset_stats"] = dataset_stats
    if extra is not None:
        payload.update(extra)
    run_config_path = sidecar_path(args.checkpoint_path, "_run_config.json")
    run_config_dir = os.path.dirname(run_config_path)
    if run_config_dir:
        os.makedirs(run_config_dir, exist_ok=True)
    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2, sort_keys=True)
        file.write("\n")
    return run_config_path


def init_wandb(args, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested with --wandb, but wandb is not installed. "
            "Install requirements.txt or run without --wandb."
        ) from exc
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "config": json_safe(config),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def log_wandb_artifact(run, artifact_name, artifact_type, file_paths, metadata=None):
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=json_safe(metadata or {}),
    )
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            artifact.add_file(file_path)
    run.log_artifact(artifact)
