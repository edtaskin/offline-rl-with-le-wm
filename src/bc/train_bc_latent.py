import os
import sys
import importlib
import json
import random
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.bc.dataset import (
    LEWM_IMAGE_MEAN,
    LEWM_IMAGE_NORMALIZATION,
    LEWM_IMAGE_SIZE,
    LEWM_IMAGE_STD,
    PushTLeWMDataset,
    PushTLeWMLatentDataset,
)
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.utils.hf_hub import push_files_to_hub


SENSITIVE_ARG_PATTERNS = ("token", "password", "secret")


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _args_for_config(args):
    config_args = vars(args).copy()
    for key, value in config_args.items():
        if value is not None and any(pattern in key.lower() for pattern in SENSITIVE_ARG_PATTERNS):
            config_args[key] = "***"
    return config_args


def _sidecar_path(checkpoint_path, suffix):
    path = Path(checkpoint_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}"))
    return f"{checkpoint_path}{suffix}"


def _write_run_config(args, dataset_stats=None, extra=None):
    payload = {
        "script": "src/bc/train_bc_latent.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": _args_for_config(args),
    }
    if dataset_stats is not None:
        payload["dataset_stats"] = dataset_stats
    if extra is not None:
        payload.update(extra)

    run_config_path = _sidecar_path(args.checkpoint_path, "_run_config.json")
    run_config_dir = os.path.dirname(run_config_path)
    if run_config_dir:
        os.makedirs(run_config_dir, exist_ok=True)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return run_config_path


def _init_wandb(args, config):
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
        "config": _json_safe(config),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def _log_wandb_artifact(run, artifact_name, artifact_type, file_paths, metadata=None):
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=_json_safe(metadata or {}),
    )
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            artifact.add_file(file_path)
    run.log_artifact(artifact)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_stable_worldmodel():
    le_wm_path = os.getenv("LE_WM_PATH")
    if le_wm_path is None:
        raise ValueError("LE_WM_PATH environment variable not set")
    if le_wm_path not in sys.path:
        sys.path.insert(0, le_wm_path)
    return importlib.import_module("stable_worldmodel")


def _path_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _default_latent_cache_path(data_path):
    data_path = Path(data_path)
    cache_name = f"{data_path.stem}_lewm_object_imagenet224_cls.pt"
    return str(Path("data", "latent_cache", cache_name))


def _num_samples_from_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    try:
        return int(data["actions"].shape[0])
    finally:
        data.close()


def _latent_cache_metadata(data_path, ckpt_path, num_samples, latent_dim):
    return {
        "format_version": 1,
        "cache_type": "lewm_cls_latents",
        "data_file": _path_fingerprint(data_path),
        "lewm_checkpoint_file": _path_fingerprint(ckpt_path),
        "num_samples": int(num_samples),
        "latent_dim": int(latent_dim),
        "image_size": list(LEWM_IMAGE_SIZE),
        "image_normalization": LEWM_IMAGE_NORMALIZATION,
        "image_mean": list(LEWM_IMAGE_MEAN),
        "image_std": list(LEWM_IMAGE_STD),
    }


def _metadata_matches(actual, expected):
    if not isinstance(actual, dict):
        return False, ["metadata is missing or not a dict"]

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    return len(mismatches) == 0, mismatches


def _check_latent_cache(cache_path, expected_metadata):
    if not cache_path or not os.path.exists(cache_path):
        return False, ["cache file does not exist"]
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        return False, [f"cache could not be loaded: {exc}"]

    if not isinstance(payload, dict) or "latents" not in payload:
        return False, ["cache payload must be a dict containing a 'latents' tensor"]
    latents = payload["latents"]
    if not torch.is_tensor(latents):
        return False, ["cache 'latents' entry is not a tensor"]
    expected_shape = (
        int(expected_metadata["num_samples"]),
        int(expected_metadata["latent_dim"]),
    )
    if tuple(latents.shape) != expected_shape:
        return False, [f"latents shape mismatch: expected {expected_shape}, got {tuple(latents.shape)}"]

    return _metadata_matches(payload.get("metadata"), expected_metadata)


def _load_lewm_encoder(ckpt_path, device):
    print("Loading official LeWM object checkpoint...")
    print(ckpt_path)
    lewm_model = torch.load(ckpt_path, map_location=device, weights_only=False)
    lewm_encoder = lewm_model.encoder.to(device)
    lewm_encoder.eval()
    for param in lewm_encoder.parameters():
        param.requires_grad = False
    print("Successfully loaded and frozen the LeWM Encoder from the official checkpoint!")
    return lewm_encoder


def _build_latent_cache(source_dataset, lewm_encoder, device, cache_path, metadata, batch_size):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if batch_size < 1:
        raise ValueError("latent_cache_batch_size must be at least 1")

    print(f"Building LeWM latent cache: {cache_path}")
    all_latents = []
    with torch.inference_mode():
        for start in range(0, len(source_dataset), batch_size):
            end = min(start + batch_size, len(source_dataset))
            obs_batch = source_dataset.image_transform(source_dataset.images[start:end]).to(device)
            encoder_outputs = lewm_encoder(obs_batch)
            latents = encoder_outputs.last_hidden_state[:, 0, :].detach().cpu()
            all_latents.append(latents)
            if start == 0 or end == len(source_dataset):
                print(f"  encoded {end}/{len(source_dataset)} frames")

    cached_latents = torch.cat(all_latents, dim=0).contiguous()
    torch.save(
        {
            "latents": cached_latents,
            "metadata": {
                **metadata,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        cache_path,
    )
    print(f"Saved LeWM latent cache with shape {tuple(cached_latents.shape)}")


def train_latent_bc(args):
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.latent_cache_batch_size is not None and args.latent_cache_batch_size < 1:
        raise ValueError("latent_cache_batch_size must be at least 1")
    seed_everything(args.seed, args.deterministic)
    print(f"Using seed: {args.seed} (deterministic={args.deterministic})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Prepare dataset. By default, cache per-frame LeWM CLS latents once and
    # rebuild stacked histories cheaply from that cache on later runs.
    latent_dim = 192
    swm = _load_stable_worldmodel()
    ckpt_path = Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")
    use_latent_cache = not args.disable_latent_cache
    latent_cache_path = args.latent_cache_path or _default_latent_cache_path(args.data_path)
    latent_cache_rebuilt = False
    expected_cache_metadata = None
    lewm_encoder = None

    if use_latent_cache:
        num_samples = _num_samples_from_data(args.data_path)
        expected_cache_metadata = _latent_cache_metadata(
            args.data_path,
            ckpt_path,
            num_samples=num_samples,
            latent_dim=latent_dim,
        )
        cache_ok, cache_messages = _check_latent_cache(latent_cache_path, expected_cache_metadata)
        if args.rebuild_latent_cache:
            cache_ok = False
            cache_messages = ["cache rebuild was requested"]

        if cache_ok:
            print(f"Using existing LeWM latent cache: {latent_cache_path}")
        else:
            print(f"LeWM latent cache will be built at: {latent_cache_path}")
            for message in cache_messages[:5]:
                print(f"  cache miss: {message}")
            source_dataset = PushTLeWMDataset(
                args.data_path,
                frame_stack=1,
                frame_stride=1,
                action_chunk_size=1,
            )
            lewm_encoder = _load_lewm_encoder(ckpt_path, device)
            cache_batch_size = args.latent_cache_batch_size or args.batch_size
            _build_latent_cache(
                source_dataset,
                lewm_encoder,
                device,
                latent_cache_path,
                expected_cache_metadata,
                batch_size=cache_batch_size,
            )
            latent_cache_rebuilt = True

        dataset = PushTLeWMLatentDataset(
            args.data_path,
            latent_cache_path,
            frame_stack=args.frame_stack,
            frame_stride=args.frame_stride,
            action_chunk_size=args.action_chunk_size,
        )
    else:
        print("Latent cache disabled; images will be encoded through LeWM during every epoch.")
        dataset = PushTLeWMDataset(
            args.data_path,
            frame_stack=args.frame_stack,
            frame_stride=args.frame_stride,
            action_chunk_size=args.action_chunk_size,
        )
        lewm_encoder = _load_lewm_encoder(ckpt_path, device)

    latent_dim = int(dataset.stats.get('latent_dim', latent_dim))
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=dataloader_generator,
    )

    run_metadata = {
        **dataset.stats,
        'frame_stack': args.frame_stack,
        'frame_stride': args.frame_stride,
        'hidden_dim': args.hidden_dim,
        'latent_dim': latent_dim,
        'action_dim': 2,
        'action_chunk_size': args.action_chunk_size,
        'latent_cache_enabled': use_latent_cache,
        'latent_cache_path': latent_cache_path if use_latent_cache else None,
        'latent_cache_rebuilt': latent_cache_rebuilt,
        'latent_cache_expected_metadata': expected_cache_metadata,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'num_workers': args.num_workers,
    }
    run_config_path = _write_run_config(
        args,
        dataset_stats=dataset.stats,
        extra={
            "device": str(device),
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "status": "running",
            "bc_contract": run_metadata,
        },
    )
    wandb_run = _init_wandb(
        args,
        {
            **_args_for_config(args),
            "device": str(device),
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "bc_contract": run_metadata,
        },
    )
    if wandb_run is not None:
        run_url = getattr(wandb_run, "url", None)
        print(f"Logging run to wandb: {run_url or 'enabled'}")

    # 3. Initialize Latent BC Policy
    policy = LatentBCPolicy(
        latent_dim=latent_dim, 
        frame_stack=args.frame_stack,
        action_dim=2, 
        hidden_dim=args.hidden_dim,
        action_chunk_size=args.action_chunk_size,
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("Starting Latent-Space Behavior Cloning training loop...")

    policy.train()
    best_loss = float("inf")
    best_epoch = None
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for batch_inputs, batch_actions in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_actions = batch_actions.to(device)

            if use_latent_cache:
                stacked_latents = batch_inputs
            else:
                with torch.no_grad():
                    # Reshape to treat frames as a larger batch: (Batch * FrameStack, C, H, W)
                    b, f, c, h, w = batch_inputs.shape
                    flat_obs = batch_inputs.reshape(b * f, c, h, w)
                    encoder_outputs = lewm_encoder(flat_obs)
                    flat_latents = encoder_outputs.last_hidden_state[:, 0, :]
                    stacked_latents = flat_latents.reshape(b, f, latent_dim)

            # Predict a future action chunk and calculate loss
            predicted_action_chunks = policy(stacked_latents)
            loss = criterion(predicted_action_chunks, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": avg_loss,
                    "train/best_loss": best_loss,
                    "train/epoch": epoch + 1,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Average MSE Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_name = args.checkpoint_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(policy.state_dict(), checkpoint_name)

    torch.save(policy.state_dict(), args.checkpoint_path)
    torch.save(
        {
            **dataset.stats,
            'frame_stack': args.frame_stack,
            'frame_stride': args.frame_stride,
            'hidden_dim': args.hidden_dim,
            'latent_dim': latent_dim,
            'action_dim': 2,
            'action_chunk_size': args.action_chunk_size,
            'latent_cache_enabled': use_latent_cache,
            'latent_cache_path': latent_cache_path if use_latent_cache else None,
            'latent_cache_rebuilt': latent_cache_rebuilt,
            'latent_cache_expected_metadata': expected_cache_metadata,
            'seed': args.seed,
            'deterministic': args.deterministic,
            'num_workers': args.num_workers,
        },
        args.checkpoint_path.replace('.pth', '_stats.pth')
    )
    stats_path = args.checkpoint_path.replace('.pth', '_stats.pth')
    run_config_path = _write_run_config(
        args,
        dataset_stats=dataset.stats,
        extra={
            "device": str(device),
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "status": "completed",
            "bc_contract": run_metadata,
            "final_checkpoint_path": args.checkpoint_path,
            "stats_path": stats_path,
            "run_config_path": run_config_path,
            "final_train_loss": avg_loss,
            "best_train_loss": best_loss,
            "best_train_loss_epoch": best_epoch,
            "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
        },
    )

    hf_upload_result = None
    if args.push_to_hf:
        hf_upload_result = push_files_to_hub(
            repo_id=args.hf_repo_id,
            file_paths=[args.checkpoint_path, stats_path, run_config_path],
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            token=args.hf_token,
            revision=args.hf_revision,
            path_prefix=args.hf_path_prefix,
            commit_message=args.hf_commit_message,
        )
        run_config_path = _write_run_config(
            args,
            dataset_stats=dataset.stats,
            extra={
                "device": str(device),
                "dataset_size": len(dataset),
                "num_batches_per_epoch": len(dataloader),
                "status": "completed",
                "bc_contract": run_metadata,
                "final_checkpoint_path": args.checkpoint_path,
                "stats_path": stats_path,
                "run_config_path": run_config_path,
                "final_train_loss": avg_loss,
                "best_train_loss": best_loss,
                "best_train_loss_epoch": best_epoch,
                "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
                "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
                "hf_upload": {
                    "repo_id": hf_upload_result.repo_id,
                    "repo_type": hf_upload_result.repo_type,
                    "repo_url": hf_upload_result.repo_url,
                    "uploaded_files": hf_upload_result.uploaded_files,
                },
            },
        )
        push_files_to_hub(
            repo_id=args.hf_repo_id,
            file_paths=[run_config_path],
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            token=args.hf_token,
            revision=args.hf_revision,
            path_prefix=args.hf_path_prefix,
            commit_message=args.hf_commit_message,
            create_repo=False,
        )
        print(f"Pushed checkpoint artifacts to Hugging Face Hub: {hf_upload_result.repo_url}")

    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = avg_loss
        wandb_run.summary["best_train_loss"] = best_loss
        wandb_run.summary["best_train_loss_epoch"] = best_epoch
        if hf_upload_result is not None:
            wandb_run.summary["hf/repo_url"] = hf_upload_result.repo_url
        _log_wandb_artifact(
            wandb_run,
            artifact_name=f"latent-bc-policy-{getattr(wandb_run, 'id', 'local')}",
            artifact_type="model",
            file_paths=[args.checkpoint_path, stats_path, run_config_path],
            metadata={
                "checkpoint_path": args.checkpoint_path,
                "stats_path": stats_path,
                "run_config_path": run_config_path,
                "best_train_loss": best_loss,
                "best_train_loss_epoch": best_epoch,
                "bc_contract": run_metadata,
                "hf_repo_url": hf_upload_result.repo_url if hf_upload_result is not None else None,
            },
        )
        wandb_run.finish()
    print(f"Latent BC Training Complete! Saved to: {args.checkpoint_path}")
    print(f"Run config saved to: {run_config_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latent-Space Behavior Cloning (BC) Prior for LeWorldModel")
    
    # Data and Logging Paths
    parser.add_argument("--data_path", type=str, default="data/expert_trajectories/pusht_expert.npz", 
                        help="Path to the converted expert dataset .npz file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/trained_policies/pusht_latent_bc.pth", 
                        help="Path to save the final trained policy weights")
    parser.add_argument(
        "--latent_cache_path",
        type=str,
        default=None,
        help="Path for cached per-frame LeWM CLS latents; defaults to data/latent_cache/<dataset>_lewm_object_imagenet224_cls.pt",
    )
    parser.add_argument(
        "--rebuild_latent_cache",
        action="store_true",
        help="Recompute and overwrite the LeWM latent cache before training",
    )
    parser.add_argument(
        "--disable_latent_cache",
        action="store_true",
        help="Disable latent caching and encode image batches through LeWM during every epoch",
    )
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for training")
    parser.add_argument(
        "--latent_cache_batch_size",
        type=int,
        default=None,
        help="Batch size for one-time cache construction; defaults to --batch_size",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic PyTorch/CuDNN behavior where available",
    )
    
    # Architecture and Context
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the BC MLP policy")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of LeWM latents to stack for temporal context")
    parser.add_argument("--frame_stride", type=int, default=5, help="Environment steps between stacked history frames")
    parser.add_argument("--action_chunk_size", type=int, default=5, help="Number of future actions to predict from one observation")
    
    # Logging and Saving Intervals
    parser.add_argument("--log_interval", type=int, default=10, help="Epochs to wait before logging loss metrics")
    parser.add_argument("--save_interval", type=int, default=10, help="Save policy checkpoint every N epochs")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="offline-rl-lewm", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default="pusht-latent-bc", help="Weights & Biases run group")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional Weights & Biases tags")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Weights & Biases mode; use offline on clusters without network access",
    )
    parser.add_argument("--push_to_hf", action="store_true", help="Push final checkpoint artifacts to the Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face repo id, for example username/repo-name")
    parser.add_argument(
        "--hf_repo_type",
        type=str,
        choices=["model", "dataset", "space"],
        default="model",
        help="Hugging Face repository type",
    )
    parser.add_argument(
        "--hf_private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create or keep the Hugging Face repo private",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token; if omitted, huggingface_hub uses the cached login or environment token",
    )
    parser.add_argument("--hf_revision", type=str, default=None, help="Optional Hugging Face branch or revision to upload to")
    parser.add_argument("--hf_path_prefix", type=str, default=None, help="Optional folder inside the Hugging Face repo")
    parser.add_argument(
        "--hf_commit_message",
        type=str,
        default="Upload latent BC checkpoint artifacts",
        help="Commit message for Hugging Face uploads",
    )

    args = parser.parse_args()
    if args.push_to_hf and not args.hf_repo_id:
        parser.error("--hf_repo_id is required when --push_to_hf is set")
    
    # Execute the training loop
    train_latent_bc(args)
