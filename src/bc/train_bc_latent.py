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

le_wm_path = os.getenv("LE_WM_PATH")
if le_wm_path is None:
    raise ValueError("LE_WM_PATH environment variable not set")
if le_wm_path not in sys.path:
    sys.path.insert(0, le_wm_path)
swm = importlib.import_module("stable_worldmodel")

from src.bc.dataset import PushTLeWMDataset
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy


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


def _sidecar_path(checkpoint_path, suffix):
    path = Path(checkpoint_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}"))
    return f"{checkpoint_path}{suffix}"


def _write_run_config(args, dataset_stats=None, extra=None):
    payload = {
        "script": "src/bc/train_bc_latent.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
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


def train_latent_bc(args):
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    seed_everything(args.seed, args.deterministic)
    print(f"Using seed: {args.seed} (deterministic={args.deterministic})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load Dataset with temporal frame history
    dataset = PushTLeWMDataset(
        args.data_path,
        frame_stack=args.frame_stack,
        frame_stride=args.frame_stride,
        action_chunk_size=args.action_chunk_size,
    )
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
        'latent_dim': 192,
        'action_dim': 2,
        'action_chunk_size': args.action_chunk_size,
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
            **vars(args),
            "device": str(device),
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(dataloader),
            "bc_contract": run_metadata,
        },
    )
    if wandb_run is not None:
        run_url = getattr(wandb_run, "url", None)
        print(f"Logging run to wandb: {run_url or 'enabled'}")

    # 2. Load the Pre-Trained LeWM Encoder (Frozen)
    print("Loading official LeWM object checkpoint...")

    # Get the path to where your conversion script just saved the checkpoint
    ckpt_path = Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")
    print(ckpt_path)

    # Because the conversion script used `torch.save(model, out)`, 
    # the file contains the fully initialized and mapped PyTorch model object!
    lewm_model = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract just the encoder for Behavior Cloning
    lewm_encoder = lewm_model.encoder.to(device)
    lewm_encoder.eval()

    # Freeze the encoder so we only train the BC policy
    for param in lewm_encoder.parameters():
        param.requires_grad = False

    print("Successfully loaded and frozen the LeWM Encoder from the official checkpoint!")

    print("Successfully loaded and frozen the LeWM Encoder via the official API!")
    # 3. Initialize Latent BC Policy
    latent_dim = 192
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
        
        for batch_obs_seq, batch_actions in dataloader:
            batch_obs_seq = batch_obs_seq.to(device) # Shape: (Batch, FrameStack, C, H, W)
            batch_actions = batch_actions.to(device)

            with torch.no_grad():
                # Reshape to treat frames as a larger batch: (Batch * FrameStack, C, H, W)
                b, f, c, h, w = batch_obs_seq.shape
                flat_obs = batch_obs_seq.reshape(b * f, c, h, w)
                
                # Extract the Hugging Face output object
                encoder_outputs = lewm_encoder(flat_obs) 
                
                # Extract the CLS token (the 0th token) from the last hidden state
                # last_hidden_state shape: (Batch * FrameStack, Sequence_Length, Hidden_Dim)
                flat_latents = encoder_outputs.last_hidden_state[:, 0, :]
                
                # Reshape back to (Batch, FrameStack, LatentDim)
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
    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = avg_loss
        wandb_run.summary["best_train_loss"] = best_loss
        wandb_run.summary["best_train_loss_epoch"] = best_epoch
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
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for training")
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

    args = parser.parse_args()
    
    # Execute the training loop
    train_latent_bc(args)
