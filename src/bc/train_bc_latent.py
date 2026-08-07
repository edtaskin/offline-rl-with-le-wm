import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset

load_dotenv()

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.bc.dataset import (
    PushTImageDataset,
    PushTLatentDataset,
    resolve_pusht_observation_resolution,
)
from src.bc.latent_cache import (
    build_latent_cache,
    check_latent_cache,
    default_latent_cache_path,
    expected_latent_cache_metadata,
    num_samples_from_data,
)
from src.representations.lewm import (
    LEWM_DEFAULT_FEATURE_DIM,
    LEWM_IMAGE_NORMALIZATION,
    LeWMEncoder,
    default_lewm_checkpoint_path,
    lewm_preprocessing_metadata,
)
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.bc.tracking import (
    args_for_config,
    init_wandb,
    log_wandb_artifact,
    write_run_config,
)
from src.utils.hf_hub import push_files_to_hub


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


def split_episode_indices(dataset, val_fraction, split_seed):
    """Partition sample indices into train / validation pools by *episode*.

    Consecutive frames inside one PushT episode are near-duplicates, so a
    frame-level split would put almost-identical samples on both sides and
    report a "validation" loss that is really a training loss. Splitting whole
    episodes is the same rule the dream trainer applies to its evaluation
    anchors (``_split_anchors`` in ``src/ppo/train_lewm.py``).

    Returns ``(train_indices, val_indices, num_val_episodes)``.
    """
    indices = np.arange(len(dataset))
    episode_of = np.searchsorted(dataset.episode_ends, indices, side="right")
    episodes = np.unique(episode_of)
    n_val = int(round(len(episodes) * val_fraction))
    if n_val < 1 or n_val >= len(episodes):
        raise ValueError(
            f"val_fraction={val_fraction} holds out {n_val} of {len(episodes)} episodes; "
            "both pools must be non-empty (use --val_fraction 0 to disable validation)"
        )
    rng = np.random.default_rng(split_seed)
    val_episodes = rng.permutation(episodes)[:n_val]
    is_val = np.isin(episode_of, val_episodes)
    return indices[~is_val], indices[is_val], int(n_val)


def stacked_latents_from_batch(batch_inputs, extractor, use_latent_cache, latent_dim):
    """Policy input for one batch: cached latents pass through, images are encoded."""
    if use_latent_cache:
        return batch_inputs
    batch_size, frames, channels, height, width = batch_inputs.shape
    flat_images = batch_inputs.reshape(batch_size * frames, channels, height, width)
    flat_latents = extractor.encode(flat_images)
    return flat_latents.reshape(batch_size, frames, latent_dim)


@torch.no_grad()
def evaluate_validation_loss(
    policy, dataloader, criterion, device, extractor, use_latent_cache, latent_dim
):
    """Mean MSE over the held-out pool, in eval mode."""
    was_training = policy.training
    policy.eval()
    total = 0.0
    count = 0
    for batch_inputs, batch_actions in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_actions = batch_actions.to(device)
        stacked_latents = stacked_latents_from_batch(
            batch_inputs, extractor, use_latent_cache, latent_dim
        )
        predicted_action_chunks = policy(stacked_latents)
        # Weight by sample count rather than averaging per-batch means: the
        # validation loader keeps its short final batch (drop_last is off), and
        # an unweighted mean would overweight those few samples.
        total += criterion(predicted_action_chunks, batch_actions).item() * len(batch_actions)
        count += len(batch_actions)
    if was_training:
        policy.train()
    return total / count if count else float("nan")


def _prepare_dataset(args, device):
    observation_resolution = resolve_pusht_observation_resolution(
        args.data_path, getattr(args, "observation_resolution", None)
    )
    latent_dim = LEWM_DEFAULT_FEATURE_DIM
    checkpoint_path = default_lewm_checkpoint_path()
    use_latent_cache = not args.disable_latent_cache
    cache_path = args.latent_cache_path or default_latent_cache_path(args.data_path)
    cache_rebuilt = False
    expected_metadata = None
    extractor = None

    if use_latent_cache:
        expected_metadata = expected_latent_cache_metadata(
            args.data_path,
            checkpoint_path,
            num_samples=num_samples_from_data(args.data_path),
            latent_dim=latent_dim,
        )
        cache_ok, cache_messages = check_latent_cache(cache_path, expected_metadata)
        if args.rebuild_latent_cache:
            cache_ok = False
            cache_messages = ["cache rebuild was requested"]
        if cache_ok:
            print(f"Using existing LeWM latent cache: {cache_path}")
        else:
            print(f"LeWM latent cache will be built at: {cache_path}")
            for message in cache_messages[:5]:
                print(f"  cache miss: {message}")
            source_dataset = PushTImageDataset(
                args.data_path,
                frame_stack=1,
                frame_stride=1,
                action_chunk_size=1,
            )
            extractor = LeWMEncoder.load(
                device=device,
                checkpoint_path=checkpoint_path,
                feature_dim=latent_dim,
            )
            build_latent_cache(
                source_dataset,
                extractor,
                cache_path,
                expected_metadata,
                batch_size=args.latent_cache_batch_size or args.batch_size,
            )
            cache_rebuilt = True
        dataset = PushTLatentDataset(
            args.data_path,
            cache_path,
            frame_stack=args.frame_stack,
            frame_stride=args.frame_stride,
            action_chunk_size=args.action_chunk_size,
        )
    else:
        print("Latent cache disabled; images will be encoded through LeWM during every epoch.")
        dataset = PushTImageDataset(
            args.data_path,
            frame_stack=args.frame_stack,
            frame_stride=args.frame_stride,
            action_chunk_size=args.action_chunk_size,
        )
        extractor = LeWMEncoder.load(
            device=device,
            checkpoint_path=checkpoint_path,
            feature_dim=latent_dim,
        )

    latent_dim = int(dataset.stats.get("latent_dim", latent_dim))
    dataset_stats = {
        **dataset.stats,
        **lewm_preprocessing_metadata(LEWM_IMAGE_NORMALIZATION),
        "source_image_shape": [observation_resolution, observation_resolution],
        "observation_resolution": observation_resolution,
    }
    return {
        "dataset": dataset,
        "dataset_stats": dataset_stats,
        "extractor": extractor,
        "latent_dim": latent_dim,
        "use_latent_cache": use_latent_cache,
        "cache_path": cache_path,
        "cache_rebuilt": cache_rebuilt,
        "expected_metadata": expected_metadata,
    }


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

    prepared = _prepare_dataset(args, device)
    dataset = prepared["dataset"]
    dataset_stats = prepared["dataset_stats"]
    extractor = prepared["extractor"]
    latent_dim = prepared["latent_dim"]
    use_latent_cache = prepared["use_latent_cache"]

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)
    if args.val_fraction > 0:
        train_indices, val_indices, num_val_episodes = split_episode_indices(
            dataset, args.val_fraction, args.val_split_seed
        )
        train_dataset = Subset(dataset, train_indices.tolist())
        val_dataset = Subset(dataset, val_indices.tolist())
        print(
            f"Episode-level split: {len(train_indices)} train / {len(val_indices)} val samples "
            f"({num_val_episodes} held-out episodes, split seed {args.val_split_seed})"
        )
    else:
        # Reproduces pre-split runs exactly: every sample stays in training.
        train_dataset, val_dataset, num_val_episodes = dataset, None, 0
        print("Validation disabled (--val_fraction 0); training on every sample.")

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=dataloader_generator,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_dataloader_worker,
        )
    run_metadata = {
        **dataset_stats,
        "frame_stack": args.frame_stack,
        "frame_stride": args.frame_stride,
        "hidden_dim": args.hidden_dim,
        "latent_dim": latent_dim,
        "action_dim": 2,
        "action_chunk_size": args.action_chunk_size,
        "latent_cache_enabled": use_latent_cache,
        "latent_cache_path": prepared["cache_path"] if use_latent_cache else None,
        "latent_cache_rebuilt": prepared["cache_rebuilt"],
        "latent_cache_expected_metadata": prepared["expected_metadata"],
        "seed": args.seed,
        "deterministic": args.deterministic,
        "num_workers": args.num_workers,
        # Carried in the stats sidecar so a checkpoint records which samples it
        # was allowed to see: two runs differing only in the split are not
        # comparable, and without this the difference is invisible after the
        # fact. _stats_contract() reads only the agent-contract keys, so these
        # are inert on the PPO side.
        "val_fraction": args.val_fraction,
        "val_split_seed": args.val_split_seed,
        "num_val_episodes": num_val_episodes,
    }
    run_config_path = write_run_config(
        args,
        dataset_stats=dataset_stats,
        extra={
            "device": str(device),
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "num_batches_per_epoch": len(dataloader),
            "status": "running",
            "bc_contract": run_metadata,
        },
    )
    wandb_run = init_wandb(
        args,
        {
            **args_for_config(args),
            "device": str(device),
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset) if val_dataset is not None else 0,
            "num_batches_per_epoch": len(dataloader),
            "bc_contract": run_metadata,
        },
    )
    if wandb_run is not None:
        print(f"Logging run to wandb: {getattr(wandb_run, 'url', None) or 'enabled'}")

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
    avg_loss = float("nan")
    val_loss = float("nan")
    best_val_loss = float("inf")
    best_val_epoch = None
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_actions in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_actions = batch_actions.to(device)
            stacked_latents = stacked_latents_from_batch(
                batch_inputs, extractor, use_latent_cache, latent_dim
            )
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

        # The first and last epochs always get a measurement so the curve has
        # both endpoints, whatever --val_interval is set to.
        if val_dataloader is not None and (
            (epoch + 1) % args.val_interval == 0 or epoch == 0 or epoch + 1 == args.epochs
        ):
            val_loss = evaluate_validation_loss(
                policy, val_dataloader, criterion, device, extractor, use_latent_cache, latent_dim
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1

        if wandb_run is not None:
            metrics = {
                "train/loss": avg_loss,
                "train/best_loss": best_loss,
                "train/epoch": epoch + 1,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if val_dataloader is not None and np.isfinite(val_loss):
                metrics["val/loss"] = val_loss
                metrics["val/best_loss"] = best_val_loss
                # The gap is the quantity that says whether a checkpoint is
                # underfit or overfit, which is what the epoch number alone
                # cannot tell you when comparing BC checkpoints downstream.
                metrics["val/train_gap"] = val_loss - avg_loss
            wandb_run.log(metrics, step=epoch + 1)
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            message = f"Epoch [{epoch + 1}/{args.epochs}] - Average MSE Loss: {avg_loss:.6f}"
            if val_dataloader is not None and np.isfinite(val_loss):
                message += f" | Val MSE: {val_loss:.6f}"
            print(message)
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_name = args.checkpoint_path.replace(".pth", f"_epoch{epoch + 1}.pth")
            torch.save(policy.state_dict(), checkpoint_name)

    torch.save(policy.state_dict(), args.checkpoint_path)
    stats_path = args.checkpoint_path.replace(".pth", "_stats.pth")
    torch.save(run_metadata, stats_path)
    completion_metadata = {
        "device": str(device),
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset) if val_dataset is not None else 0,
        "num_batches_per_epoch": len(dataloader),
        "status": "completed",
        "bc_contract": run_metadata,
        "final_checkpoint_path": args.checkpoint_path,
        "stats_path": stats_path,
        "run_config_path": run_config_path,
        "final_train_loss": avg_loss,
        "best_train_loss": best_loss,
        "best_train_loss_epoch": best_epoch,
        "final_val_loss": val_loss if val_dataloader is not None else None,
        "best_val_loss": best_val_loss if best_val_epoch is not None else None,
        "best_val_loss_epoch": best_val_epoch,
        "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
        "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
    }
    run_config_path = write_run_config(
        args,
        dataset_stats=dataset_stats,
        extra=completion_metadata,
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
        completion_metadata["run_config_path"] = run_config_path
        completion_metadata["hf_upload"] = {
            "repo_id": hf_upload_result.repo_id,
            "repo_type": hf_upload_result.repo_type,
            "repo_url": hf_upload_result.repo_url,
            "uploaded_files": hf_upload_result.uploaded_files,
        }
        run_config_path = write_run_config(
            args,
            dataset_stats=dataset_stats,
            extra=completion_metadata,
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
        if best_val_epoch is not None:
            wandb_run.summary["final_val_loss"] = val_loss
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["best_val_loss_epoch"] = best_val_epoch
        if hf_upload_result is not None:
            wandb_run.summary["hf/repo_url"] = hf_upload_result.repo_url
        log_wandb_artifact(
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
                "best_val_loss": best_val_loss if best_val_epoch is not None else None,
                "best_val_loss_epoch": best_val_epoch,
                "bc_contract": run_metadata,
                "hf_repo_url": hf_upload_result.repo_url if hf_upload_result is not None else None,
            },
        )
        wandb_run.finish()
    print(f"Latent BC Training Complete! Saved to: {args.checkpoint_path}")
    if best_val_epoch is not None:
        print(
            f"Validation: final {val_loss:.6f} | best {best_val_loss:.6f} @ epoch {best_val_epoch}"
            f" of {args.epochs}"
        )
    print(f"Run config saved to: {run_config_path}")
    return {
        "checkpoint_path": args.checkpoint_path,
        "stats_path": stats_path,
        "final_train_loss": avg_loss,
        "best_train_loss": best_loss,
        "final_val_loss": val_loss if val_dataloader is not None else None,
        "best_val_loss": best_val_loss if best_val_epoch is not None else None,
        "best_val_loss_epoch": best_val_epoch,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Latent-Space Behavior Cloning (BC) Prior for LeWorldModel")
    parser.add_argument("--data_path", type=str, default="data/expert_trajectories/pusht_expert_224.npz", help="Path to the high-resolution expert dataset generated by scripts/regenerate_pusht_expert.py")
    parser.add_argument("--observation-resolution", "--observation_resolution", dest="observation_resolution", type=int, default=None, help="Native expert-image resolution. When omitted it is inferred from the dataset; an explicit mismatch is rejected")
    parser.add_argument("--checkpoint_path", type=str, default="runs/bc/pusht_latent_bc.pth", help="Temporary local output path used before optional Hugging Face upload")
    parser.add_argument("--latent_cache_path", type=str, default=None, help="Path for cached per-frame LeWM CLS latents; defaults to data/latent_cache/<dataset>_lewm_object_imagenet224_cls.pt")
    parser.add_argument("--rebuild_latent_cache", action="store_true", help="Recompute and overwrite the LeWM latent cache before training")
    parser.add_argument("--disable_latent_cache", action="store_true", help="Disable latent caching and encode image batches through LeWM during every epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for training")
    parser.add_argument("--latent_cache_batch_size", type=int, default=None, help="Batch size for one-time cache construction; defaults to --batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Enable deterministic PyTorch/CuDNN behavior where available")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the BC MLP policy")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of LeWM latents to stack for temporal context")
    parser.add_argument("--frame_stride", type=int, default=5, help="Environment steps between stacked history frames")
    parser.add_argument("--action_chunk_size", type=int, default=5, help="Number of future actions to predict from one observation")
    parser.add_argument("--log_interval", type=int, default=10, help="Epochs to wait before logging loss metrics")
    parser.add_argument("--save_interval", type=int, default=10, help="Save policy checkpoint every N epochs")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of expert EPISODES held out for validation; 0 disables validation and trains on every sample (the pre-split behavior)")
    parser.add_argument("--val_split_seed", type=int, default=0, help="Seed for the episode-level train/validation split; independent of --seed so the split is stable across training seeds")
    parser.add_argument("--val_interval", type=int, default=1, help="Evaluate validation loss every N epochs; the first and last epoch are always measured")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="offline-rl-lewm", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default="pusht-latent-bc", help="Weights & Biases run group")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional Weights & Biases tags")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default=None, help="Weights & Biases mode; use offline on clusters without network access")
    parser.add_argument("--push_to_hf", action="store_true", help="Push final checkpoint artifacts to the Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face repo id, for example username/repo-name")
    parser.add_argument("--hf_repo_type", type=str, choices=["model", "dataset", "space"], default="model", help="Hugging Face repository type")
    parser.add_argument("--hf_private", action=argparse.BooleanOptionalAction, default=False, help="Create or keep the Hugging Face repo private")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token; if omitted, huggingface_hub uses the cached login or environment token")
    parser.add_argument("--hf_revision", type=str, default=None, help="Optional Hugging Face branch or revision to upload to")
    parser.add_argument("--hf_path_prefix", type=str, default=None, help="Optional folder inside the Hugging Face repo")
    parser.add_argument("--hf_commit_message", type=str, default="Upload latent BC checkpoint artifacts", help="Commit message for Hugging Face uploads")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.push_to_hf and not args.hf_repo_id:
        parser.error("--hf_repo_id is required when --push_to_hf is set")
    train_latent_bc(args)


if __name__ == "__main__":
    main()
