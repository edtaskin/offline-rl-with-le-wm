"""Train PushT probes on noisy-action simulator states encoded by LeWM.

If the noisy-action dataset is missing, this script first calls
``scripts/data/noisy_actions_dataset.py`` with matching dataset-generation
arguments. It then encodes the rendered WM-step frames with LeWM and trains the
same linear/MLP probe formats used by ``scripts/probes/train_state.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import stable_worldmodel as swm
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.train_state import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    SplitConfig,
    encode_rows,
    load_latent_cache,
    repo_path,
    save_latent_cache,
    selected_probes,
    train_all_probes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="models/probes/pusht_noisy_actions/noisy_action_dataset.h5")
    parser.add_argument("--expert-dataset", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--output-dir", default="models/probes/pusht_noisy_actions/probes")
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--imagined-latent-cache", default=None)
    parser.add_argument("--cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--force-imagined-recache", action="store_true")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument(
        "--training-sources",
        choices=["direct", "imagined", "direct_and_imagined"],
        default="direct_and_imagined",
        help="direct uses encoded simulator frames; imagined uses LeWM rollouts; direct_and_imagined mixes both.",
    )
    parser.add_argument(
        "--imagined-fraction",
        type=float,
        default=0.5,
        help="Fraction of samples drawn from imagined rollout latents when --training-sources=direct_and_imagined.",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["agent_pos", "block_pos", "block_angle", "block_rel_objective", "block_rel_agent"],
        choices=[
            "all",
            "agent_pos",
            "block_pos",
            "block_angle",
            "block_rel_objective",
            "block_rel_agent",
            "objective_met",
        ],
    )

    # Dataset generation defaults: partially off-policy expert+noise tubes.
    parser.add_argument("--num-trajectories", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--noise-stds", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--action-noise-mode", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable expert-dataset action z-score normalization before feeding noisy actions to LeWM rollouts.",
    )

    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    return parser.parse_args()


def generate_dataset_if_needed(args: argparse.Namespace, dataset_path: Path) -> None:
    if dataset_path.exists() and not args.force_dataset:
        print(f"using noisy-action dataset: {dataset_path}")
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/data/noisy_actions_dataset.py"),
        "--expert-dataset",
        str(repo_path(args.expert_dataset)),
        "--output",
        str(dataset_path),
        "--num-trajectories",
        str(args.num_trajectories),
        "--horizon",
        str(args.horizon),
        "--context-steps",
        str(args.context_steps),
        "--frameskip",
        str(args.frameskip),
        "--noise-stds",
        args.noise_stds,
        "--action-noise-mode",
        args.action_noise_mode,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.generation_batch_size),
        "--resolution",
        str(args.resolution),
    ]
    if args.force_dataset:
        cmd.append("--force")
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def rows_by_split(h5_path: Path, max_samples: int | None, seed: int) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        splits = {split: h5[f"{split}_idx"][:].astype(np.int64) for split in ("train", "val", "test")}
    if max_samples is None:
        return splits

    rng = np.random.default_rng(seed + 1234)
    total = sum(len(v) for v in splits.values())
    if max_samples >= total:
        return splits
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    sampled = {}
    remaining = int(max_samples)
    for split in ("train", "val", "test"):
        if split == "test":
            n = remaining
        else:
            n = min(len(splits[split]), int(round(max_samples * fractions[split])))
            remaining -= n
        idx = splits[split]
        sampled[split] = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False).astype(np.int64))
    return sampled


def action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    action = h5["action"][:].astype(np.float32)
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def preprocess_context_pixels(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
    x = x.permute(0, 1, 4, 2, 3).div_(255.0)
    return (x - IMAGENET_MEAN.view(1, 1, 3, 1, 1).to(device)) / IMAGENET_STD.view(1, 1, 3, 1, 1).to(device)


@torch.inference_mode()
def rollout_embeddings(
    model: torch.nn.Module,
    context_pixels: np.ndarray,
    action_blocks: np.ndarray,
    horizon: int,
    history_size: int,
    device: torch.device,
) -> np.ndarray:
    context = preprocess_context_pixels(context_pixels, device)
    actions = torch.from_numpy(action_blocks).to(device=device, dtype=torch.float32)
    init = model.encode({"pixels": context})
    emb_list = list(init["emb"].unbind(dim=1))
    all_act_emb = model.action_encoder(actions)
    context_steps = context.shape[1]

    for step in range(horizon):
        end = context_steps + step
        lo = max(0, end - history_size)
        emb_trunc = torch.stack(emb_list[lo:end], dim=1)
        act_trunc = all_act_emb[:, lo:end]
        pred = model.predict(emb_trunc, act_trunc)[:, -1]
        emb_list.append(pred)

    return torch.stack(emb_list[context_steps:], dim=1).detach().cpu().float().numpy()


def trajectory_start_rows(h5: h5py.File) -> np.ndarray:
    starts = np.flatnonzero(h5["model_step"][:] == 1).astype(np.int64)
    if len(starts) == 0:
        raise ValueError("No trajectory starts found in noisy-action dataset.")
    return starts


def load_imagined_rollout_batch(
    noisy_h5: h5py.File,
    expert_h5: h5py.File,
    trajectory_rows: np.ndarray,
    *,
    context_steps: int,
    horizon: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalize_actions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    bsz = len(trajectory_rows)
    action_steps = context_steps + horizon - 1
    context_pixels = np.empty((bsz, context_steps, 224, 224, 3), dtype=np.uint8)
    raw_blocks = np.empty((bsz, action_steps, frameskip, 2), dtype=np.float32)

    for i, row0 in enumerate(trajectory_rows):
        row0 = int(row0)
        source_start = int(noisy_h5["source_start"][row0])
        context_idx = source_start + np.arange(context_steps) * frameskip
        context_pixels[i] = expert_h5["pixels"][context_idx]

        for t in range(context_steps - 1):
            a0 = source_start + t * frameskip
            raw_blocks[i, t] = expert_h5["action"][a0 : a0 + frameskip].astype(np.float32)
        for step in range(horizon):
            raw_blocks[i, context_steps - 1 + step] = noisy_h5["action"][row0 + step].astype(np.float32)

    blocks = raw_blocks
    if normalize_actions:
        blocks = (blocks - action_mean.reshape(1, 1, 1, 2)) / action_std.reshape(1, 1, 1, 2)
    return context_pixels, blocks.reshape(bsz, action_steps, frameskip * 2).astype(np.float32)


def imagined_metadata(args: argparse.Namespace, dataset_path: Path) -> dict:
    return {
        "dataset": str(dataset_path),
        "expert_dataset": str(repo_path(args.expert_dataset)),
        "checkpoint": args.checkpoint,
        "context_steps": int(args.context_steps),
        "horizon": int(args.horizon),
        "frameskip": int(args.frameskip),
        "normalize_actions": not args.no_normalize_actions,
    }


def load_imagined_cache(path: Path) -> tuple[np.ndarray, dict]:
    raw = np.load(path, allow_pickle=False)
    return raw["z"].astype(np.float32), json.loads(str(raw["metadata"]))


def save_imagined_cache(path: Path, z: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, z=z.astype(np.float32), metadata=np.array(json.dumps(metadata)))


def get_or_build_imagined_latents(
    args: argparse.Namespace,
    dataset_path: Path,
    cache_path: Path,
    model: torch.nn.Module,
) -> tuple[np.ndarray, dict]:
    expected = imagined_metadata(args, dataset_path)
    if cache_path.exists() and not args.force_imagined_recache:
        z, metadata = load_imagined_cache(cache_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(f"loaded imagined latent cache: {cache_path}")
            return z, metadata
        print(f"imagined latent cache metadata mismatch, recaching: {cache_path}")

    device = torch.device(args.device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))

    with h5py.File(dataset_path, "r") as noisy_h5, h5py.File(repo_path(args.expert_dataset), "r") as expert_h5:
        trajectory_rows = trajectory_start_rows(noisy_h5)
        total_rows = int(noisy_h5["state"].shape[0])
        z_all: np.ndarray | None = None
        action_mean, action_std = action_stats(expert_h5)

        for start in range(0, len(trajectory_rows), args.generation_batch_size):
            batch_rows = trajectory_rows[start : start + args.generation_batch_size]
            context_pixels, action_blocks = load_imagined_rollout_batch(
                noisy_h5,
                expert_h5,
                batch_rows,
                context_steps=args.context_steps,
                horizon=args.horizon,
                frameskip=args.frameskip,
                action_mean=action_mean,
                action_std=action_std,
                normalize_actions=not args.no_normalize_actions,
            )
            emb = rollout_embeddings(model, context_pixels, action_blocks, args.horizon, history_size, device)
            if z_all is None:
                z_all = np.empty((total_rows, emb.shape[-1]), dtype=np.float32)
            for i, row0 in enumerate(batch_rows):
                row0 = int(row0)
                z_all[row0 : row0 + args.horizon] = emb[i]
            print(f"imagined noisy-action latents: {min(start + len(batch_rows), len(trajectory_rows))}/{len(trajectory_rows)} trajectories")

    if z_all is None:
        raise RuntimeError("No imagined latents were generated.")
    metadata = expected
    save_imagined_cache(cache_path, z_all, metadata)
    print(f"saved imagined latent cache: {cache_path}")
    return z_all, metadata


def select_imagined_indices(count: int, take: int, seed: int) -> np.ndarray:
    if take <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=min(take, count), replace=False).astype(np.int64))


def split_seed_offset(split: str) -> int:
    return {"train": 101, "val": 202, "test": 303}.get(split, 404)


def mix_direct_and_imagined(
    direct_data: dict[str, dict[str, np.ndarray]],
    imagined_z_all: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray]]:
    if args.training_sources == "direct":
        return direct_data
    if imagined_z_all is None:
        raise ValueError(f"--training-sources={args.training_sources} requires imagined latents.")
    if not 0.0 <= args.imagined_fraction <= 1.0:
        raise ValueError("--imagined-fraction must be between 0 and 1.")

    mixed: dict[str, dict[str, np.ndarray]] = {}
    for split, split_data in direct_data.items():
        rows = split_data["rows"].astype(np.int64)
        direct_z = split_data["z"].astype(np.float32)
        states = split_data["state"].astype(np.float32)
        if args.training_sources == "imagined":
            mixed[split] = {"z": imagined_z_all[rows], "state": states, "rows": rows}
            continue

        n = len(rows)
        n_imagined = int(np.floor(n * args.imagined_fraction + 0.5))
        imagined_idx = select_imagined_indices(n, n_imagined, args.seed + split_seed_offset(split))
        direct_mask = np.ones(n, dtype=bool)
        direct_mask[imagined_idx] = False
        z = np.concatenate([direct_z[direct_mask], imagined_z_all[rows[imagined_idx]]], axis=0)
        state = np.concatenate([states[direct_mask], states[imagined_idx]], axis=0)
        mixed_rows = np.concatenate([rows[direct_mask], rows[imagined_idx]], axis=0)
        mixed[split] = {"z": z, "state": state, "rows": mixed_rows}
        print(f"{split} source mix: {direct_mask.sum()} direct + {len(imagined_idx)} imagined")
    return mixed


def main() -> None:
    args = parse_args()
    dataset_path = repo_path(args.dataset)
    output_dir = repo_path(args.output_dir)
    latent_cache = repo_path(args.latent_cache) if args.latent_cache else output_dir / "latents.npz"
    imagined_latent_cache = (
        repo_path(args.imagined_latent_cache)
        if args.imagined_latent_cache
        else output_dir / "imagined_latents.npz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_dataset_if_needed(args, dataset_path)

    model: torch.nn.Module | None = None
    if latent_cache.exists() and not args.force_recache:
        data, metadata = load_latent_cache(latent_cache)
        print(f"loaded latent cache: {latent_cache}")
    else:
        rows = rows_by_split(dataset_path, args.max_samples, args.seed)
        model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=repo_path(args.cache_dir))
        data = encode_rows(
            model=model,
            h5_path=dataset_path,
            rows_by_split=rows,
            batch_size=args.encode_batch_size,
            device=torch.device(args.device),
        )
        metadata = {
            "source": "noisy_action_dataset",
            "dataset": str(dataset_path),
            "checkpoint": args.checkpoint,
            "cache_dir": str(repo_path(args.cache_dir)),
            "max_samples": args.max_samples,
            "split": asdict(SplitConfig(seed=args.seed)),
        }
        save_latent_cache(latent_cache, data, metadata)
        print(f"saved latent cache: {latent_cache}")

    imagined_metadata_info = None
    imagined_z_all = None
    if args.training_sources in ("imagined", "direct_and_imagined"):
        if model is None:
            model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=repo_path(args.cache_dir))
        imagined_z_all, imagined_metadata_info = get_or_build_imagined_latents(
            args,
            dataset_path,
            imagined_latent_cache,
            model,
        )

    train_data = mix_direct_and_imagined(data, imagined_z_all, args)
    results = train_all_probes(train_data, args)
    results["latent_cache_metadata"] = metadata
    results["imagined_latent_cache"] = str(imagined_latent_cache) if imagined_z_all is not None else None
    results["imagined_latent_cache_metadata"] = imagined_metadata_info
    results["noisy_action_dataset"] = str(dataset_path)
    results["training_sources"] = args.training_sources
    results["imagined_fraction"] = args.imagined_fraction
    results_path = output_dir / "metrics.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"saved noisy-action probe metrics: {results_path}")
    print(f"trained probes: {selected_probes(args)}")


if __name__ == "__main__":
    main()
