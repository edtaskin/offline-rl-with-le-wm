"""Train a visualization decoder from frozen PushT LeWM latents to RGB images.

This follows the decoder described in the LeWorldModel paper: a compact latent
is projected to a hidden dimension and used as memory for cross-attention from a
fixed set of learnable patch queries. Each query predicts one RGB image patch.

The decoder is diagnostic only. LeWM stays frozen and no reconstruction loss is
fed back into the world model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class SplitConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    seed: int = 3072


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--output-dir", default="models/latent_decoder/pusht_lewm")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--sample-block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument(
        "--foreground-weight",
        type=float,
        default=10.0,
        help="Extra reconstruction weight for non-white pixels to avoid the trivial white-background solution.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.05,
        help="Pixel is foreground when any RGB channel differs from white by more than this value.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cache-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute frozen LeWM latents for the sampled rows before decoder training.",
    )
    return parser.parse_args()


def split_episodes(num_episodes: int, cfg: SplitConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    episodes = rng.permutation(num_episodes)
    n_train = int(num_episodes * cfg.train_fraction)
    n_val = int(num_episodes * cfg.val_fraction)
    return {
        "train": np.sort(episodes[:n_train]),
        "val": np.sort(episodes[n_train : n_train + n_val]),
        "test": np.sort(episodes[n_train + n_val :]),
    }


def rows_for_episodes(
    offsets: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
    max_rows: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_rows < 0:
        rows = np.concatenate(
            [np.arange(offsets[ep], offsets[ep] + lengths[ep], dtype=np.int64) for ep in episodes]
        )
        return np.sort(rows)

    block_size = max(1, block_size)
    rows: list[int] = []
    seen: set[int] = set()
    episode_lengths = lengths[episodes].astype(np.float64)
    episode_probs = episode_lengths / episode_lengths.sum()
    max_attempts = max(10_000, 20 * int(np.ceil(max_rows / block_size)))

    attempts = 0
    while len(rows) < max_rows and attempts < max_attempts:
        attempts += 1
        ep = int(rng.choice(episodes, p=episode_probs))
        ep_len = int(lengths[ep])
        start = int(rng.integers(0, max(1, ep_len - block_size + 1)))
        global_start = int(offsets[ep] + start)
        global_end = min(global_start + block_size, int(offsets[ep] + ep_len))
        for row in range(global_start, global_end):
            if row not in seen:
                seen.add(row)
                rows.append(row)
                if len(rows) >= max_rows:
                    break

    return np.asarray(rows, dtype=np.int64)


def sample_rows(h5_path: Path, max_samples: int, block_size: int, split_cfg: SplitConfig) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        lengths = f["ep_len"][:]
        offsets = f["ep_offset"][:]

    splits = split_episodes(len(lengths), split_cfg)
    if max_samples < 0:
        per_split = {name: -1 for name in splits}
    else:
        per_split = {
            "train": int(max_samples * split_cfg.train_fraction),
            "val": int(max_samples * split_cfg.val_fraction),
            "test": max_samples
            - int(max_samples * split_cfg.train_fraction)
            - int(max_samples * split_cfg.val_fraction),
        }

    rng = np.random.default_rng(split_cfg.seed + 1)
    return {
        name: rows_for_episodes(offsets, lengths, episodes, per_split[name], block_size, rng)
        for name, episodes in splits.items()
    }


class H5PixelsDataset(Dataset):
    def __init__(self, h5_path: Path, rows: np.ndarray):
        self.h5_path = Path(h5_path)
        self.rows = rows.astype(np.int64)
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx: int) -> torch.Tensor:
        pixel = self._file()["pixels"][int(self.rows[idx])]
        return torch.from_numpy(pixel.copy())


class H5LatentPixelsDataset(H5PixelsDataset):
    def __init__(self, h5_path: Path, rows: np.ndarray, latents: torch.Tensor):
        super().__init__(h5_path, rows)
        if len(latents) != len(rows):
            raise ValueError("latents and rows must have the same length")
        self.latents = latents.float().cpu()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], super().__getitem__(idx)


def encoder_pixels(pixels_hwc_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = pixels_hwc_u8.to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2).div_(255.0)
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def target_pixels(pixels_hwc_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    return pixels_hwc_u8.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn, _ = self.cross_attn(self.query_norm(query), self.memory_norm(memory), memory)
        query = query + attn
        query = query + self.mlp(query)
        return query


class LatentImageDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 256,
        image_size: int = 224,
        patch_size: int = 16,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.latent_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim))
        self.query = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.to_patch = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        memory = self.latent_proj(latent).unsqueeze(1)
        query = self.query.expand(latent.size(0), -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        patches = self.to_patch(self.out_norm(query))
        image = rearrange(
            patches,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            gh=self.grid_size,
            gw=self.grid_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        return torch.sigmoid(image)


def encode_latent(lewm: nn.Module, pixels_hwc_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        pixels = encoder_pixels(pixels_hwc_u8, device)
        latent = lewm.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
    return latent.detach()


def unpack_batch(
    batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    lewm: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)):
        latent_cpu, pixels = batch
        latent = latent_cpu.to(device=device, dtype=torch.float32)
    else:
        pixels = batch
        latent = encode_latent(lewm, pixels, device)
    target = target_pixels(pixels, device)
    return latent, target, pixels


@torch.inference_mode()
def precompute_latents(
    lewm: nn.Module,
    h5_path: Path,
    rows: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        H5PixelsDataset(h5_path, rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    latents = []
    for pixels in loader:
        latents.append(encode_latent(lewm, pixels, device).cpu())
    return torch.cat(latents, dim=0)


def weighted_mean(loss_map: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (loss_map * weight).sum() / (weight.sum() * loss_map.size(1)).clamp_min(1.0)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float,
    foreground_weight: float,
    foreground_threshold: float,
) -> torch.Tensor:
    foreground = (target.sub(1.0).abs().amax(dim=1, keepdim=True) > foreground_threshold).float()
    weight = 1.0 + (foreground_weight - 1.0) * foreground
    l1 = weighted_mean((pred - target).abs(), weight)
    mse = weighted_mean((pred - target).pow(2), weight)
    return l1 + mse_weight * mse


@torch.inference_mode()
def save_reconstruction_grid(
    lewm: nn.Module,
    decoder: nn.Module,
    batch: torch.Tensor,
    path: Path,
    device: torch.device,
    max_images: int = 8,
) -> None:
    if isinstance(batch, (tuple, list)):
        batch = (batch[0][:max_images], batch[1][:max_images])
    else:
        batch = batch[:max_images]
    latent, target, _ = unpack_batch(batch, lewm, device)
    recon = decoder(latent).clamp(0, 1)
    rows = torch.stack([target.cpu(), recon.cpu()], dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(rows, path, nrow=2)


def run_epoch(
    lewm: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mse_weight: float,
    foreground_weight: float,
    foreground_threshold: float,
) -> float:
    training = optimizer is not None
    decoder.train(training)
    total = 0.0
    n = 0
    for batch in loader:
        with torch.no_grad():
            latent, target, _ = unpack_batch(batch, lewm, device)

        pred = decoder(latent)
        loss = reconstruction_loss(pred, target, mse_weight, foreground_weight, foreground_threshold)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = target.size(0)
        total += float(loss.detach().cpu()) * batch_size
        n += batch_size
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    h5_path = repo_path(args.dataset_path)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    split_cfg = SplitConfig(seed=args.seed)
    rows = sample_rows(h5_path, args.max_samples, args.sample_block_size, split_cfg)
    with (output_dir / "rows.json").open("w") as f:
        json.dump({k: v.tolist() for k, v in rows.items()}, f)

    checkpoint_cache_dir = repo_path(args.checkpoint_cache_dir)
    lewm = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=checkpoint_cache_dir)
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)

    if args.cache_latents:
        print("precomputing frozen LeWM latents")
        latent_by_split = {
            name: precompute_latents(lewm, h5_path, split_rows, args.batch_size, args.num_workers, device)
            for name, split_rows in rows.items()
        }
        dataset_by_split = {
            name: H5LatentPixelsDataset(h5_path, rows[name], latent_by_split[name]) for name in rows
        }
    else:
        dataset_by_split = {name: H5PixelsDataset(h5_path, split_rows) for name, split_rows in rows.items()}

    train_loader = DataLoader(
        dataset_by_split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_by_split["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset_by_split["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    decoder = LatentImageDecoder(
        latent_dim=192,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fixed_val = next(iter(val_loader))
    best_val = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            lewm,
            decoder,
            train_loader,
            optimizer,
            device,
            args.mse_weight,
            args.foreground_weight,
            args.foreground_threshold,
        )
        val_loss = run_epoch(
            lewm,
            decoder,
            val_loader,
            None,
            device,
            args.mse_weight,
            args.foreground_weight,
            args.foreground_threshold,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        save_reconstruction_grid(
            lewm,
            decoder,
            fixed_val,
            output_dir / "reconstructions" / f"epoch_{epoch:03d}.png",
            device,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            stale = 0
            torch.save(
                {
                    "decoder": best_state,
                    "config": {
                        "latent_dim": 192,
                        "hidden_dim": args.hidden_dim,
                        "image_size": args.image_size,
                        "patch_size": args.patch_size,
                        "depth": args.depth,
                        "heads": args.heads,
                        "mlp_ratio": args.mlp_ratio,
                    },
                    "args": vars(args),
                    "split": asdict(split_cfg),
                    "best_val_loss": best_val,
                },
                output_dir / "decoder_best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        decoder.load_state_dict(best_state)
    test_loss = run_epoch(
        lewm,
        decoder,
        test_loader,
        None,
        device,
        args.mse_weight,
        args.foreground_weight,
        args.foreground_threshold,
    )
    save_reconstruction_grid(lewm, decoder, fixed_val, output_dir / "reconstruction_best.png", device)

    metrics = {
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "history": history,
        "args": vars(args),
        "split": asdict(split_cfg),
        "num_samples": {k: int(len(v)) for k, v in rows.items()},
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"best_val_loss": best_val, "test_loss": test_loss}, indent=2))
    print(f"saved decoder to {output_dir / 'decoder_best.pt'}")


if __name__ == "__main__":
    main()
