"""Shared dataset sampling and imagined-rollout helpers for the de-projector scripts.

Everything here mirrors :class:`src.ppo.train_lewm.LeWMDreamWorld` -- same
context construction, same action normalization, same autoregressive predictor
loop -- so latents produced here are drawn from the distribution the dream
trainer actually rolls in. Rollouts use ground-truth dataset actions, which is
what makes each imagined latent pairable with the real frame it should
correspond to.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def open_expert_h5(path):
    """Open the LeWM expert h5.

    The pixel dataset is Blosc/Zstd-compressed, so the hdf5plugin filters must
    be registered before the first read or h5py raises a bare "can't open
    directory /usr/local/lib/plugin" OSError.
    """
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import h5py

    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expert dataset not found: {path}")
    return h5py.File(path, "r")


def action_normalizer(h5, device):
    """Dataset action mean/std, matching ``LeWMDreamWorld``'s z-scoring."""
    action = h5["action"][:]
    mean = action.mean(axis=0).astype(np.float32)
    std = action.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        torch.as_tensor(mean, device=device),
        torch.as_tensor(std, device=device),
        bool(np.abs(mean).max() > 10.0),
    )


def split_episodes(n_episodes: int, val_fraction: float, seed: int):
    """Hold out whole episodes, never frames.

    Anchors are dense within an episode, so a frame-level split would put
    near-duplicate context windows on both sides of the split.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def sample_anchors(h5, episodes, *, context_steps, frameskip, horizon, count, rng):
    """``[N, 2]`` (episode, anchor row) pairs with full context and full horizon.

    The anchor is the last context frame -- the imagined "now". It needs
    ``(context_steps - 1) * frameskip`` rows of history behind it and
    ``horizon * frameskip`` rows of ground truth ahead of it.
    """
    lengths = h5["ep_len"][:]
    offsets = h5["ep_offset"][:]
    lead = (context_steps - 1) * frameskip
    tail = horizon * frameskip

    usable = [ep for ep in episodes if int(lengths[ep]) > lead + tail]
    if not usable:
        raise ValueError(
            f"No episode is long enough for context {context_steps} and horizon "
            f"{horizon} at frameskip {frameskip} (needs {lead + tail + 1} frames)"
        )
    usable = np.asarray(usable)

    picks = rng.choice(usable, size=count, replace=len(usable) < count)
    anchors = np.empty((count, 2), dtype=np.int64)
    for i, ep in enumerate(picks):
        local = rng.integers(lead, int(lengths[ep]) - tail)
        anchors[i] = (ep, int(offsets[ep]) + int(local))
    return anchors, len(usable), len(episodes)


def read_anchor_window(h5, anchor: int, *, context_steps, frameskip, horizon):
    """Frames and action blocks for one anchor, as two contiguous h5 reads.

    Returns ``frames [context_steps + horizon, H, W, 3]`` (the context window
    followed by the ground-truth future, sampled every ``frameskip``) and
    ``blocks [(context_steps - 1) + horizon, frameskip, 2]`` aligned so that
    ``blocks[t]`` is the action block taken *at* ``frames[t]``.
    """
    start = int(anchor) - (context_steps - 1) * frameskip
    stop = int(anchor) + horizon * frameskip + 1
    frames = h5["pixels"][start:stop:frameskip]
    n_blocks = (context_steps - 1) + horizon
    actions = h5["action"][start : start + n_blocks * frameskip]
    return frames, actions.reshape(n_blocks, frameskip, 2)


@torch.no_grad()
def encode_cls(encoder, frames_uint8, batch_size, device) -> torch.Tensor:
    """``[N, H, W, 3]`` uint8 -> raw CLS ``[N, D]`` through the frozen encoder."""
    out = []
    for start in range(0, len(frames_uint8), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(frames_uint8[start : start + batch_size]))
        out.append(encoder(chunk.permute(0, 3, 1, 2).to(device)).cpu())
    return torch.cat(out)


def normalize_blocks(blocks: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """``[..., k, 2]`` dataset-space actions -> flattened z-scored ``[..., k*2]``."""
    return ((blocks - mean) / std).reshape(*blocks.shape[:-2], -1)


@torch.no_grad()
def imagine(wm, context_emb: torch.Tensor, blocks: torch.Tensor, *, horizon: int, history_size: int):
    """Autoregressive ground-truth-action rollout in projected latent space.

    ``context_emb``: ``[B, context_steps, D]`` projected context latents.
    ``blocks``: ``[B, (context_steps - 1) + horizon, k*2]`` normalized action
    blocks, aligned with the frames they were taken at.

    Returns ``[B, horizon, D]``: the imagined latent after each predictor step,
    where step ``t`` corresponds to the real frame ``anchor + (t + 1) * frameskip``.
    """
    context_steps = context_emb.shape[1]
    emb_hist = deque(
        [context_emb[:, i] for i in range(context_steps - history_size, context_steps)],
        maxlen=history_size,
    )
    act_hist = deque(
        [blocks[:, i] for i in range(context_steps - 1)],
        maxlen=history_size,
    )

    preds = []
    for step in range(horizon):
        act_hist.append(blocks[:, context_steps - 1 + step])
        emb = torch.stack(tuple(emb_hist), dim=1)
        act = torch.stack(tuple(act_hist), dim=1)
        pred = wm.predict(emb, wm.action_encoder(act))[:, -1]
        emb_hist.append(pred)
        preds.append(pred)
    return torch.stack(preds, dim=1)


def collect_rollouts(
    h5,
    wm,
    encoder,
    anchors,
    *,
    context_steps,
    frameskip,
    horizon,
    history_size,
    group_size,
    encode_batch,
    device,
    action_mean,
    action_std,
    progress_every=0,
):
    """Encode anchor windows and imagine forward from each.

    Returns ``(cls, imagined)`` where ``cls`` is ``[N, context_steps + horizon, D]``
    of real-frame CLS latents and ``imagined`` is ``[N, horizon, D]`` of predictor
    outputs, so ``imagined[:, t]`` pairs with ``cls[:, context_steps + t]``.
    """
    all_cls, all_pred = [], []
    for group_start in range(0, len(anchors), group_size):
        group = anchors[group_start : group_start + group_size]
        frames, blocks = [], []
        for _, anchor in group:
            f, b = read_anchor_window(
                h5, anchor, context_steps=context_steps, frameskip=frameskip, horizon=horizon
            )
            frames.append(f)
            blocks.append(b)
        frames = np.stack(frames)  # [G, context+horizon, H, W, 3]
        g, t = frames.shape[:2]
        cls = encode_cls(encoder, frames.reshape(g * t, *frames.shape[2:]), encode_batch, device)
        cls = cls.reshape(g, t, -1)

        blocks = torch.as_tensor(np.stack(blocks), dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = wm.projector(cls[:, :context_steps].to(device).reshape(-1, cls.shape[-1]))
            emb = emb.reshape(g, context_steps, -1)
            pred = imagine(
                wm,
                emb,
                normalize_blocks(blocks, action_mean, action_std),
                horizon=horizon,
                history_size=history_size,
            )
        all_cls.append(cls)
        all_pred.append(pred.cpu())
        if progress_every and (group_start // group_size) % progress_every == 0:
            print(f"  rollouts {group_start + len(group)}/{len(anchors)}", flush=True)
    return torch.cat(all_cls), torch.cat(all_pred)
