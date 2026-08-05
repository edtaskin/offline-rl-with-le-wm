"""Train the projected -> raw-CLS de-projector on the distribution imagination produces.

Two losses, because the bridge has two jobs and only one of them has ground truth:

* **inversion** (supervised): on real frames the pair ``(projector(cls), cls)``
  is exact, so ``g`` is trained to invert the projector where truth exists.
* **cycle consistency**: imagined latents drift away from any real frame, so
  there is no true CLS to regress onto. Asking ``g`` to map a drifted latent
  onto the real future frame would train it to *undo world-model error* -- which
  it cannot do, and which would hide model error inside the bridge and corrupt
  the RQ2 optimism measurements. Instead ``g`` is pinned by
  ``projector(g(z)) ~= z``: whatever the world model imagined, the bridge must
  hand the policy a CLS latent that projects back to it.

Rollouts use ground-truth dataset actions, so the imagined latents come from the
same predictor loop ``LeWMDreamWorld`` runs. Training consumes zero environment
interaction.

Usage::

    python -m scripts.deprojector.train_deprojector_pusht --anchors 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.deprojector.common import (  # noqa: E402
    action_normalizer,
    collect_rollouts,
    open_expert_h5,
    repo_path,
    sample_anchors,
    split_episodes,
)
from src.representations.deprojector import Deprojector, save_deprojector  # noqa: E402
from src.representations.lewm import LeWMEncoder, load_lewm_world_model  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--wm-checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--wm-cache-dir", default="le-wm/models")
    parser.add_argument("--output", default="models/deprojector/pusht_lewm/deprojector.pt")
    # Rollout / sampling geometry. Defaults mirror the dream trainer's contract.
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20, help="predictor steps per rollout")
    parser.add_argument("--anchors", type=int, default=1500, help="training rollouts")
    parser.add_argument("--val-anchors", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="held-out episodes")
    parser.add_argument("--group-size", type=int, default=16, help="anchors imagined at once")
    parser.add_argument("--encode-batch", type=int, default=96)
    # Optimization.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cycle-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_split(h5, wm, encoder, episodes, count, args, device, action_mean, action_std, label):
    rng = np.random.default_rng(args.seed + (0 if label == "train" else 1))
    anchors, usable, total = sample_anchors(
        h5,
        episodes,
        context_steps=args.context_steps,
        frameskip=args.frameskip,
        horizon=args.horizon,
        count=count,
        rng=rng,
    )
    print(
        f"{label}: {count} anchors from {usable}/{total} episodes long enough for "
        f"horizon {args.horizon} ({args.horizon * args.frameskip} env steps)",
        flush=True,
    )
    cls, imagined = collect_rollouts(
        h5,
        wm,
        encoder,
        anchors,
        context_steps=args.context_steps,
        frameskip=args.frameskip,
        horizon=args.horizon,
        history_size=int(getattr(wm.predictor, "num_frames", 3) or 3),
        group_size=args.group_size,
        encode_batch=args.encode_batch,
        device=device,
        action_mean=action_mean,
        action_std=action_std,
        progress_every=10,
    )
    return cls, imagined


@torch.no_grad()
def evaluate(model, projector, cls, imagined, device, batch_size=1024):
    """Inversion R^2 on real pairs, cycle error on imagined latents."""
    model.eval()
    flat_cls = cls.reshape(-1, cls.shape[-1])
    sse = ssz = 0.0
    for start in range(0, len(flat_cls), batch_size):
        target = flat_cls[start : start + batch_size].to(device)
        pred = model(projector(target))
        sse += ((pred - target) ** 2).sum().item()
        ssz += ((target - flat_cls.mean(0).to(device)) ** 2).sum().item()
    inversion_r2 = 1.0 - sse / max(ssz, 1e-12)

    flat_img = imagined.reshape(-1, imagined.shape[-1])
    cyc = ref = 0.0
    for start in range(0, len(flat_img), batch_size):
        z = flat_img[start : start + batch_size].to(device)
        cyc += ((projector(model(z)) - z) ** 2).sum().item()
        ref += (z**2).sum().item()
    return {
        "inversion_r2": inversion_r2,
        "inversion_mse": sse / flat_cls.numel(),
        "cycle_rel_l2": float(np.sqrt(cyc / max(ref, 1e-12))),
    }


@torch.no_grad()
def horizon_diagnostic(model, cls, imagined, context_steps, device, batch_size=1024):
    """Distance from ``g(imagined_t)`` to the real CLS at each horizon.

    Diagnostic only: it mixes bridge error with world-model drift, which is
    exactly why it is not the training target. `bridge_horizon.py` separates them.
    """
    model.eval()
    out = []
    for t in range(imagined.shape[1]):
        z = imagined[:, t]
        target = cls[:, context_steps + t]
        num = den = 0.0
        for start in range(0, len(z), batch_size):
            pred = model(z[start : start + batch_size].to(device))
            tgt = target[start : start + batch_size].to(device)
            num += ((pred - tgt) ** 2).sum().item()
            den += (tgt**2).sum().item()
        out.append(float(np.sqrt(num / max(den, 1e-12))))
    return out


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wm = load_lewm_world_model(args.wm_checkpoint, cache_dir=repo_path(args.wm_cache_dir), device=device)
    projector = wm.projector.eval().requires_grad_(False)
    encoder = LeWMEncoder(wm.encoder, device=device)
    history_size = int(getattr(wm.predictor, "num_frames", 3) or 3)

    h5 = open_expert_h5(args.dataset_path)
    try:
        action_mean, action_std, absolute = action_normalizer(h5, device)
        print(
            f"dataset actions {'absolute' if absolute else 'relative'} | "
            f"mean {action_mean.cpu().numpy().round(3).tolist()} | history {history_size}",
            flush=True,
        )
        train_eps, val_eps = split_episodes(len(h5["ep_len"]), args.val_fraction, args.seed)
        train_cls, train_img = build_split(
            h5, wm, encoder, train_eps, args.anchors, args, device, action_mean, action_std, "train"
        )
        val_cls, val_img = build_split(
            h5, wm, encoder, val_eps, args.val_anchors, args, device, action_mean, action_std, "val"
        )
    finally:
        h5.close()

    # Supervised pairs come from every real frame in every rollout window; the
    # cycle term uses the imagined latents from the same rollouts.
    sup = train_cls.reshape(-1, train_cls.shape[-1]).to(device)
    dream = train_img.reshape(-1, train_img.shape[-1]).to(device)
    print(f"training on {len(sup)} real pairs + {len(dream)} imagined latents", flush=True)

    model = Deprojector(train_cls.shape[-1], args.hidden_dim, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        sup_order = torch.randperm(len(sup), generator=generator, device=device)
        n_batches = max(1, len(sup) // args.batch_size)
        run_sup = run_cyc = 0.0
        for b in range(n_batches):
            s_idx = sup_order[b * args.batch_size : (b + 1) * args.batch_size]
            target = sup[s_idx]
            loss_sup = nn.functional.mse_loss(model(projector(target)), target)

            loss_cyc = torch.zeros((), device=device)
            if args.cycle_weight > 0:
                d_idx = torch.randint(
                    len(dream), (min(args.batch_size, len(dream)),), generator=generator, device=device
                )
                z = dream[d_idx]
                loss_cyc = nn.functional.mse_loss(projector(model(z)), z)

            loss = loss_sup + args.cycle_weight * loss_cyc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_sup += loss_sup.item()
            run_cyc += float(loss_cyc.detach())

        if epoch % 5 == 0 or epoch == args.epochs:
            stats = evaluate(model, projector, val_cls, val_img, device)
            history.append({"epoch": epoch, **stats})
            print(
                f"epoch {epoch:3d} | train sup {run_sup / n_batches:.5f} cyc {run_cyc / n_batches:.5f} "
                f"| val inversion R2 {stats['inversion_r2']:.4f} | val cycle rel-L2 {stats['cycle_rel_l2']:.4f}",
                flush=True,
            )
            score = stats["inversion_r2"] - stats["cycle_rel_l2"]
            if best is None or score > best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()}, stats)

    model.load_state_dict(best[1])
    final = best[2]
    per_horizon = horizon_diagnostic(model, val_cls, val_img, args.context_steps, device)

    output = repo_path(args.output)
    save_deprojector(
        model,
        output,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(args.dataset_path),
            "wm_checkpoint": args.wm_checkpoint,
            "frameskip": args.frameskip,
            "context_steps": args.context_steps,
            "horizon": args.horizon,
            "history_size": history_size,
            "train_anchors": args.anchors,
            "val_anchors": args.val_anchors,
            "cycle_weight": args.cycle_weight,
            "epochs": args.epochs,
            "seed": args.seed,
            "val": final,
            "val_rel_l2_to_real_cls_by_horizon": per_horizon,
            "env_steps_consumed": 0,
        },
    )
    metrics_path = output.with_name(output.stem + "_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {"args": vars(args), "history": history, "val": final,
             "val_rel_l2_to_real_cls_by_horizon": per_horizon},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved de-projector to {output}")
    print(f"saved metrics to {metrics_path}")
    print(
        f"val inversion R2 {final['inversion_r2']:.4f} | cycle rel-L2 {final['cycle_rel_l2']:.4f} | "
        f"rel-L2 to real CLS: step1 {per_horizon[0]:.4f} -> step{len(per_horizon)} {per_horizon[-1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
