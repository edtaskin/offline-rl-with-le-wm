"""How deep into imagination can the success probe still be trusted?

The dream reward *and* the dream episode's termination both come from one frozen
``objective_met`` classifier reading a latent the simulator never corrects. If
its false-positive rate grows with depth, dream PPO is paid to walk into the
part of the horizon where the probe is wrong -- and the fix is to shorten
``dream_episode_steps``, not to change the bridge.

Rolls ground-truth-action trajectories so the true success label is known at
every step, and reads the probe two ways:

* ``encoded_gt``  probe on ``projector(cls)`` of the *real* frame -- the probe's
                  own error, with no world-model drift involved
* ``imagined``    probe on the imagined latent -- probe error plus drift

The gap between the two curves is what the world model contributes, which is the
distinction that decides whether a shorter horizon actually helps.

This is the subset of ``scripts/rq2/probe_horizon.py``'s measurement that needs
only the published classifier: that script additionally drives
``scripts/rollouts/state_probes.py``, which requires the unpublished
``block_rel_*`` regression probes and does not accept the
``--classifier-checkpoint`` flag it is passed.

Usage::

    python -m scripts.deprojector.probe_trust_horizon --anchors 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
from src.envs import PUSHT_FIXED_TARGET_POSE  # noqa: E402
from src.ppo.train_lewm import _StateProbe  # noqa: E402
from src.representations.lewm import LeWMEncoder, load_lewm_world_model  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--wm-checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--wm-cache-dir", default="le-wm/models")
    parser.add_argument("--probe-dir", default="models/probes/pusht_lewm")
    parser.add_argument("--output-dir", default="runs/deprojector")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--encode-batch", type=int, default=96)
    parser.add_argument("--success-pos-tol", type=float, default=20.0)
    parser.add_argument("--success-angle-tol", type=float, default=float(np.pi / 9))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def true_success(states, pos_tol, angle_tol):
    """The fixed-target success condition the probes were trained against."""
    target_xy = np.asarray(PUSHT_FIXED_TARGET_POSE)[:2]
    target_angle = float(np.asarray(PUSHT_FIXED_TARGET_POSE)[2])
    pos_err = np.linalg.norm(states[..., 2:4] - target_xy, axis=-1)
    delta = states[..., 4] - target_angle
    angle_err = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    return (pos_err <= pos_tol) & (angle_err <= angle_tol)


def rates(predicted, actual):
    pos, neg = actual.sum(), (~actual).sum()
    return {
        "positive_rate": float(actual.mean()),
        "predicted_rate": float(predicted.mean()),
        "false_positive_rate": float((predicted & ~actual).sum() / neg) if neg else float("nan"),
        "true_positive_rate": float((predicted & actual).sum() / pos) if pos else float("nan"),
        "precision": float((predicted & actual).sum() / predicted.sum()) if predicted.sum() else float("nan"),
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    wm = load_lewm_world_model(args.wm_checkpoint, cache_dir=repo_path(args.wm_cache_dir), device=device)
    encoder = LeWMEncoder(wm.encoder, device=device)
    probe = _StateProbe.find(
        repo_path(args.probe_dir),
        ("objective_met/mlp_probe.pt", "is_objective_met_probe_baseline.pt"),
        device,
    )
    if probe is None:
        raise FileNotFoundError(f"No objective_met classifier under {repo_path(args.probe_dir)}")
    print(f"probe threshold {probe.threshold:.4f}", flush=True)

    h5 = open_expert_h5(args.dataset_path)
    try:
        action_mean, action_std, _ = action_normalizer(h5, device)
        _, val_eps = split_episodes(len(h5["ep_len"]), args.val_fraction, args.seed)
        rng = np.random.default_rng(args.seed + 1)
        anchors, usable, total = sample_anchors(
            h5,
            val_eps,
            context_steps=args.context_steps,
            frameskip=args.frameskip,
            horizon=args.horizon,
            count=args.anchors,
            rng=rng,
        )
        print(f"{args.anchors} held-out anchors from {usable}/{total} episodes", flush=True)
        cls, imagined = collect_rollouts(
            h5, wm, encoder, anchors,
            context_steps=args.context_steps,
            frameskip=args.frameskip,
            horizon=args.horizon,
            history_size=int(getattr(wm.predictor, "num_frames", 3) or 3),
            group_size=args.group_size,
            encode_batch=args.encode_batch,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
            progress_every=8,
        )
        # Ground-truth state at each imagined step's real counterpart.
        states = np.stack(
            [
                h5["state"][
                    int(a) + args.frameskip : int(a) + (args.horizon + 1) * args.frameskip : args.frameskip
                ]
                for _, a in anchors
            ]
        )
    finally:
        h5.close()

    labels = true_success(states, args.success_pos_tol, args.success_angle_tol)  # [N, horizon]
    rows = []
    with torch.no_grad():
        for t in range(args.horizon):
            actual = labels[:, t]
            imagined_fire = (
                probe(imagined[:, t].to(device))[:, 0] >= probe.threshold
            ).cpu().numpy()
            encoded_fire = (
                probe(wm.projector(cls[:, args.context_steps + t].to(device)))[:, 0] >= probe.threshold
            ).cpu().numpy()
            rows.append(
                {
                    "step": t + 1,
                    "env_steps": (t + 1) * args.frameskip,
                    "imagined": rates(imagined_fire, actual),
                    "encoded_gt": rates(encoded_fire, actual),
                }
            )

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_trust_horizon.json").write_text(
        json.dumps({"args": vars(args), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'step':>4s} {'env':>4s} {'true+':>7s} | {'imagined FP':>11s} {'TP':>6s} {'prec':>6s}"
          f" | {'encGT FP':>9s} {'TP':>6s} {'prec':>6s}")
    for row in rows:
        im, gt = row["imagined"], row["encoded_gt"]
        print(
            f"{row['step']:4d} {row['env_steps']:4d} {im['positive_rate']:7.3f} | "
            f"{im['false_positive_rate']:11.3f} {im['true_positive_rate']:6.3f} {im['precision']:6.3f} | "
            f"{gt['false_positive_rate']:9.3f} {gt['true_positive_rate']:6.3f} {gt['precision']:6.3f}"
        )
    print(f"\nsaved {output_dir / 'probe_trust_horizon.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
