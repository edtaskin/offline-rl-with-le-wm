"""How much does each dream bridge cost the policy, as a function of horizon?

Rolls ground-truth-action trajectories through LeWM on held-out episodes, so
every imagined latent has the real frame it should correspond to, and scores
both ways of getting from an imagined latent back to the policy's input space:

* ``decoder``      imagined latent -> RGB frame -> ViT -> CLS (what
                   ``src/ppo/train_lewm.py`` does today)
* ``deprojector``  imagined latent -> CLS directly

Three quantities are reported per horizon, and keeping them apart is the point:

* ``wm_drift``    how far the imagined latent has drifted from the real frame's
                  projected latent -- the world model's error, which no bridge
                  can undo
* ``bridge_only`` the bridge applied to an *undrifted* latent
                  (``projector(cls_true)``) -- the bridge's own error floor
* ``end_to_end``  the bridge applied to the imagined latent, versus the real CLS

and finally what actually matters: the action difference between running the BC
policy on bridged latents and on the true CLS latents it was trained for.

Usage::

    python -m scripts.deprojector.bridge_horizon --anchors 300
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
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy  # noqa: E402
from src.bc.dataset import PUSHT_ACTION_SCALE  # noqa: E402
from src.representations.deprojector import load_deprojector  # noqa: E402
from src.representations.lewm import LeWMEncoder, load_lewm_world_model  # noqa: E402
from src.utils.hf_hub import resolve_artifact  # noqa: E402

try:  # branches without the projected-latent switch have no such constant
    from src.representations.lewm import LEWM_LATENT_RAW_CLS
except ImportError:
    LEWM_LATENT_RAW_CLS = "raw_cls"

DEFAULT_BC = "hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth"
DEFAULT_BC_STATS = "hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best_stats.pth"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--wm-checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument("--wm-cache-dir", default="le-wm/models")
    parser.add_argument("--deprojector", default="models/deprojector/pusht_lewm/deprojector.pt")
    parser.add_argument(
        "--decoder-checkpoint", default="models/latent_decoder/pusht_lewm/decoder_lewm_pusht.pt"
    )
    parser.add_argument("--skip-decoder", action="store_true", help="score only the de-projector")
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC)
    parser.add_argument("--bc-stats", default=DEFAULT_BC_STATS)
    parser.add_argument("--output-dir", default="runs/deprojector")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--anchors", type=int, default=300)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--encode-batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def load_raw_cls_bc_policy(checkpoint, stats_path, device):
    """The published raw-CLS BC policy, used only to score bridge error in action units."""
    stats = torch.load(resolve_artifact(stats_path), map_location=device, weights_only=False)
    representation = stats.get("latent_representation", LEWM_LATENT_RAW_CLS)
    if representation != LEWM_LATENT_RAW_CLS:
        raise ValueError(
            f"bridge scoring needs a raw-CLS BC policy, got {representation!r}; a projected "
            "policy consumes the dynamics latent directly and needs no bridge"
        )
    policy = LatentBCPolicy(
        latent_dim=int(stats.get("latent_dim", 192)),
        frame_stack=int(stats.get("frame_stack", 3)),
        action_dim=int(stats.get("action_dim", 2)),
        hidden_dim=int(stats.get("hidden_dim", 256)),
        action_chunk_size=int(stats.get("action_chunk_size", 5)),
    )
    policy.load_state_dict(torch.load(resolve_artifact(checkpoint), map_location=device))
    return policy.to(device).eval().requires_grad_(False), int(stats.get("frame_stack", 3))


def rel_l2(pred, target):
    return float(torch.sqrt(((pred - target) ** 2).sum() / (target**2).sum()))


def cosine(pred, target):
    return float(torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean())


def dilated_stacks(sequence, frame_stack):
    """``[B, T, D]`` per-step latents -> ``[B, T, frame_stack, D]`` dilated stacks.

    One predictor step is one history hop in the dream, matching
    ``LatentHistory(frame_stack, 1)`` in the dream trainer; early steps repeat
    the oldest available latent exactly as that buffer does.
    """
    b, t, d = sequence.shape
    idx = torch.arange(t, device=sequence.device)[:, None] - torch.arange(
        frame_stack - 1, -1, -1, device=sequence.device
    )[None, :]
    idx = idx.clamp(min=0)
    return sequence[:, idx.reshape(-1)].reshape(b, t, frame_stack, d)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)

    wm = load_lewm_world_model(args.wm_checkpoint, cache_dir=repo_path(args.wm_cache_dir), device=device)
    projector = wm.projector.eval().requires_grad_(False)
    cls_encoder = LeWMEncoder(wm.encoder, device=device)
    history_size = int(getattr(wm.predictor, "num_frames", 3) or 3)

    deprojector = load_deprojector(repo_path(args.deprojector), device)
    decoder = None
    if not args.skip_decoder:
        from src.ppo.train_lewm import _load_decoder

        decoder = _load_decoder(repo_path(args.decoder_checkpoint), device)

    policy, frame_stack = load_raw_cls_bc_policy(args.bc_checkpoint, args.bc_stats, device)

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
            h5,
            wm,
            cls_encoder,
            anchors,
            context_steps=args.context_steps,
            frameskip=args.frameskip,
            horizon=args.horizon,
            history_size=history_size,
            group_size=args.group_size,
            encode_batch=args.encode_batch,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
            progress_every=5,
        )
    finally:
        h5.close()

    cls = cls.to(device)
    imagined = imagined.to(device)
    context_steps = args.context_steps

    bridges = {"deprojector": lambda z: deprojector(z)}
    if decoder is not None:
        bridges["decoder"] = lambda z: cls_encoder(decoder(z).clamp(0.0, 1.0))

    # Sequences the policy would actually see: real context, then bridged latents.
    truth_seq = torch.cat([cls[:, :context_steps], cls[:, context_steps:]], dim=1)
    truth_stacks = dilated_stacks(truth_seq, frame_stack)
    with torch.no_grad():
        truth_actions = policy(truth_stacks.reshape(-1, frame_stack, cls.shape[-1]))
    truth_actions = truth_actions.reshape(truth_seq.shape[0], truth_seq.shape[1], -1)

    results = {name: [] for name in bridges}
    drift = []
    with torch.no_grad():
        for name, bridge in bridges.items():
            bridged = torch.stack(
                [bridge(imagined[:, t]) for t in range(imagined.shape[1])], dim=1
            )
            seq = torch.cat([cls[:, :context_steps], bridged], dim=1)
            stacks = dilated_stacks(seq, frame_stack)
            actions = policy(stacks.reshape(-1, frame_stack, cls.shape[-1]))
            actions = actions.reshape(seq.shape[0], seq.shape[1], -1)
            for t in range(imagined.shape[1]):
                target = cls[:, context_steps + t]
                undrifted = bridge(projector(target))
                step = context_steps + t
                action_err = torch.sqrt(
                    ((actions[:, step] - truth_actions[:, step]) ** 2).mean()
                ).item()
                results[name].append(
                    {
                        "step": t + 1,
                        "env_steps": (t + 1) * args.frameskip,
                        "end_to_end_rel_l2": rel_l2(bridged[:, t], target),
                        "end_to_end_cos": cosine(bridged[:, t], target),
                        "bridge_only_rel_l2": rel_l2(undrifted, target),
                        "policy_action_rmse": action_err,
                        "policy_action_rmse_px": action_err * PUSHT_ACTION_SCALE,
                    }
                )
        for t in range(imagined.shape[1]):
            projected_truth = projector(cls[:, context_steps + t])
            drift.append(
                {
                    "step": t + 1,
                    "env_steps": (t + 1) * args.frameskip,
                    "wm_drift_rel_l2": rel_l2(imagined[:, t], projected_truth),
                }
            )

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "anchors": len(anchors), "wm_drift": drift, "bridges": results}
    (output_dir / "bridge_horizon.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    header = f"{'step':>5s} {'env':>5s} {'wm_drift':>9s}"
    for name in bridges:
        header += f" | {name[:11]:>11s} e2e  bridge  act_px"
    print("\n" + header)
    for i in range(len(drift)):
        line = f"{drift[i]['step']:5d} {drift[i]['env_steps']:5d} {drift[i]['wm_drift_rel_l2']:9.4f}"
        for name in bridges:
            row = results[name][i]
            line += (
                f" | {row['end_to_end_rel_l2']:15.4f} {row['bridge_only_rel_l2']:7.4f}"
                f" {row['policy_action_rmse_px']:7.2f}"
            )
        print(line)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in drift]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(steps, [row["wm_drift_rel_l2"] for row in drift], "k--", label="world-model drift")
        for name in bridges:
            axes[0].plot(steps, [r["end_to_end_rel_l2"] for r in results[name]], label=f"{name} end-to-end")
            axes[0].plot(
                steps,
                [r["bridge_only_rel_l2"] for r in results[name]],
                linestyle=":",
                label=f"{name} bridge only",
            )
        axes[0].set_xlabel("predictor steps imagined")
        axes[0].set_ylabel("relative L2")
        axes[0].set_title("latent error vs horizon")
        axes[0].legend(fontsize=8)
        for name in bridges:
            axes[1].plot(steps, [r["policy_action_rmse_px"] for r in results[name]], label=name)
        axes[1].set_xlabel("predictor steps imagined")
        axes[1].set_ylabel("action RMSE vs true-CLS policy (px)")
        axes[1].set_title("what the bridge costs the policy")
        axes[1].legend(fontsize=8)
        fig.savefig(output_dir / "bridge_horizon.png", dpi=150)
        print(f"\nsaved {output_dir / 'bridge_horizon.png'}")
    except ImportError:
        pass
    print(f"saved {output_dir / 'bridge_horizon.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
