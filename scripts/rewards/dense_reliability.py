"""Evaluate dense reward classifier reliability on LeWM imagined rollouts.

This is the dense-reward analogue of the sparse objective-met rollout plot. It
stratifies samples by true time-to-success on the expert trajectory, then
compares dense reward classifier outputs on:

  1. encoded GT target-frame latents
  2. LeWM imagined target latents rolled out with GT actions

The main poster plot is mean dense score versus true time-to-success bin. A
good dense reward should increase as states get closer to success, and the
imagined curve should track the encoded-GT curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel as swm
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rollouts import state_probes as rollout_eval  # noqa: E402
from src.ppo.dense_reward import DenseRewardShaper, parse_dense_reward_weights  # noqa: E402


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


AXIS_COLOR = "0.35"
TITLE_COLOR = "0.22"
SPINE_COLOR = "0.65"
GRID_COLOR = "0.86"


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    if title is not None:
        ax.set_title(title, color=TITLE_COLOR, fontsize=12, fontweight="semibold", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=AXIS_COLOR, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=AXIS_COLOR, labelpad=6)
    ax.tick_params(axis="both", colors=AXIS_COLOR, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_color(SPINE_COLOR)
    if grid:
        ax.grid(True, alpha=0.45, color=GRID_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="le-wm/models/datasets/pusht_expert_train.h5")
    parser.add_argument("--checkpoint-cache-dir", default="le-wm/models")
    parser.add_argument("--checkpoint", default="hf_pusht/weights.pt")
    parser.add_argument(
        "--dense-reward-checkpoint",
        default="models/probes/pusht_dense_reward_1M_imagined50_hn_mono_fpr_0_05/dense_reward_classifier.pt",
    )
    parser.add_argument("--dense-reward-weights", default="1 0.75 0.4 0.1")
    parser.add_argument("--output-dir", default="models/rollout_probe/dense_reward_reliability")
    parser.add_argument("--samples-per-bin-step", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--exact-tts-max", type=int, default=20)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--objective-x", type=float, default=256.0)
    parser.add_argument("--objective-y", type=float, default=256.0)
    parser.add_argument("--objective-angle", type=float, default=float(np.pi / 4))
    parser.add_argument("--objective-pos-tol", type=float, default=20.0)
    parser.add_argument("--objective-angle-tol", type=float, default=float(np.pi / 9))
    parser.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable dataset z-score normalization of GT actions before feeding LeWM.",
    )
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Regenerate plots from existing dense reward CSV outputs without loading LeWM.",
    )
    return parser.parse_args()


def objective_met_mask(states: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return rollout_eval.ground_truth_targets(states, args)["objective_met"][..., 0].astype(bool)


def next_success_rows(offsets: np.ndarray, lengths: np.ndarray, met: np.ndarray) -> np.ndarray:
    next_rows = np.full(len(met), -1, dtype=np.int64)
    for offset, length in zip(offsets, lengths):
        offset = int(offset)
        length = int(length)
        next_success = -1
        for row in range(offset + length - 1, offset - 1, -1):
            if met[row]:
                next_success = row
            next_rows[row] = next_success
    return next_rows


def bin_specs(horizons: list[int]) -> list[dict[str, object]]:
    horizons = sorted(int(h) for h in horizons)
    specs: list[dict[str, object]] = [
        {
            "key": f"gt_{horizons[-1]}wm",
            "label": f">{horizons[-1]} wm / no success",
            "lo": horizons[-1],
            "hi": None,
            "success_now": False,
        }
    ]
    for lo, hi in zip(reversed(horizons[:-1]), reversed(horizons[1:])):
        specs.append({"key": f"{lo}_{hi}wm", "label": f"{lo}-{hi} wm", "lo": lo, "hi": hi, "success_now": False})
    specs.append(
        {
            "key": f"le_{horizons[0]}wm",
            "label": f"0-{horizons[0]} wm",
            "lo": 0,
            "hi": horizons[0],
            "success_now": False,
        }
    )
    specs.append({"key": "success_now", "label": "success now", "lo": 0, "hi": 0, "success_now": True})
    return specs


def assign_time_to_success_bins(
    target_rows: np.ndarray,
    next_success: np.ndarray,
    horizons: list[int],
    frameskip: int,
) -> np.ndarray:
    specs = bin_specs(horizons)
    keys = np.asarray([str(spec["key"]) for spec in specs], dtype=object)
    out = np.empty(len(target_rows), dtype=object)
    out[:] = keys[0]

    next_rows = next_success[target_rows]
    has_success = next_rows >= 0
    tts = np.full(len(target_rows), np.inf, dtype=np.float32)
    tts[has_success] = np.ceil((next_rows[has_success] - target_rows[has_success]) / float(frameskip))
    tts = np.maximum(tts, 0.0)

    for spec in specs:
        key = str(spec["key"])
        if bool(spec["success_now"]):
            keep = tts == 0
        else:
            lo = float(spec["lo"])
            hi = spec["hi"]
            if hi is None:
                keep = tts > lo
            elif lo == 0:
                keep = (tts > 0) & (tts <= float(hi))
            else:
                keep = (tts > lo) & (tts <= float(hi))
        out[keep] = key
    return out


def exact_time_to_success_wm(target_rows: np.ndarray, next_success: np.ndarray, frameskip: int) -> np.ndarray:
    target_rows = target_rows.astype(np.int64)
    next_rows = next_success[target_rows]
    out = np.full(len(target_rows), np.inf, dtype=np.float32)
    has_success = next_rows >= 0
    out[has_success] = np.ceil((next_rows[has_success] - target_rows[has_success]) / float(frameskip))
    return np.maximum(out, 0.0)


def sample_bin_starts(
    offsets: np.ndarray,
    lengths: np.ndarray,
    bin_for_row: np.ndarray,
    bin_key: str,
    model_step: int,
    count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    required_last_frame = (args.context_steps + int(args.horizon) - 1) * args.frameskip
    target_offset = (args.context_steps + model_step - 1) * args.frameskip
    ep_parts = []
    start_parts = []

    for ep, (offset, length) in enumerate(zip(offsets, lengths)):
        offset = int(offset)
        length = int(length)
        max_start_local = length - required_last_frame - 1
        if max_start_local < 0:
            continue
        local_starts = np.arange(max_start_local + 1, dtype=np.int64)
        starts = offset + local_starts
        target_rows = starts + target_offset
        keep = bin_for_row[target_rows] == bin_key
        if not np.any(keep):
            continue
        start_parts.append(starts[keep])
        ep_parts.append(np.full(int(keep.sum()), ep, dtype=np.int64))

    if not start_parts:
        return []
    starts = np.concatenate(start_parts)
    eps = np.concatenate(ep_parts)
    replace = len(starts) < count
    if replace:
        print(
            f"step {model_step} bin {bin_key}: "
            f"{len(starts)} candidates for {count} requested, sampling with replacement"
        )
    idx = rng.choice(len(starts), size=count, replace=replace)
    return [(int(eps[i]), int(starts[i])) for i in idx]


@torch.inference_mode()
def dense_outputs(
    shaper: DenseRewardShaper,
    z: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    probs = []
    for start in range(0, len(z), batch_size):
        chunk = torch.from_numpy(z[start : start + batch_size]).to(device=device, dtype=torch.float32)
        score, prob = shaper.score(chunk)
        scores.append(score.detach().cpu().float().numpy())
        probs.append(prob.detach().cpu().float().numpy())
    return np.concatenate(scores, axis=0), np.concatenate(probs, axis=0)


@torch.inference_mode()
def evaluate_starts_at_step(
    args: argparse.Namespace,
    h5: h5py.File,
    model: torch.nn.Module,
    shaper: DenseRewardShaper,
    starts: list[tuple[int, int]],
    model_step: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    history_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    imagined_embs = []
    encoded_embs = []

    for start in range(0, len(starts), args.batch_size):
        batch_starts = starts[start : start + args.batch_size]
        context_pixels, future_pixels, action_blocks, _ = rollout_eval.load_batch(
            h5,
            batch_starts,
            context_steps=args.context_steps,
            horizon=model_step,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
            normalize_actions=not args.no_normalize_actions,
        )
        pred_emb = rollout_eval.rollout_embeddings(
            model,
            context_pixels,
            action_blocks,
            horizon=model_step,
            history_size=history_size,
            device=device,
        )[:, -1]
        gt_emb = rollout_eval.encode_future_embeddings(
            model,
            future_pixels,
            encode_batch_size=args.encode_batch_size,
            device=device,
        )[:, -1]
        imagined_embs.append(pred_emb)
        encoded_embs.append(gt_emb)

    imagined_z = np.concatenate(imagined_embs, axis=0)
    encoded_z = np.concatenate(encoded_embs, axis=0)
    imagined_score, imagined_prob = dense_outputs(shaper, imagined_z, args.batch_size, device)
    encoded_score, encoded_prob = dense_outputs(shaper, encoded_z, args.batch_size, device)
    return {
        "imagined_z": imagined_z,
        "encoded_gt_z": encoded_z,
        "imagined_score": imagined_score,
        "encoded_gt_score": encoded_score,
        "imagined_prob": imagined_prob,
        "encoded_gt_prob": encoded_prob,
    }


def latent_pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    numerator = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = float(np.mean(numerator / np.maximum(denom, 1e-8)))
    return rmse, cosine


def summarize_values(
    bin_key: str,
    bin_label: str,
    model_step: int,
    outputs: dict[str, np.ndarray],
    horizons: list[int],
    sample_count: int,
    frameskip: int,
) -> dict[str, float | str]:
    rmse, cosine = latent_pair_metrics(outputs["imagined_z"], outputs["encoded_gt_z"])
    row: dict[str, float | str] = {
        "bin": bin_key,
        "bin_label": bin_label,
        "model_step": float(model_step),
        "env_step": float(model_step * frameskip),
        "sample_count": float(sample_count),
        "encoded_gt_score_mean": float(outputs["encoded_gt_score"].mean()),
        "encoded_gt_score_std": float(outputs["encoded_gt_score"].std()),
        "imagined_score_mean": float(outputs["imagined_score"].mean()),
        "imagined_score_std": float(outputs["imagined_score"].std()),
        "score_gap_imagined_minus_encoded": float(outputs["imagined_score"].mean() - outputs["encoded_gt_score"].mean()),
        "latent_rmse": rmse,
        "latent_cosine": cosine,
    }
    for i, horizon in enumerate(horizons):
        row[f"encoded_gt_prob_{horizon}wm_mean"] = float(outputs["encoded_gt_prob"][:, i].mean())
        row[f"imagined_prob_{horizon}wm_mean"] = float(outputs["imagined_prob"][:, i].mean())
        row[f"prob_gap_{horizon}wm"] = float(outputs["imagined_prob"][:, i].mean() - outputs["encoded_gt_prob"][:, i].mean())
    return row


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"bin", "bin_label", "time_to_success_label"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def horizons_from_rows(rows: list[dict[str, float | str]]) -> list[int]:
    horizons = set()
    for row in rows:
        for key in row:
            prefix = "imagined_prob_"
            suffix = "wm_mean"
            if key.startswith(prefix) and key.endswith(suffix):
                horizons.add(int(key.removeprefix(prefix).removesuffix(suffix)))
    return sorted(horizons)


def specs_from_rows(rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    specs = []
    seen = set()
    for row in rows:
        key = str(row["bin"])
        if key in seen:
            continue
        seen.add(key)
        specs.append({"key": key, "label": str(row["bin_label"])})
    return specs


def aggregate_by_bin(rows: list[dict[str, float | str]], specs: list[dict[str, object]], horizons: list[int]) -> list[dict[str, float | str]]:
    out = []
    for spec in specs:
        key = str(spec["key"])
        group = [row for row in rows if row["bin"] == key]
        if not group:
            continue
        total = sum(float(row["sample_count"]) for row in group)
        agg: dict[str, float | str] = {"bin": key, "bin_label": str(spec["label"]), "sample_count": total}
        for metric in (
            "encoded_gt_score_mean",
            "imagined_score_mean",
            "score_gap_imagined_minus_encoded",
            "latent_rmse",
            "latent_cosine",
        ):
            agg[metric] = float(
                sum(float(row[metric]) * float(row["sample_count"]) for row in group) / max(total, 1.0)
            )
        for horizon in horizons:
            for prefix in ("encoded_gt", "imagined"):
                metric = f"{prefix}_prob_{horizon}wm_mean"
                agg[metric] = float(
                    sum(float(row[metric]) * float(row["sample_count"]) for row in group) / max(total, 1.0)
                )
        out.append(agg)
    return out


def plot_dense_score(path: Path, rows: list[dict[str, float | str]]) -> None:
    labels = [str(row["bin_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(x, [float(row["imagined_score_mean"]) for row in rows], marker="o", linewidth=2.8, label="imagined rollout")
    ax.plot(
        x,
        [float(row["encoded_gt_score_mean"]) for row in rows],
        marker="o",
        linestyle="--",
        linewidth=2.2,
        label="encoded GT future",
    )
    style_axis(
        ax,
        title="Does the dense reward increase as success gets closer?",
        xlabel="true time to success on expert trajectory",
        ylabel="weighted dense reward score",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_head_probs(path: Path, rows: list[dict[str, float | str]], horizons: list[int]) -> None:
    labels = [str(row["bin_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(9, 2.4 * len(horizons)), sharex=True, constrained_layout=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        ax.plot(x, [float(row[f"imagined_prob_{horizon}wm_mean"]) for row in rows], linewidth=2.4, label="imagined")
        ax.plot(
            x,
            [float(row[f"encoded_gt_prob_{horizon}wm_mean"]) for row in rows],
            linestyle="--",
            linewidth=2.0,
            label="encoded GT",
        )
        style_axis(
            ax,
            title=f"Probability of success within {horizon} world-model steps",
            ylabel="predicted probability",
        )
        ax.set_ylim(0.0, 1.05)
        boundary = horizon_boundary_x(rows, horizon)
        if boundary is not None:
            ax.axvline(
                boundary,
                color="tab:orange",
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=f"{horizon} WM boundary",
            )
        ax.legend(frameon=False)
    axes[-1].set_xlabel("true time to success on expert trajectory", color=AXIS_COLOR, labelpad=6)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bin_is_positive_for_horizon(bin_key: str, horizon: int) -> bool:
    if bin_key == "success_now":
        return True
    if bin_key.startswith("gt_"):
        return False
    if bin_key.startswith("le_") and bin_key.endswith("wm"):
        return int(bin_key.removeprefix("le_").removesuffix("wm")) <= horizon
    if "_" in bin_key and bin_key.endswith("wm"):
        lo, hi = bin_key.removesuffix("wm").split("_", 1)
        return float(hi) <= float(horizon)
    return False


def horizon_boundary_x(rows: list[dict[str, float | str]], horizon: int) -> float | None:
    positive = [bin_is_positive_for_horizon(str(row["bin"]), horizon) for row in rows]
    for i, is_positive in enumerate(positive):
        if is_positive:
            return float(i) - 0.5
    return None


def plot_head_probs_imagined_only(path: Path, rows: list[dict[str, float | str]], horizons: list[int]) -> None:
    labels = [str(row["bin_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(9, 2.4 * len(horizons)), sharex=True, constrained_layout=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        ax.plot(
            x,
            [float(row[f"imagined_prob_{horizon}wm_mean"]) for row in rows],
            linewidth=2.5,
            color="tab:blue",
        )
        style_axis(
            ax,
            title=f"Probability of success within {horizon} world-model steps",
            ylabel="predicted probability",
        )
        ax.set_ylim(0.0, 1.05)
        boundary = horizon_boundary_x(rows, horizon)
        if boundary is not None:
            ax.axvline(
                boundary,
                color="tab:orange",
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
            )
    axes[-1].set_xlabel("true time to success on expert trajectory", color=AXIS_COLOR, labelpad=6)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_head_probs_imagined_only_combined(path: Path, rows: list[dict[str, float | str]], horizons: list[int]) -> None:
    labels = [str(row["bin_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for i, horizon in enumerate(horizons):
        color = colors[i % len(colors)]
        ax.plot(
            x,
            [float(row[f"imagined_prob_{horizon}wm_mean"]) for row in rows],
            linewidth=2.6,
            color=color,
            label=f"{horizon} WM head",
        )
        boundary = horizon_boundary_x(rows, horizon)
        if boundary is not None:
            ax.axvline(
                boundary,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.75,
                label=f"{horizon} WM boundary",
            )
    style_axis(
        ax,
        title="Do dense reward heads activate at the right time?",
        xlabel="true time to success on expert trajectory",
        ylabel="predicted probability on imagined rollout",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def wm_bin_label_to_env_steps(label: str, frameskip: int) -> str:
    if label == "success now":
        return label
    if label.startswith(">") and "wm" in label:
        value = int(label.split("wm", 1)[0].strip().removeprefix(">"))
        return label.replace(f">{value} wm", f">{value * frameskip} env steps")
    if "-" in label and label.endswith("wm"):
        lo, hi = label.removesuffix("wm").strip().split("-", 1)
        return f"{int(lo) * frameskip}-{int(hi) * frameskip} env steps"
    return label.replace("wm", "world-model steps")


def wm_bin_label_to_compact_env_steps(label: str, frameskip: int) -> str:
    if label == "success now":
        return label
    if label.startswith(">") and "wm" in label:
        value = int(label.split("wm", 1)[0].strip().removeprefix(">"))
        return label.replace(f">{value} wm", f">{value * frameskip}").replace(" / no success", " / no success")
    if "-" in label and label.endswith("wm"):
        lo, hi = label.removesuffix("wm").strip().split("-", 1)
        return f"{int(lo) * frameskip}-{int(hi) * frameskip}"
    return label


def plot_head_probs_imagined_only_combined_env_steps(
    path: Path,
    rows: list[dict[str, float | str]],
    horizons: list[int],
    *,
    frameskip: int = 5,
) -> None:
    labels = [wm_bin_label_to_compact_env_steps(str(row["bin_label"]), frameskip) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for i, horizon in enumerate(horizons):
        color = colors[i % len(colors)]
        horizon_env = int(horizon * frameskip)
        ax.plot(
            x,
            [float(row[f"imagined_prob_{horizon}wm_mean"]) for row in rows],
            linewidth=2.6,
            color=color,
            label=f"{horizon_env} env-step head",
        )
        boundary = horizon_boundary_x(rows, horizon)
        if boundary is not None:
            ax.axvline(
                boundary,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.75,
                label=f"{horizon_env} env-step boundary",
            )
    style_axis(
        ax,
        title="Do dense reward heads activate at the right time?",
        xlabel="true time-to-success bins (environment steps)",
        ylabel="predicted probability on imagined rollout",
    )
    ax.title.set_fontsize(17)
    ax.title.set_fontweight("semibold")
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8, ncols=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate_exact_tts_records(
    records: list[dict[str, np.ndarray]],
    horizons: list[int],
    max_tts: int,
) -> list[dict[str, float | str]]:
    if not records:
        return []
    tts = np.concatenate([record["tts"] for record in records], axis=0)
    imagined = np.concatenate([record["imagined_prob"] for record in records], axis=0)
    encoded = np.concatenate([record["encoded_gt_prob"] for record in records], axis=0)
    bin_ids = np.full(len(tts), max_tts + 1, dtype=np.int64)
    finite = np.isfinite(tts) & (tts <= max_tts)
    bin_ids[finite] = tts[finite].astype(np.int64)
    rows: list[dict[str, float | str]] = []
    for bin_id in range(max_tts + 2):
        keep = bin_ids == bin_id
        if not np.any(keep):
            continue
        label = str(bin_id) if bin_id <= max_tts else f">{max_tts} / no success"
        row: dict[str, float | str] = {
            "time_to_success_wm": float(bin_id),
            "time_to_success_label": label,
            "sample_count": float(np.sum(keep)),
        }
        for i, horizon in enumerate(horizons):
            row[f"imagined_prob_{horizon}wm_mean"] = float(imagined[keep, i].mean())
            row[f"encoded_gt_prob_{horizon}wm_mean"] = float(encoded[keep, i].mean())
        rows.append(row)
    return rows


def plot_exact_tts_head_probs(
    path: Path,
    rows: list[dict[str, float | str]],
    horizons: list[int],
    *,
    imagined_only: bool = False,
) -> None:
    if not rows:
        return
    x = np.asarray([float(row["time_to_success_wm"]) for row in rows], dtype=np.float32)
    labels = [str(row["time_to_success_label"]) for row in rows]
    fig, axes = plt.subplots(len(horizons), 1, figsize=(9, 2.4 * len(horizons)), sharex=True, constrained_layout=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        ax.plot(
            x,
            [float(row[f"imagined_prob_{horizon}wm_mean"]) for row in rows],
            color="tab:blue",
            linewidth=2.5,
            label="imagined rollout",
        )
        if not imagined_only:
            ax.plot(
                x,
                [float(row[f"encoded_gt_prob_{horizon}wm_mean"]) for row in rows],
                color="tab:blue",
                linestyle="--",
                linewidth=2.0,
                label="encoded GT future",
            )
        ax.axvline(float(horizon), color="tab:orange", linestyle="--", linewidth=1.8, alpha=0.9, label=f"{horizon} WM")
        style_axis(
            ax,
            title=f"Probability of success within {horizon} world-model steps",
            ylabel="predicted probability",
        )
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8, frameon=False)
    axes[-1].set_xlabel("true time to success (world-model steps)", color=AXIS_COLOR, labelpad=6)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=25, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rollout_heatmap(
    path: Path,
    rows: list[dict[str, float | str]],
    specs: list[dict[str, object]],
    metric: str,
    title: str,
    cmap: str = "viridis",
) -> None:
    steps = sorted({int(row["model_step"]) for row in rows})
    bin_keys = [str(spec["key"]) for spec in specs]
    labels = [str(spec["label"]) for spec in specs]
    matrix = np.full((len(bin_keys), len(steps)), np.nan, dtype=np.float32)
    for row in rows:
        i = bin_keys.index(str(row["bin"]))
        j = steps.index(int(row["model_step"]))
        matrix[i, j] = float(row[metric])
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    style_axis(
        ax,
        title=title,
        xlabel="rollout horizon (environment steps)",
        ylabel="true time-to-success bin",
        grid=False,
    )
    ax.set_xticks(np.arange(len(steps)))
    env_steps = [int(next(row["env_step"] for row in rows if int(row["model_step"]) == step)) for step in steps]
    ax.set_xticklabels([str(step) for step in env_steps])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_all_outputs(
    output_dir: Path,
    aggregate_rows: list[dict[str, float | str]],
    step_rows: list[dict[str, float | str]],
    exact_rows: list[dict[str, float | str]],
    specs: list[dict[str, object]],
    horizons: list[int],
) -> None:
    plot_dense_score(output_dir / "plots" / "dense_score_by_time_to_success.png", aggregate_rows)
    plot_head_probs(output_dir / "plots" / "dense_head_probabilities_by_time_to_success.png", aggregate_rows, horizons)
    plot_head_probs_imagined_only(
        output_dir / "plots" / "dense_head_probabilities_by_time_to_success_imagined_only.png",
        aggregate_rows,
        horizons,
    )
    plot_head_probs_imagined_only_combined(
        output_dir / "plots" / "dense_head_probabilities_by_time_to_success_imagined_only_combined.png",
        aggregate_rows,
        horizons,
    )
    plot_head_probs_imagined_only_combined_env_steps(
        output_dir / "plots" / "dense_head_probabilities_by_time_to_success_imagined_only_combined_env_steps.png",
        aggregate_rows,
        horizons,
    )
    plot_exact_tts_head_probs(
        output_dir / "plots" / "dense_head_probabilities_by_exact_time_to_success.png",
        exact_rows,
        horizons,
        imagined_only=False,
    )
    plot_exact_tts_head_probs(
        output_dir / "plots" / "dense_head_probabilities_by_exact_time_to_success_imagined_only.png",
        exact_rows,
        horizons,
        imagined_only=True,
    )
    plot_rollout_heatmap(
        output_dir / "plots" / "imagined_score_by_bin_and_rollout_step.png",
        step_rows,
        specs,
        "imagined_score_mean",
        "Dense reward score across rollout depth",
    )
    plot_rollout_heatmap(
        output_dir / "plots" / "score_gap_by_bin_and_rollout_step.png",
        step_rows,
        specs,
        "score_gap_imagined_minus_encoded",
        "How imagination shifts the dense reward score",
        cmap="coolwarm",
    )
    plot_rollout_heatmap(
        output_dir / "plots" / "latent_cosine_by_bin_and_rollout_step.png",
        step_rows,
        specs,
        "latent_cosine",
        "How similar are imagined latents to encoded futures?",
    )


def plot_existing_outputs(output_dir: Path) -> None:
    aggregate_path = output_dir / "dense_reward_by_bin.csv"
    step_path = output_dir / "dense_reward_by_bin_step.csv"
    exact_path = output_dir / "dense_reward_by_exact_tts.csv"
    if not aggregate_path.exists() or not step_path.exists():
        print(f"skipped dense reward plots, missing {aggregate_path} or {step_path}")
        return

    aggregate_rows = read_csv_rows(aggregate_path)
    step_rows = read_csv_rows(step_path)
    exact_rows = read_csv_rows(exact_path) if exact_path.exists() else []
    horizons = horizons_from_rows(aggregate_rows)
    specs = specs_from_rows(aggregate_rows)
    plot_all_outputs(output_dir, aggregate_rows, step_rows, exact_rows, specs, horizons)
    print(f"regenerated dense reward reliability plots from {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_existing:
        plot_existing_outputs(output_dir)
        return
    device = torch.device(args.device)

    shaper = DenseRewardShaper(
        repo_path(args.dense_reward_checkpoint),
        weights=parse_dense_reward_weights(args.dense_reward_weights),
        scale=1.0,
        clip=0.0,
        device=device,
    ).eval()
    horizons = [int(h) for h in shaper.horizons]
    if args.horizon is None:
        args.horizon = max(horizons)
    if len(parse_dense_reward_weights(args.dense_reward_weights)) != len(horizons):
        raise ValueError("--dense-reward-weights must have one value per dense reward head")

    model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=repo_path(args.checkpoint_cache_dir))
    model = model.to(device).eval()
    model.requires_grad_(False)
    history_size = int(getattr(model.predictor, "num_frames", args.context_steps))

    specs = bin_specs(horizons)
    rows: list[dict[str, float | str]] = []
    exact_records: list[dict[str, np.ndarray]] = []
    arrays: dict[str, np.ndarray] = {}

    with h5py.File(repo_path(args.dataset_path), "r") as h5:
        offsets = h5["ep_offset"][:].astype(np.int64)
        lengths = h5["ep_len"][:].astype(np.int64)
        state = h5["state"][:].astype(np.float32)
        met = objective_met_mask(state, args)
        next_success = next_success_rows(offsets, lengths, met)
        all_rows = np.arange(len(state), dtype=np.int64)
        bin_for_row = assign_time_to_success_bins(all_rows, next_success, horizons, args.frameskip)
        action_mean, action_std = rollout_eval.action_stats(h5)

        for model_step in range(1, int(args.horizon) + 1):
            for spec_idx, spec in enumerate(specs):
                key = str(spec["key"])
                rng = np.random.default_rng(args.seed + 100_000 * (spec_idx + 1) + model_step)
                starts = sample_bin_starts(
                    offsets,
                    lengths,
                    bin_for_row,
                    key,
                    model_step,
                    args.samples_per_bin_step,
                    args,
                    rng,
                )
                if not starts:
                    print(f"step {model_step}: no candidates for bin {key}, skipping")
                    continue
                outputs = evaluate_starts_at_step(
                    args,
                    h5,
                    model,
                    shaper,
                    starts,
                    model_step,
                    action_mean,
                    action_std,
                    history_size,
                    device,
                )
                row = summarize_values(
                    key,
                    str(spec["label"]),
                    model_step,
                    outputs,
                    horizons,
                    len(starts),
                    args.frameskip,
                )
                rows.append(row)
                starts_arr = np.asarray(starts, dtype=np.int64)
                target_rows = starts_arr[:, 1] + (args.context_steps + model_step - 1) * args.frameskip
                exact_records.append(
                    {
                        "tts": exact_time_to_success_wm(target_rows, next_success, args.frameskip),
                        "imagined_prob": outputs["imagined_prob"],
                        "encoded_gt_prob": outputs["encoded_gt_prob"],
                    }
                )
                arrays[f"step_{model_step}_{key}_starts"] = np.asarray(starts, dtype=np.int64)
                arrays[f"step_{model_step}_{key}_imagined_score"] = outputs["imagined_score"]
                arrays[f"step_{model_step}_{key}_encoded_gt_score"] = outputs["encoded_gt_score"]
                arrays[f"step_{model_step}_{key}_imagined_prob"] = outputs["imagined_prob"]
                arrays[f"step_{model_step}_{key}_encoded_gt_prob"] = outputs["encoded_gt_prob"]
                print(
                    f"step {model_step}/{args.horizon} bin {key}: "
                    f"score imagined={row['imagined_score_mean']:.3f}, encoded={row['encoded_gt_score_mean']:.3f}"
                )

    aggregate_rows = aggregate_by_bin(rows, specs, horizons)
    exact_rows = aggregate_exact_tts_records(exact_records, horizons, args.exact_tts_max)
    write_csv(output_dir / "dense_reward_by_bin_step.csv", rows)
    write_csv(output_dir / "dense_reward_by_bin.csv", aggregate_rows)
    write_csv(output_dir / "dense_reward_by_exact_tts.csv", exact_rows)
    np.savez_compressed(output_dir / "arrays.npz", **arrays)

    plot_all_outputs(output_dir, aggregate_rows, rows, exact_rows, specs, horizons)

    summary = {
        "config": vars(args),
        "dense_reward_checkpoint": str(repo_path(args.dense_reward_checkpoint)),
        "horizons": horizons,
        "weights": parse_dense_reward_weights(args.dense_reward_weights),
        "poster_plot": str(output_dir / "plots" / "dense_score_by_time_to_success.png"),
        "csv": str(output_dir / "dense_reward_by_bin.csv"),
        "exact_tts_csv": str(output_dir / "dense_reward_by_exact_tts.csv"),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved dense reward rollout reliability analysis to {output_dir}")


if __name__ == "__main__":
    main()
