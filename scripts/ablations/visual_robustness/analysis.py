"""Aggregation and paired statistics for the visual-robustness ablation."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.bc.dataset import PushTLatentDataset

from .cache import cache_path
from .shifts import CONDITION_NAMES
from .training import load_policy


BASELINE_TRAIN_CONDITIONS = ("clean", "resolution_224")


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_evaluations(
    output_root,
    encoder_names=None,
    observation_resolution=None,
):
    if observation_resolution is not None and int(observation_resolution) < 1:
        raise ValueError("observation_resolution must be positive")
    selected_encoders = set(encoder_names) if encoder_names is not None else None
    records = []
    for path in sorted(Path(output_root).glob("evaluations/**/eval_*.json")):
        payload = json.loads(path.read_text())
        ablation = payload.get("ablation")
        if ablation and (
            selected_encoders is None or ablation.get("encoder") in selected_encoders
        ):
            base_resolution = ablation.get("evaluation_base_resolution")
            if base_resolution is None:
                base_resolution = payload.get("config", {}).get(
                    "observation_resolution"
                )
            if (
                observation_resolution is not None
                and base_resolution != int(observation_resolution)
            ):
                continue
            records.append({"path": str(path), "payload": payload, **ablation})
    return records


def validate_evaluation_protocols(records):
    """Reject analyses that would silently pool incompatible evaluation runs."""

    if not records:
        return
    protocol_keys = (
        "episodes",
        "repeats",
        "repeat_seeds",
        "seed",
        "max_episode_steps",
        "block_start_radius",
    )
    signatures = {
        tuple(
            json.dumps(record["payload"].get("config", {}).get(key), sort_keys=True)
            for key in protocol_keys
        )
        for record in records
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "evaluation artifacts use incompatible episode/repeat/seed protocols; "
            "replace stale results before analyzing"
        )
    resolutions = defaultdict(set)
    for record in records:
        resolutions[record["eval_condition"]].add(
            record["payload"].get("config", {}).get("observation_resolution")
        )
    mismatched = sorted(
        condition for condition, values in resolutions.items() if len(values) != 1
    )
    if mismatched:
        raise RuntimeError(
            "evaluation artifacts mix observation resolutions for conditions: "
            + ", ".join(mismatched)
        )


def _episode_map(record, field="success"):
    result = {}
    for episode in record["payload"]["episodes"]:
        value = (
            episode[field]
            if field in episode
            else episode.get("final_metrics", {}).get(field)
        )
        if value is not None:
            result[int(episode["seed"])] = float(value)
    return result


def _group_records(records):
    return {
        (
            row["encoder"],
            row["train_condition"],
            int(row["training_seed"]),
            row["eval_condition"],
        ): row
        for row in records
    }


def _hierarchical_interval(differences, *, samples=10000, seed=12345):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not differences:
        return float("nan"), float("nan"), float("nan")
    model_seeds = sorted(differences)
    observed = float(np.mean([np.mean(differences[key]) for key in model_seeds]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.choice(model_seeds, size=len(model_seeds), replace=True)
        model_means = []
        for selected_seed in selected:
            values = np.asarray(differences[int(selected_seed)], dtype=np.float64)
            model_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        draws[draw] = np.mean(model_means)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper)


def evaluation_rows(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["encoder"], record["train_condition"], record["eval_condition"])
        grouped[key].append(record)
    rows = []
    for (encoder, train_condition, eval_condition), group in sorted(grouped.items()):
        summaries = [item["payload"]["summary"] for item in group]
        row = {
            "encoder": encoder,
            "train_condition": train_condition,
            "eval_condition": eval_condition,
            "training_seeds": len(group),
        }
        for key in (
            "success_rate",
            "mean_return",
            "mean_final_block_pos_dist",
            "mean_final_block_angle_dist",
        ):
            values = [float(summary[key]) for summary in summaries if key in summary]
            row[key] = float(np.mean(values)) if values else ""
        rows.append(row)
    return rows


def paired_robustness_rows(records, bootstrap_samples=10000, margin=0.10):
    lookup = _group_records(records)
    comparisons = defaultdict(dict)
    for encoder, train_condition, training_seed, eval_condition in lookup:
        if (
            train_condition not in BASELINE_TRAIN_CONDITIONS
            or eval_condition == "clean"
        ):
            continue
        shifted = lookup[(encoder, train_condition, training_seed, eval_condition)]
        clean = lookup.get((encoder, train_condition, training_seed, "clean"))
        if clean is None:
            continue
        clean_values = _episode_map(clean)
        shifted_values = _episode_map(shifted)
        episode_seeds = sorted(set(clean_values) & set(shifted_values))
        if episode_seeds:
            comparisons[(encoder, train_condition, eval_condition)][training_seed] = [
                shifted_values[key] - clean_values[key] for key in episode_seeds
            ]
    rows = []
    for (encoder, train_condition, eval_condition), differences in sorted(comparisons.items()):
        estimate, lower, upper = _hierarchical_interval(
            differences, samples=bootstrap_samples
        )
        rows.append(
            {
                "encoder": encoder,
                "train_condition": train_condition,
                "eval_condition": eval_condition,
                "training_seeds": len(differences),
                "success_delta": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "noninferiority_margin": -float(margin),
                "noninferior": bool(lower > -float(margin)),
            }
        )
    return rows


def adaptation_rows(records, bootstrap_samples=10000, margin=0.10):
    lookup = _group_records(records)
    comparisons = defaultdict(dict)
    for encoder, train_condition, training_seed, eval_condition in lookup:
        if (
            train_condition in BASELINE_TRAIN_CONDITIONS
            or eval_condition != train_condition
        ):
            continue
        adapted = lookup[(encoder, train_condition, training_seed, eval_condition)]
        baseline = lookup.get((encoder, "clean", training_seed, "clean"))
        if baseline is None:
            continue
        baseline_values = _episode_map(baseline)
        adapted_values = _episode_map(adapted)
        episode_seeds = sorted(set(baseline_values) & set(adapted_values))
        if episode_seeds:
            comparisons[(encoder, train_condition)][training_seed] = [
                adapted_values[key] - baseline_values[key] for key in episode_seeds
            ]
    rows = []
    for (encoder, train_condition), differences in sorted(comparisons.items()):
        estimate, lower, upper = _hierarchical_interval(
            differences, samples=bootstrap_samples, seed=24680
        )
        rows.append(
            {
                "encoder": encoder,
                "train_condition": train_condition,
                "eval_condition": train_condition,
                "training_seeds": len(differences),
                "matched_delta_vs_clean_baseline": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "noninferiority_margin": -float(margin),
                "matched_performance_recovered": bool(lower > -float(margin)),
            }
        )
    return rows


def difference_in_differences_rows(records, bootstrap_samples=10000):
    lookup = _group_records(records)
    conditions = sorted(
        {
            row["eval_condition"]
            for row in records
            if row["train_condition"] == "clean" and row["eval_condition"] != "clean"
        }
    )
    rows = []
    for condition in conditions:
        model_differences = {}
        training_seeds = sorted(
            {
                int(row["training_seed"])
                for row in records
                if row["train_condition"] == "clean"
            }
        )
        for training_seed in training_seeds:
            required = [
                lookup.get((encoder, "clean", training_seed, eval_condition))
                for encoder in ("lewm", "dinov2")
                for eval_condition in ("clean", condition)
            ]
            if any(item is None for item in required):
                continue
            maps = [_episode_map(item) for item in required]
            episode_seeds = sorted(set.intersection(*(set(values) for values in maps)))
            model_differences[training_seed] = [
                (maps[1][key] - maps[0][key]) - (maps[3][key] - maps[2][key])
                for key in episode_seeds
            ]
        if model_differences:
            estimate, lower, upper = _hierarchical_interval(
                model_differences, samples=bootstrap_samples, seed=54321
            )
            rows.append(
                {
                    "eval_condition": condition,
                    "training_seeds": len(model_differences),
                    "lewm_minus_dinov2_degradation": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "lewm_significantly_better": bool(lower > 0.0),
                }
            )
    return rows


def latent_metric_rows(output_root, encoder_names=("lewm", "dinov2")):
    rows = []
    for encoder_name in encoder_names:
        clean_path = cache_path(output_root, encoder_name, "clean")
        if not clean_path.exists():
            continue
        clean = torch.load(clean_path, map_location="cpu", weights_only=False)["latents"].float()
        for condition_name in CONDITION_NAMES:
            if condition_name == "clean":
                continue
            shifted_path = cache_path(output_root, encoder_name, condition_name)
            if not shifted_path.exists():
                continue
            shifted = torch.load(shifted_path, map_location="cpu", weights_only=False)["latents"].float()
            if shifted.shape != clean.shape:
                raise ValueError(f"paired cache shape mismatch: {clean_path} vs {shifted_path}")
            cosine = torch.nn.functional.cosine_similarity(clean, shifted, dim=-1)
            normalized_l2 = (shifted - clean).norm(dim=-1) / clean.norm(dim=-1).clamp_min(1e-12)
            variance_ratio = shifted.var(dim=0).mean() / clean.var(dim=0).mean().clamp_min(1e-12)
            rows.append(
                {
                    "encoder": encoder_name,
                    "eval_condition": condition_name,
                    "samples": len(clean),
                    "cosine_mean": float(cosine.mean()),
                    "cosine_std": float(cosine.std()),
                    "cosine_p05": float(torch.quantile(cosine, 0.05)),
                    "normalized_l2_mean": float(normalized_l2.mean()),
                    "variance_ratio": float(variance_ratio),
                }
            )
    return rows


def action_metric_rows(
    output_root,
    data_path,
    batch_size=512,
    encoder_names=("lewm", "dinov2"),
):
    selected_encoders = set(encoder_names)
    rows = []
    for metadata_path in sorted(Path(output_root).glob("models/*/clean/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        policy, _ = load_policy(metadata_path.with_name("policy.pth"), "cpu")
        contract = metadata["contract"]
        encoder_name = metadata["encoder"]
        if encoder_name not in selected_encoders:
            continue
        clean_cache = cache_path(output_root, encoder_name, "clean")
        for condition_name in CONDITION_NAMES:
            if condition_name == "clean":
                continue
            shifted_cache = cache_path(output_root, encoder_name, condition_name)
            if not shifted_cache.exists():
                continue
            dataset_args = {
                "frame_stack": int(contract["frame_stack"]),
                "frame_stride": int(contract["frame_stride"]),
                "action_chunk_size": int(contract["action_chunk_size"]),
            }
            clean_data = PushTLatentDataset(data_path, clean_cache, **dataset_args)
            shifted_data = PushTLatentDataset(data_path, shifted_cache, **dataset_args)
            clean_loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False)
            shifted_loader = DataLoader(shifted_data, batch_size=batch_size, shuffle=False)
            absolute, squared = [], []
            with torch.no_grad():
                for (clean_x, _), (shifted_x, _) in zip(clean_loader, shifted_loader):
                    difference = policy(shifted_x) - policy(clean_x)
                    absolute.append(difference.abs().reshape(-1))
                    squared.append(difference.square().reshape(-1))
            rows.append(
                {
                    "encoder": encoder_name,
                    "training_seed": metadata["seed"],
                    "eval_condition": condition_name,
                    "action_mae": float(torch.cat(absolute).mean()),
                    "action_rmse": float(torch.cat(squared).mean().sqrt()),
                }
            )
    return rows


def _save_plots(analysis_dir, evaluation, latent):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    zero_shot = [
        row
        for row in evaluation
        if row["train_condition"] in BASELINE_TRAIN_CONDITIONS
    ]
    if zero_shot:
        fig, ax = plt.subplots(figsize=(12, 4))
        for encoder in ("lewm", "dinov2"):
            train_conditions = sorted(
                {
                    row["train_condition"]
                    for row in zero_shot
                    if row["encoder"] == encoder
                }
            )
            for train_condition in train_conditions:
                values = {
                    row["eval_condition"]: row["success_rate"]
                    for row in zero_shot
                    if row["encoder"] == encoder
                    and row["train_condition"] == train_condition
                }
                names = [name for name in CONDITION_NAMES if name in values]
                label = encoder
                if len(train_conditions) > 1 or train_condition != "clean":
                    label = f"{encoder} train={train_condition}"
                ax.plot(
                    names,
                    [values[name] for name in names],
                    marker="o",
                    label=label,
                )
        ax.set_ylabel("Success rate")
        ax.set_title("Clean-trained zero-shot performance")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        path = analysis_dir / "zero_shot_success.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    if latent:
        fig, ax = plt.subplots(figsize=(12, 4))
        for encoder in ("lewm", "dinov2"):
            values = {
                row["eval_condition"]: row["cosine_mean"]
                for row in latent
                if row["encoder"] == encoder
            }
            names = [name for name in CONDITION_NAMES if name in values]
            ax.plot(names, [values[name] for name in names], marker="o", label=encoder)
        ax.set_ylabel("Paired latent cosine similarity")
        ax.set_title("Representation invariance")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        path = analysis_dir / "latent_similarity.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def analyze(
    *,
    output_root,
    data_path,
    bootstrap_samples=10000,
    margin=0.10,
    include_action_metrics=True,
    encoder_names=("lewm", "dinov2"),
    observation_resolution=None,
):
    if bootstrap_samples < 1 or margin < 0:
        raise ValueError("bootstrap_samples must be positive and margin non-negative")
    encoder_names = tuple(encoder_names)
    records = load_evaluations(
        output_root,
        encoder_names,
        observation_resolution=observation_resolution,
    )
    validate_evaluation_protocols(records)
    evaluation = evaluation_rows(records)
    robustness = paired_robustness_rows(records, bootstrap_samples, margin)
    adaptation = adaptation_rows(records, bootstrap_samples, margin)
    comparative = difference_in_differences_rows(records, bootstrap_samples)
    latent = latent_metric_rows(output_root, encoder_names)
    actions = (
        action_metric_rows(
            output_root,
            data_path,
            encoder_names=encoder_names,
        )
        if include_action_metrics
        else []
    )
    analysis_dir = Path(output_root) / "analysis"
    if observation_resolution is not None:
        analysis_dir = analysis_dir / f"resolution_{int(observation_resolution)}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "evaluation_summary": evaluation,
        "paired_robustness": robustness,
        "adaptation": adaptation,
        "difference_in_differences": comparative,
        "latent_metrics": latent,
        "action_metrics": actions,
    }
    for name, rows in tables.items():
        _write_csv(analysis_dir / f"{name}.csv", rows)
    plots = _save_plots(analysis_dir, evaluation, latent)
    report = {
        "encoders": list(encoder_names),
        "evaluation_base_resolution": (
            int(observation_resolution)
            if observation_resolution is not None
            else None
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "noninferiority_margin": -float(margin),
        "tables": tables,
        "plots": [str(path) for path in plots],
    }
    report_path = analysis_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path
