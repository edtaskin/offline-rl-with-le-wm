"""Command-line entry point for the isolated visual-robustness study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.agents import resolve_device

from .analysis import analyze
from .cache import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_ROOT,
    build_cache,
    save_manifest,
    save_thumbnail_grid,
)
from .encoders import load_encoder
from .evaluation import evaluate_checkpoint, save_environment_screenshots
from .shifts import ADAPTATION_CONDITIONS, CONDITION_NAMES
from .training import model_paths, train_policy


TRAINING_SEEDS = (42, 43, 44)
ENCODERS = ("lewm", "dinov2")


def _add_paths(parser):
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def _add_encoder_paths(parser):
    parser.add_argument("--lewm-checkpoint", default=None)
    parser.add_argument("--dinov2-checkpoint", default=None)
    parser.add_argument("--dinov2-repo", default=None)


def _load_named_encoder(name, args, device):
    return load_encoder(
        name,
        device,
        lewm_checkpoint=getattr(args, "lewm_checkpoint", None),
        dinov2_checkpoint=getattr(args, "dinov2_checkpoint", None),
        dinov2_repo=getattr(args, "dinov2_repo", None),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Isolated frozen-encoder PushT visual-robustness ablation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="render states and build latent caches")
    _add_paths(cache)
    _add_encoder_paths(cache)
    cache.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    cache.add_argument(
        "--conditions", nargs="+", choices=CONDITION_NAMES, default=list(CONDITION_NAMES)
    )
    cache.add_argument("--device", default="auto")
    cache.add_argument("--batch-size", type=int, default=128)
    cache.add_argument("--resolution", type=int, default=96)
    cache.add_argument("--thumbnail-count", type=int, default=4)
    cache.add_argument("--screenshot-seeds", nargs="+", type=int, default=[1000, 1001])
    cache.add_argument("--block-start-radius", type=float, default=200.0)
    cache.add_argument("--force", action="store_true")

    screenshots = subparsers.add_parser(
        "screenshots", help="save paired screenshots from disrupted evaluation envs"
    )
    screenshots.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    screenshots.add_argument(
        "--conditions", nargs="+", choices=CONDITION_NAMES, default=list(CONDITION_NAMES)
    )
    screenshots.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001])
    screenshots.add_argument("--block-start-radius", type=float, default=200.0)
    screenshots.add_argument("--max-episode-steps", type=int, default=300)
    screenshots.add_argument("--observation-resolution", type=int, default=96)

    train = subparsers.add_parser("train", help="train BC heads from ablation caches")
    _add_paths(train)
    train.add_argument("--encoders", nargs="+", choices=ENCODERS, default=["lewm"])
    train.add_argument("--conditions", nargs="+", choices=CONDITION_NAMES, default=["clean"])
    train.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate one or more trained heads")
    _add_paths(evaluate)
    _add_encoder_paths(evaluate)
    evaluate.add_argument("--checkpoints", nargs="*", default=None)
    evaluate.add_argument("--encoders", nargs="+", choices=ENCODERS, default=["lewm"])
    evaluate.add_argument(
        "--train-conditions", nargs="+", choices=CONDITION_NAMES, default=["clean"]
    )
    evaluate.add_argument(
        "--conditions", nargs="+", choices=CONDITION_NAMES, default=list(CONDITION_NAMES)
    )
    evaluate.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--episodes", type=int, default=50)
    evaluate.add_argument("--repeats", type=int, default=3)
    evaluate.add_argument("--eval-seed", type=int, default=42)
    evaluate.add_argument("--max-episode-steps", type=int, default=300)
    evaluate.add_argument("--observation-resolution", type=int, default=96)
    evaluate.add_argument("--block-start-radius", type=float, default=200.0)
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("--force", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="aggregate metrics and confidence intervals"
    )
    _add_paths(analyze_parser)
    analyze_parser.add_argument(
        "--encoders", nargs="+", choices=ENCODERS, default=["lewm"]
    )
    analyze_parser.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze_parser.add_argument("--margin", type=float, default=0.10)
    analyze_parser.add_argument(
        "--observation-resolution",
        type=int,
        default=None,
        help=(
            "analyze only one evaluation suite resolution; required when "
            "multiple resolution suites coexist under the output root"
        ),
    )
    analyze_parser.add_argument("--skip-action-metrics", action="store_true")

    all_parser = subparsers.add_parser(
        "all", help="run the full predefined experiment suite"
    )
    _add_paths(all_parser)
    _add_encoder_paths(all_parser)
    all_parser.add_argument(
        "--encoders", nargs="+", choices=ENCODERS, default=["lewm"]
    )
    all_parser.add_argument("--device", default="auto")
    all_parser.add_argument("--cache-batch-size", type=int, default=128)
    all_parser.add_argument("--train-batch-size", type=int, default=64)
    all_parser.add_argument("--epochs", type=int, default=100)
    all_parser.add_argument("--learning-rate", type=float, default=1e-3)
    all_parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    all_parser.add_argument("--episodes", type=int, default=50)
    all_parser.add_argument("--repeats", type=int, default=3)
    all_parser.add_argument("--eval-seed", type=int, default=42)
    all_parser.add_argument("--max-episode-steps", type=int, default=300)
    all_parser.add_argument("--observation-resolution", type=int, default=224)
    all_parser.add_argument("--block-start-radius", type=float, default=200.0)
    all_parser.add_argument("--screenshot-seeds", nargs="+", type=int, default=[1000, 1001])
    all_parser.add_argument("--bootstrap-samples", type=int, default=10000)
    all_parser.add_argument("--video", action="store_true")
    all_parser.add_argument("--force-cache", action="store_true")
    all_parser.add_argument("--force-train", action="store_true")
    all_parser.add_argument("--force-evaluate", action="store_true")
    return parser


def _run_cache(args, encoder_names=None, condition_names=None):
    encoder_names = list(encoder_names or args.encoders)
    condition_names = list(condition_names or args.conditions)
    device = resolve_device(args.device)
    save_manifest(args.output_root, args.data_path, condition_names)
    save_thumbnail_grid(
        data_path=args.data_path,
        output_root=args.output_root,
        condition_names=condition_names,
        resolution=getattr(args, "resolution", 96),
        count=getattr(args, "thumbnail_count", 4),
    )
    save_environment_screenshots(
        output_root=args.output_root,
        condition_names=condition_names,
        seeds=getattr(args, "screenshot_seeds", (1000, 1001)),
        block_start_radius=getattr(args, "block_start_radius", 200.0),
        max_episode_steps=getattr(args, "max_episode_steps", 300),
        observation_resolution=getattr(
            args, "observation_resolution", getattr(args, "resolution", 96)
        ),
    )
    encoders = {}
    for name in encoder_names:
        encoder = _load_named_encoder(name, args, device)
        encoders[name] = encoder
        for condition in condition_names:
            build_cache(
                data_path=args.data_path,
                output_root=args.output_root,
                encoder_name=name,
                encoder=encoder,
                condition_name=condition,
                batch_size=getattr(args, "batch_size", getattr(args, "cache_batch_size", 128)),
                resolution=getattr(args, "resolution", 96),
                force=getattr(args, "force", getattr(args, "force_cache", False)),
            )
    return encoders


def _run_train(args, jobs=None):
    jobs = jobs or [
        (encoder, condition, seed)
        for encoder in args.encoders
        for condition in args.conditions
        for seed in args.seeds
    ]
    checkpoints = []
    for encoder, condition, seed in jobs:
        checkpoints.append(
            train_policy(
                data_path=args.data_path,
                output_root=args.output_root,
                encoder_name=encoder,
                condition_name=condition,
                seed=seed,
                device=args.device,
                epochs=args.epochs,
                batch_size=getattr(args, "batch_size", getattr(args, "train_batch_size", 64)),
                lr=args.learning_rate,
                num_workers=getattr(args, "num_workers", 0),
                force=getattr(args, "force", getattr(args, "force_train", False)),
            )
        )
    return checkpoints


def _run_screenshots(args):
    return save_environment_screenshots(
        output_root=args.output_root,
        condition_names=args.conditions,
        seeds=args.seeds,
        block_start_radius=args.block_start_radius,
        max_episode_steps=args.max_episode_steps,
        observation_resolution=args.observation_resolution,
    )


def _discover_checkpoints(args):
    if args.checkpoints:
        return [Path(path) for path in args.checkpoints]
    checkpoints = []
    for encoder in args.encoders:
        for condition in args.train_conditions:
            for seed in args.seeds:
                checkpoint, _ = model_paths(args.output_root, encoder, condition, seed)
                if not checkpoint.exists():
                    raise FileNotFoundError(f"trained head does not exist: {checkpoint}")
                checkpoints.append(checkpoint)
    return checkpoints


def _run_evaluate(args, checkpoints=None, condition_selector=None, encoders=None):
    checkpoints = checkpoints or _discover_checkpoints(args)
    device = resolve_device(args.device)
    encoders = encoders or {}
    outputs = []
    for checkpoint in checkpoints:
        metadata = json.loads(Path(checkpoint).with_name("metadata.json").read_text())
        encoder_name = metadata["encoder"]
        if encoder_name not in encoders:
            encoders[encoder_name] = _load_named_encoder(encoder_name, args, device)
        conditions = condition_selector(metadata) if condition_selector else args.conditions
        outputs.extend(
            evaluate_checkpoint(
                checkpoint_path=checkpoint,
                output_root=args.output_root,
                condition_names=conditions,
                device=device,
                episodes=args.episodes,
                repeats=args.repeats,
                eval_seed=args.eval_seed,
                max_episode_steps=args.max_episode_steps,
                observation_resolution=args.observation_resolution,
                block_start_radius=args.block_start_radius,
                video=args.video,
                force=getattr(args, "force", getattr(args, "force_evaluate", False)),
                encoder=encoders[encoder_name],
            )
        )
    return outputs


def _run_all(args):
    selected_encoders = list(args.encoders)
    args.conditions = list(CONDITION_NAMES)
    encoders = _run_cache(args)
    jobs = [
        (encoder, "clean", seed)
        for encoder in selected_encoders
        for seed in args.seeds
    ]
    if "lewm" in selected_encoders:
        jobs += [
            ("lewm", condition, seed)
            for condition in ADAPTATION_CONDITIONS
            for seed in args.seeds
        ]
    checkpoints = _run_train(args, jobs)

    def conditions_for(metadata):
        if metadata["train_condition"] == "clean":
            return list(CONDITION_NAMES)
        return ["clean", metadata["train_condition"]]

    _run_evaluate(args, checkpoints, conditions_for, encoders)
    return analyze(
        output_root=args.output_root,
        data_path=args.data_path,
        bootstrap_samples=args.bootstrap_samples,
        margin=0.10,
        include_action_metrics=True,
        encoder_names=selected_encoders,
        observation_resolution=args.observation_resolution,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "cache":
        _run_cache(args)
    elif args.command == "screenshots":
        _run_screenshots(args)
    elif args.command == "train":
        _run_train(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "analyze":
        analyze(
            output_root=args.output_root,
            data_path=args.data_path,
            bootstrap_samples=args.bootstrap_samples,
            margin=args.margin,
            include_action_metrics=not args.skip_action_metrics,
            encoder_names=args.encoders,
            observation_resolution=args.observation_resolution,
        )
    elif args.command == "all":
        _run_all(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
