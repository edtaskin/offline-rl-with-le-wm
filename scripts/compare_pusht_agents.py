"""Run paired deterministic PushT comparisons for latent BC and PPO."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.evaluation.agents import (
    load_bc_components,
    load_ppo_components,
    make_bc_evaluation_agent,
    make_ppo_evaluation_agent,
)
from src.evaluation.pusht import PushTEvalConfig, run_evaluation


TASKS = {
    "training_matched": {
        "block_start_radius": 200.0,
    },
    "fixed_unrestricted": {},
}
EXECUTION_MODES = ("open-loop", "receding-horizon", "temporal-ensemble")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-checkpoint",
        default="hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc.pth",
    )
    parser.add_argument(
        "--bc-stats",
        default="hf://offline-rl-with-le-wm/behavioral-cloning/pusht_latent_bc_stats.pth",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default="hf://offline-rl-with-le-wm/ppo/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", default="runs/pusht_comparison")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def save_result(output_dir, name, result):
    path = output_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def normal_record(suite, scenario, execution_mode, result):
    summary = result.summary
    return {
        "suite": suite,
        "scenario": scenario,
        "agent": result.agent_type,
        "execution_mode": execution_mode,
        "episodes": summary["episodes"],
        "success_rate": summary["success_rate"],
        "mean_return": summary["mean_return"],
        "mean_length": summary["mean_length"],
        "mean_final_block_pos_dist": summary.get("mean_final_block_pos_dist"),
        "mean_final_block_angle_dist": summary.get("mean_final_block_angle_dist"),
    }


def write_summary(output_dir, args, records):
    payload = {"settings": vars(args), "records": records}
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    fieldnames = list(records[0])
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BC components...")
    bc_components = load_bc_components(args.bc_checkpoint, args.bc_stats, args.device)
    print("Loading PPO components...")
    ppo_components = load_ppo_components(args.ppo_checkpoint, args.device)
    factories = {
        "bc": lambda mode: make_bc_evaluation_agent(
            checkpoint=args.bc_checkpoint,
            stats_path=args.bc_stats,
            components=bc_components,
            execution_mode=mode,
            temporal_ensemble_decay=args.temporal_ensemble_decay,
        ),
        "ppo": lambda mode: make_ppo_evaluation_agent(
            checkpoint=args.ppo_checkpoint,
            components=ppo_components,
            execution_mode=mode,
            temporal_ensemble_decay=args.temporal_ensemble_decay,
            deterministic=True,
        ),
    }
    records = []
    training_matched_results = {}

    for scenario, task_kwargs in TASKS.items():
        for agent_type, factory in factories.items():
            print(f"\n=== task={scenario} agent={agent_type} mode=open-loop ===")
            result = run_evaluation(
                factory("open-loop"),
                PushTEvalConfig(
                    episodes=args.episodes,
                    seed=args.seed,
                    max_episode_steps=args.max_episode_steps,
                    **task_kwargs,
                ),
            )
            save_result(output_dir, f"task_{scenario}_{agent_type}_open_loop", result)
            records.append(normal_record("task", scenario, "open-loop", result))
            if scenario == "training_matched":
                training_matched_results[agent_type] = result

    for execution_mode in EXECUTION_MODES:
        for agent_type, factory in factories.items():
            if execution_mode == "open-loop":
                result = training_matched_results[agent_type]
            else:
                print(
                    f"\n=== task=training_matched agent={agent_type} "
                    f"mode={execution_mode} ==="
                )
                result = run_evaluation(
                    factory(execution_mode),
                    PushTEvalConfig(
                        episodes=args.episodes,
                        seed=args.seed,
                        max_episode_steps=args.max_episode_steps,
                        **TASKS["training_matched"],
                    ),
                )
                save_result(
                    output_dir,
                    f"execution_training_matched_{agent_type}_{execution_mode.replace('-', '_')}",
                    result,
                )
            records.append(
                normal_record("execution", "training_matched", execution_mode, result)
            )

    write_summary(output_dir, args, records)
    print(f"\nSaved comparison outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
