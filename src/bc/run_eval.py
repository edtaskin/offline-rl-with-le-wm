"""Compatibility entrypoint for BC evaluation.

New code should use ``python -m src.evaluation.evaluate_pusht --agent-type bc``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.bc.tracking import init_wandb
from src.envs import PUSHT_FIXED_TARGET_POSE
from src.evaluation.agents import load_bc_components, make_bc_evaluation_agent
from src.evaluation.pusht import PushTEvalConfig, run_evaluation

def evaluate(args):
    components = load_bc_components(args.checkpoint, args.stats_path, device="auto")
    wandb_run = init_wandb(
        args,
        {
            **vars(args),
            "checkpoint_contract": components.contract,
        },
    )
    if args.render:
        print("WARNING: live --render is not supported by the canonical evaluator; use video output.")
    execution_mode = "temporal-ensemble" if args.temporal_ensemble else "open-loop"
    adapter = make_bc_evaluation_agent(
        checkpoint=args.checkpoint,
        stats_path=args.stats_path,
        components=components,
        execution_mode=execution_mode,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
    )
    video_dir = args.video_path
    if video_dir and "." in video_dir.rsplit("/", 1)[-1]:
        video_dir = video_dir.rsplit(".", 1)[0]
    result = run_evaluation(
        adapter,
        PushTEvalConfig(
            episodes=args.episodes,
            seed=args.eval_seed,
            max_episode_steps=args.max_steps,
            fixed_target_pose=tuple(args.fixed_target_pose),
            fixed_target_block_success=not args.fixed_target_full_state_success,
            fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
            record_video=args.video_path is not None,
            video_dir=video_dir or "runs/eval_videos",
            video_fps=args.video_fps,
        ),
    )
    if wandb_run is not None:
        for episode in result.episodes:
            wandb_run.log(
                {
                    "eval/episode_return": episode.episode_return,
                    "eval/episode_length": episode.length,
                    "eval/episode_success": episode.success,
                },
                step=episode.episode + 1,
            )
        for key, value in result.summary.items():
            wandb_run.summary[f"eval/{key}"] = value
        wandb_run.finish()
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate latent BC on canonical PushT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument(
        "--fixed_target_pose",
        type=float,
        nargs=3,
        default=PUSHT_FIXED_TARGET_POSE.tolist(),
        metavar=("X", "Y", "ANGLE"),
    )
    parser.add_argument("--fixed_target_full_state_success", action="store_true")
    parser.add_argument("--fixed_target_max_reset_attempts", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--action_chunk_size", type=int, default=5)
    parser.add_argument("--temporal_ensemble", action="store_true")
    parser.add_argument("--temporal_ensemble_decay", type=float, default=0.01)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="offline-rl-lewm")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default="pusht-latent-bc-eval")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    return parser


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
