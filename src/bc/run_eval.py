import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.bc.history import FeatureHistory, temporal_ensemble_action
from src.bc.lewm import (
    LEWM_IMAGE_NORMALIZATION,
    LEWM_LEGACY_IMAGE_NORMALIZATION,
    LeWMFeatureExtractor,
)
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.bc.tracking import init_wandb
from src.bc.video import save_evaluation_video
from src.envs import PUSHT_FIXED_TARGET_POSE, make_pusht_env


def _success_from_info(info):
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return float(info[key])
    return None


def _episode_success(info, episode_terminated):
    info_success = _success_from_info(info)
    if info_success is not None:
        return info_success
    return float(episode_terminated)


def load_stats(stats_path, device="cpu"):
    return torch.load(stats_path, map_location=device)


def _load_checkpoint_contract(args, device):
    stats = load_stats(args.stats_path, device)
    contract = {
        "frame_stack": int(stats.get("frame_stack", args.frame_stack)),
        "frame_stride": int(stats.get("frame_stride", args.frame_stride)),
        "hidden_dim": int(stats.get("hidden_dim", args.hidden_dim)),
        "latent_dim": int(stats.get("latent_dim", args.latent_dim)),
        "action_dim": int(stats.get("action_dim", args.action_dim)),
        "action_chunk_size": int(stats.get("action_chunk_size", 1)),
        "action_space": stats.get("action_space"),
        "image_normalization": stats.get(
            "image_normalization", LEWM_LEGACY_IMAGE_NORMALIZATION
        ),
    }
    if contract["frame_stack"] < 1:
        raise ValueError("frame_stack must be at least 1")
    if contract["frame_stride"] < 1:
        raise ValueError("frame_stride must be at least 1")
    if contract["action_chunk_size"] < 1:
        raise ValueError("action_chunk_size must be at least 1")
    if args.temporal_ensemble_decay < 0.0:
        raise ValueError("temporal_ensemble_decay must be non-negative")

    for key in ("frame_stack", "frame_stride", "hidden_dim", "action_chunk_size"):
        cli_value = getattr(args, key)
        if contract[key] != cli_value:
            print(f"Using {key}={contract[key]} from stats file instead of CLI value {cli_value}.")
    if "action_chunk_size" not in stats:
        print(
            "WARNING: stats file does not declare action_chunk_size. "
            "Assuming an old one-step BC checkpoint; retrain for 5-step chunking."
        )
    if "frame_stride" not in stats:
        print(
            "WARNING: stats file does not declare frame_stride. "
            f"Using CLI/default value {contract['frame_stride']}."
        )
    if contract["action_space"] != "swm_relative":
        print(
            "WARNING: stats file does not declare action_space='swm_relative'. "
            "Old checkpoints trained on absolute pixel actions should be retrained."
        )
    if contract["image_normalization"] != LEWM_IMAGE_NORMALIZATION:
        print(
            "WARNING: stats file does not declare image_normalization='imagenet'. "
            "This checkpoint used legacy /255-only LeWM preprocessing, which made "
            "the latent BC policy collapse to near-mean actions in diagnostics. "
            "Retrain with the current dataset preprocessing."
        )
    return stats, contract


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)
    print(f"Using eval_seed={args.eval_seed}.")

    _, contract = _load_checkpoint_contract(args, device)
    extractor = LeWMFeatureExtractor.load(
        device=device,
        feature_dim=contract["latent_dim"],
        normalization=contract["image_normalization"],
    )
    policy = LatentBCPolicy(
        latent_dim=contract["latent_dim"],
        frame_stack=contract["frame_stack"],
        action_dim=contract["action_dim"],
        hidden_dim=contract["hidden_dim"],
        action_chunk_size=contract["action_chunk_size"],
    ).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()
    wandb_run = init_wandb(
        args,
        {
            **vars(args),
            "device": str(device),
            "checkpoint_contract": contract,
        },
    )
    if wandb_run is not None:
        print(f"Logging evaluation to wandb: {getattr(wandb_run, 'url', None) or 'enabled'}")

    if args.swm_world_eval:
        if args.fixed_target_eval:
            print(
                "WARNING: --fixed_target_eval is ignored with --swm_world_eval; "
                "swm.World dataset eval defines its own start and goal states."
            )
        if args.temporal_ensemble:
            print(
                "WARNING: --temporal_ensemble is ignored by --swm_world_eval; "
                "the World policy adapter uses open-loop action chunks."
            )
        from src.bc.lewm_world_eval import evaluate_lewm_world

        metrics = evaluate_lewm_world(
            args,
            extractor=extractor,
            policy=policy,
            frame_stack=contract["frame_stack"],
            frame_stride=contract["frame_stride"],
            wandb_run=wandb_run,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return metrics

    if args.fixed_target_eval:
        success_mode = "block pose" if not args.fixed_target_full_state_success else "full state"
        print(
            "Using fixed-target PushT eval: "
            f"target_pose={np.round(args.fixed_target_pose, 3).tolist()}, "
            f"success_mode={success_mode}."
        )
    env = make_pusht_env(
        align_sampled_goal_to_fixed_target=args.fixed_target_eval,
        fixed_target_pose=args.fixed_target_pose,
        fixed_target_block_success=not args.fixed_target_full_state_success,
        fixed_target_max_reset_attempts=args.fixed_target_max_reset_attempts,
    )
    env.action_space.seed(args.eval_seed)
    env.observation_space.seed(args.eval_seed)

    action_chunk_size = contract["action_chunk_size"]
    use_temporal_ensemble = args.temporal_ensemble and action_chunk_size > 1
    if args.temporal_ensemble and action_chunk_size == 1:
        print("Temporal ensembling requested, but action_chunk_size=1; using one-step evaluation.")
    if use_temporal_ensemble:
        print(
            "Using ACT-style temporal ensembling: querying every step, "
            f"decay={args.temporal_ensemble_decay:.4f}."
        )
    else:
        print("Using open-loop action chunk execution.")

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episode_traces = []
    video_frames = [] if args.video_path is not None else None

    def encode_observation(observation):
        image = torch.as_tensor(observation, dtype=torch.float32, device=device)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        return extractor.encode(image)

    try:
        for episode_index in range(args.episodes):
            print(f"--- Starting Episode {episode_index + 1}/{args.episodes} ---")
            observation, info = env.reset(seed=args.eval_seed + episode_index)
            if video_frames is not None:
                video_frames.append(np.asarray(observation).copy())
            done = False
            step_count = 0
            episode_return = 0.0
            episode_terminated = False
            feature_history = FeatureHistory(
                contract["frame_stack"], contract["frame_stride"]
            )
            feature_history.append(encode_observation(observation))
            action_buffers = (
                [deque() for _ in range(args.max_steps + action_chunk_size)]
                if use_temporal_ensemble
                else None
            )
            trace = {"actions": [], "rewards": [], "terminated": [], "truncated": []}

            if use_temporal_ensemble:
                while not done and step_count < args.max_steps:
                    stacked_features = feature_history.stacked(device)
                    with torch.no_grad():
                        predicted_chunk = torch.clamp(
                            policy(stacked_features), -1.0, 1.0
                        ).squeeze(0).cpu()
                    for offset, predicted_action in enumerate(predicted_chunk):
                        action_buffers[step_count + offset].append(predicted_action)
                    action_tensor = temporal_ensemble_action(
                        action_buffers[step_count], args.temporal_ensemble_decay
                    )
                    action_array = torch.clamp(action_tensor, -1.0, 1.0).numpy()
                    action_buffers[step_count].clear()
                    observation, reward, terminated, truncated, info = env.step(action_array)
                    trace["actions"].append(np.asarray(action_array).copy())
                    trace["rewards"].append(float(reward))
                    trace["terminated"].append(bool(terminated))
                    trace["truncated"].append(bool(truncated))
                    episode_return += float(reward)
                    episode_terminated = episode_terminated or bool(terminated)
                    if video_frames is not None:
                        video_frames.append(np.asarray(observation).copy())
                    if args.render:
                        import cv2

                        cv2.imshow("PushT Latent BC Evaluation", observation[..., ::-1])
                        cv2.waitKey(1)
                    done = terminated or truncated
                    step_count += 1
                    if not done and step_count < args.max_steps:
                        feature_history.append(encode_observation(observation))
            else:
                while not done and step_count < args.max_steps:
                    stacked_features = feature_history.stacked(device)
                    with torch.no_grad():
                        predicted_chunk = torch.clamp(
                            policy(stacked_features), -1.0, 1.0
                        ).squeeze(0)
                    for action_array in predicted_chunk.cpu().numpy():
                        observation, reward, terminated, truncated, info = env.step(action_array)
                        trace["actions"].append(np.asarray(action_array).copy())
                        trace["rewards"].append(float(reward))
                        trace["terminated"].append(bool(terminated))
                        trace["truncated"].append(bool(truncated))
                        episode_return += float(reward)
                        episode_terminated = episode_terminated or bool(terminated)
                        if video_frames is not None:
                            video_frames.append(np.asarray(observation).copy())
                        if args.render:
                            import cv2

                            cv2.imshow("PushT Latent BC Evaluation", observation[..., ::-1])
                            cv2.waitKey(1)
                        done = terminated or truncated
                        step_count += 1
                        if done or step_count >= args.max_steps:
                            break
                        feature_history.append(encode_observation(observation))

            print(
                f"Episode {episode_index + 1} finished after {step_count} steps. "
                f"Return: {episode_return:.4f}"
            )
            episode_returns.append(episode_return)
            episode_lengths.append(step_count)
            episode_success = _episode_success(info, episode_terminated)
            episode_successes.append(episode_success)
            episode_traces.append(trace)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/episode_return": episode_return,
                        "eval/episode_length": step_count,
                        "eval/episode": episode_index + 1,
                        "eval/episode_success": episode_success,
                    },
                    step=episode_index + 1,
                )
    finally:
        env.close()

    save_evaluation_video(video_frames, args.video_path, args.video_fps)
    summary = {}
    if episode_returns:
        summary = {
            "eval/return_mean": statistics.fmean(episode_returns),
            "eval/return_min": min(episode_returns),
            "eval/return_max": max(episode_returns),
            "eval/length_mean": statistics.fmean(episode_lengths),
            "eval/return_std": statistics.pstdev(episode_returns)
            if len(episode_returns) > 1
            else 0.0,
            "eval/success_rate": statistics.fmean(episode_successes),
        }
        print(
            "Evaluation summary: "
            f"return_mean={summary['eval/return_mean']:.4f}, "
            f"return_std={summary['eval/return_std']:.4f}, "
            f"length_mean={summary['eval/length_mean']:.2f}"
        )
        print(f"Success rate: {summary['eval/success_rate']:.4f}")
        if wandb_run is not None:
            for key, value in summary.items():
                wandb_run.summary[key] = value
            wandb_run.log(summary, step=args.episodes)
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "summary": summary,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "episode_traces": episode_traces,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Latent BC Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved policy weights (.pth)")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to the saved _stats.pth file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    parser.add_argument("--video_path", type=str, default=None, help="Optional path to save evaluation video")
    parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for saved evaluation video")
    parser.add_argument("--fixed_target_eval", action="store_true", help="For normal eval, rigidly align each sampled PushT task to the fixed expert target pose and use block-pose success by default.")
    parser.add_argument("--fixed_target_pose", type=float, nargs=3, default=PUSHT_FIXED_TARGET_POSE.tolist(), metavar=("X", "Y", "ANGLE"), help="Fixed PushT target pose used by --fixed_target_eval.")
    parser.add_argument("--fixed_target_full_state_success", action="store_true", help="With --fixed_target_eval, keep SWM full-state reward/success instead of standard block-pose reward/success.")
    parser.add_argument("--fixed_target_max_reset_attempts", type=int, default=100, help="Maximum resampling attempts when aligned fixed-target starts leave the board.")
    parser.add_argument("--swm_world_eval", action="store_true", help="Evaluate through swm.World.evaluate(dataset=...), matching the LeWM dataset-conditioned eval path.")
    parser.add_argument("--eval_data_path", type=str, default="data/expert_trajectories/pusht_expert.npz", help="NPZ PushT expert dataset used by --swm_world_eval.")
    parser.add_argument("--goal_offset_steps", type=int, default=25, help="Future dataset offset used as the goal by --swm_world_eval.")
    parser.add_argument("--lewm_eval_budget", type=int, default=50, help="Number of env steps for --swm_world_eval, matching the LeWM PushT default.")
    parser.add_argument("--eval_seed", type=int, default=42, help="Random seed for eval sampling, policy/env resets, and reproducible rollouts.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the BC MLP")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of frames to stack (must match training)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Environment steps between stacked history frames")
    parser.add_argument("--latent_dim", type=int, default=192, help="LeWM encoder hidden size")
    parser.add_argument("--action_dim", type=int, default=2, help="Per-step PushT action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=5, help="Number of future actions predicted from one observation")
    parser.add_argument("--temporal_ensemble", action="store_true", help="Query every step and average overlapping predicted action chunks, following ACT inference.")
    parser.add_argument("--temporal_ensemble_decay", type=float, default=0.01, help="Exponential decay for temporal ensembling weights; 0.0 gives a uniform average.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases evaluation tracking")
    parser.add_argument("--wandb_project", type=str, default="offline-rl-lewm", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default="pusht-latent-bc-eval", help="Weights & Biases run group")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional Weights & Biases tags")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default=None, help="Weights & Biases mode; use offline on clusters without network access")
    return parser


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
