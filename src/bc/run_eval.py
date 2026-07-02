import os
import sys
import importlib
import argparse
import statistics
import torch
import torchvision.transforms as T
import numpy as np
from collections import deque
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

le_wm_path = os.getenv("LE_WM_PATH")
if le_wm_path is None:
    raise ValueError("LE_WM_PATH environment variable not set")
if le_wm_path not in sys.path:
    sys.path.insert(0, le_wm_path)
swm = importlib.import_module("stable_worldmodel")
# ------------------------------------------------------------------

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.bc.dataset import (
    LEWM_IMAGE_MEAN,
    LEWM_IMAGE_NORMALIZATION,
    LEWM_IMAGE_SIZE,
    LEWM_IMAGE_STD,
)
from src.envs import make_pusht_env


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _init_wandb(args, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested with --wandb, but wandb is not installed. "
            "Install requirements.txt or run without --wandb."
        ) from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "config": _json_safe(config),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def _success_from_info(info):
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return float(info[key])
    return None


def save_evaluation_video(frames, video_path, fps=30):
    """Save RGB evaluation frames to a video file."""
    if video_path is None:
        return None
    if frames is None:
        frames = []
    elif isinstance(frames, np.ndarray):
        frames = list(frames)
    else:
        frames = list(frames)
    if len(frames) == 0:
        print(f"No evaluation frames captured; skipping video save to {video_path}.")
        return None
    if fps <= 0:
        raise ValueError("video fps must be positive")

    output_path = Path(video_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare_frame(frame, expected_shape=None):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "evaluation video frames must be RGB arrays with shape (height, width, 3)"
            )
        if expected_shape is not None and frame.shape[:2] != expected_shape:
            raise ValueError("all evaluation video frames must have the same size")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    first_frame = prepare_frame(frames[0])
    height, width = first_frame.shape[:2]

    import cv2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        writer.write(first_frame[..., ::-1])
        for frame in frames[1:]:
            writer.write(prepare_frame(frame, (height, width))[..., ::-1])
    finally:
        writer.release()

    print(f"Saved evaluation video to: {output_path}")
    return output_path


def load_stats(stats_path, device):
    stats = torch.load(stats_path, map_location=device)
    if 'action_min' in stats:
        stats['action_min'] = stats['action_min'].to(device)
    if 'action_max' in stats:
        stats['action_max'] = stats['action_max'].to(device)
    return stats


def temporal_ensemble_action(action_predictions, decay):
    """Average overlapping predictions for one target timestep."""
    if not action_predictions:
        raise ValueError("temporal ensemble needs at least one action prediction")
    stacked_actions = torch.stack(list(action_predictions), dim=0)
    if decay == 0.0 or len(action_predictions) == 1:
        return stacked_actions.mean(dim=0)

    prediction_age = torch.arange(
        len(action_predictions),
        dtype=stacked_actions.dtype,
        device=stacked_actions.device,
    )
    weights = torch.exp(-decay * prediction_age)
    weights = weights / weights.sum()
    return (stacked_actions * weights[:, None]).sum(dim=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Load the Official LeWM Encoder
    print("Loading official LeWM object checkpoint...")
    ckpt_path = Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")
    
    # weights_only=False is required for PyTorch 2.6 to unpickle the custom JEPA object
    lewm_model = torch.load(ckpt_path, map_location=device, weights_only=False)
    lewm_encoder = lewm_model.encoder.to(device)
    lewm_encoder.eval()
    for param in lewm_encoder.parameters():
        param.requires_grad = False
        
    # 2. Setup image preprocessing. Whether ImageNet normalization is applied
    # is decided from checkpoint metadata after stats are loaded below.
    resize = T.Resize(LEWM_IMAGE_SIZE, antialias=True)
    normalize = T.Normalize(mean=LEWM_IMAGE_MEAN, std=LEWM_IMAGE_STD)
    use_imagenet_normalization = False

    def encode_observation(obs_pixels):
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        obs_tensor = torch.tensor(obs_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_tensor = resize(obs_tensor)
        if use_imagenet_normalization:
            obs_tensor = normalize(obs_tensor)
        obs_tensor = obs_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            encoder_outputs = lewm_encoder(obs_tensor)
            return encoder_outputs.last_hidden_state[:, 0, :]
    
    # 3. Load training metadata
    stats = load_stats(args.stats_path, device)
    frame_stack = int(stats.get('frame_stack', args.frame_stack))
    frame_stride = int(stats.get('frame_stride', args.frame_stride))
    hidden_dim = int(stats.get('hidden_dim', args.hidden_dim))
    latent_dim = int(stats.get('latent_dim', args.latent_dim))
    action_dim = int(stats.get('action_dim', args.action_dim))
    action_chunk_size = int(stats.get('action_chunk_size', 1))
    action_space = stats.get('action_space')
    image_normalization = stats.get('image_normalization', 'legacy_div255')
    use_imagenet_normalization = image_normalization == LEWM_IMAGE_NORMALIZATION

    if frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if action_chunk_size < 1:
        raise ValueError("action_chunk_size must be at least 1")
    if args.temporal_ensemble_decay < 0.0:
        raise ValueError("temporal_ensemble_decay must be non-negative")

    if frame_stack != args.frame_stack:
        print(f"Using frame_stack={frame_stack} from stats file instead of CLI value {args.frame_stack}.")
    if frame_stride != args.frame_stride:
        print(f"Using frame_stride={frame_stride} from stats file instead of CLI value {args.frame_stride}.")
    if hidden_dim != args.hidden_dim:
        print(f"Using hidden_dim={hidden_dim} from stats file instead of CLI value {args.hidden_dim}.")
    if action_chunk_size != args.action_chunk_size:
        print(
            f"Using action_chunk_size={action_chunk_size} from stats file "
            f"instead of CLI value {args.action_chunk_size}."
        )
    if 'action_chunk_size' not in stats:
        print(
            "WARNING: stats file does not declare action_chunk_size. "
            "Assuming an old one-step BC checkpoint; retrain for 5-step chunking."
        )
    if 'frame_stride' not in stats:
        print(
            "WARNING: stats file does not declare frame_stride. "
            f"Using CLI/default value {frame_stride}."
        )
    if action_space != 'swm_relative':
        print(
            "WARNING: stats file does not declare action_space='swm_relative'. "
            "Old checkpoints trained on absolute pixel actions should be retrained."
        )
    if image_normalization != LEWM_IMAGE_NORMALIZATION:
        print(
            "WARNING: stats file does not declare image_normalization='imagenet'. "
            "This checkpoint used legacy /255-only LeWM preprocessing, which made "
            "the latent BC policy collapse to near-mean actions in diagnostics. "
            "Retrain with the current dataset preprocessing."
        )

    eval_config = {
        **vars(args),
        "device": str(device),
        "checkpoint_contract": {
            "frame_stack": frame_stack,
            "frame_stride": frame_stride,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "action_chunk_size": action_chunk_size,
            "action_space": action_space,
            "image_normalization": image_normalization,
        },
    }
    wandb_run = _init_wandb(args, eval_config)
    if wandb_run is not None:
        run_url = getattr(wandb_run, "url", None)
        print(f"Logging evaluation to wandb: {run_url or 'enabled'}")
    
    # 4. Initialize Latent BC Policy
    policy = LatentBCPolicy(
        latent_dim=latent_dim, 
        frame_stack=frame_stack, 
        action_dim=action_dim, 
        hidden_dim=hidden_dim,
        action_chunk_size=action_chunk_size,
    ).to(device)
    
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    # 5. Initialize PushT Environment. SWM PushT already consumes relative
    # [-1, 1] actions, so the clamped policy output is passed through directly.
    env = make_pusht_env()

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
    video_frames = [] if args.video_path is not None else None

    for ep in range(args.episodes):
        print(f"--- Starting Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset()
        if video_frames is not None:
            video_frames.append(np.asarray(obs).copy())
        done = False
        step_count = 0
        episode_return = 0.0
        
        # Keep enough step-level latents to select a dilated history ending at
        # the current observation.
        max_history_len = (frame_stack - 1) * frame_stride + 1
        latent_history = deque(maxlen=max_history_len)
        latent_history.append(encode_observation(obs))
        action_buffers = [deque() for _ in range(args.max_steps + action_chunk_size)]

        def build_stacked_latents():
            history = list(latent_history)
            oldest_latent = history[0]
            selected = []
            for offset in range(frame_stack - 1, -1, -1):
                history_idx = len(history) - 1 - offset * frame_stride
                selected.append(history[history_idx] if history_idx >= 0 else oldest_latent)
            return torch.stack(selected, dim=1)

        if use_temporal_ensemble:
            while not done and step_count < args.max_steps:
                stacked_latents = build_stacked_latents()

                with torch.no_grad():
                    norm_action_chunk = policy(stacked_latents)
                    norm_action_chunk = torch.clamp(norm_action_chunk, -1.0, 1.0).squeeze(0).cpu()

                for offset, predicted_action in enumerate(norm_action_chunk):
                    action_buffers[step_count + offset].append(predicted_action)

                action_tensor = temporal_ensemble_action(
                    action_buffers[step_count],
                    args.temporal_ensemble_decay,
                )
                action_array = torch.clamp(action_tensor, -1.0, 1.0).numpy()
                action_buffers[step_count].clear()

                obs, reward, terminated, truncated, info = env.step(action_array)
                episode_return += float(reward)
                if video_frames is not None:
                    video_frames.append(np.asarray(obs).copy())
                if args.render:
                    import cv2
                    # OpenCV expects BGR color format, so we reverse the RGB channels.
                    cv2.imshow("PushT Latent BC Evaluation", obs[..., ::-1])
                    cv2.waitKey(1)

                done = terminated or truncated
                step_count += 1
                if not done and step_count < args.max_steps:
                    latent_history.append(encode_observation(obs))
            print(f"Episode {ep + 1} finished after {step_count} steps. Return: {episode_return:.4f}")
            episode_returns.append(episode_return)
            episode_lengths.append(step_count)
            episode_success = _success_from_info(info)
            if episode_success is not None:
                episode_successes.append(episode_success)
            if wandb_run is not None:
                metrics = {
                    "eval/episode_return": episode_return,
                    "eval/episode_length": step_count,
                    "eval/episode": ep + 1,
                }
                if episode_success is not None:
                    metrics["eval/episode_success"] = episode_success
                wandb_run.log(metrics, step=ep + 1)
            continue

        while not done and step_count < args.max_steps:
            # Stack the deque elements into a single tensor: (Batch, Frame_Stack, Latent_Dim) -> (1, F, 192)
            stacked_latents = build_stacked_latents()
            
            with torch.no_grad():
                # Predict an open-loop chunk of SWM PushT relative actions in [-1, 1].
                norm_action_chunk = policy(stacked_latents)
                norm_action_chunk = torch.clamp(norm_action_chunk, -1.0, 1.0).squeeze(0)
            
            action_chunk = norm_action_chunk.cpu().numpy()
            for action_array in action_chunk:
                obs, reward, terminated, truncated, info = env.step(action_array)
                episode_return += float(reward)
                if video_frames is not None:
                    video_frames.append(np.asarray(obs).copy())
                if args.render:
                    import cv2
                    # OpenCV expects BGR color format, so we reverse the RGB channels.
                    cv2.imshow("PushT Latent BC Evaluation", obs[..., ::-1])
                    cv2.waitKey(1)

                done = terminated or truncated
                step_count += 1
                if done or step_count >= args.max_steps:
                    break
                latent_history.append(encode_observation(obs))
                
        print(f"Episode {ep + 1} finished after {step_count} steps. Return: {episode_return:.4f}")
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        episode_success = _success_from_info(info)
        if episode_success is not None:
            episode_successes.append(episode_success)
        if wandb_run is not None:
            metrics = {
                "eval/episode_return": episode_return,
                "eval/episode_length": step_count,
                "eval/episode": ep + 1,
            }
            if episode_success is not None:
                metrics["eval/episode_success"] = episode_success
            wandb_run.log(metrics, step=ep + 1)
        
    env.close()
    save_evaluation_video(video_frames, args.video_path, args.video_fps)

    if episode_returns:
        summary = {
            "eval/return_mean": statistics.fmean(episode_returns),
            "eval/return_min": min(episode_returns),
            "eval/return_max": max(episode_returns),
            "eval/length_mean": statistics.fmean(episode_lengths),
        }
        if len(episode_returns) > 1:
            summary["eval/return_std"] = statistics.pstdev(episode_returns)
        else:
            summary["eval/return_std"] = 0.0
        if episode_successes:
            summary["eval/success_rate"] = statistics.fmean(episode_successes)
        print(
            "Evaluation summary: "
            f"return_mean={summary['eval/return_mean']:.4f}, "
            f"return_std={summary['eval/return_std']:.4f}, "
            f"length_mean={summary['eval/length_mean']:.2f}"
        )
        if "eval/success_rate" in summary:
            print(f"Success rate: {summary['eval/success_rate']:.4f}")
        if wandb_run is not None:
            for key, value in summary.items():
                wandb_run.summary[key] = value
            wandb_run.log(summary, step=args.episodes)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent BC Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved policy weights (.pth)")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to the saved _stats.pth file")
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--render", action='store_true', help="Render the environment visually")
    parser.add_argument("--video_path", type=str, default=None, help="Optional path to save evaluation video")
    parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for saved evaluation video")
    
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the BC MLP")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of frames to stack (must match training)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Environment steps between stacked history frames")
    parser.add_argument("--latent_dim", type=int, default=192, help="LeWM encoder hidden size")
    parser.add_argument("--action_dim", type=int, default=2, help="Per-step PushT action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=5, help="Number of future actions predicted from one observation")
    parser.add_argument(
        "--temporal_ensemble",
        action="store_true",
        help="Query every step and average overlapping predicted action chunks, following ACT inference.",
    )
    parser.add_argument(
        "--temporal_ensemble_decay",
        type=float,
        default=0.01,
        help="Exponential decay for temporal ensembling weights; 0.0 gives a uniform average.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases evaluation tracking")
    parser.add_argument("--wandb_project", type=str, default="offline-rl-lewm", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_group", type=str, default="pusht-latent-bc-eval", help="Weights & Biases run group")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional Weights & Biases tags")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Weights & Biases mode; use offline on clusters without network access",
    )

    args = parser.parse_args()
    evaluate(args)
