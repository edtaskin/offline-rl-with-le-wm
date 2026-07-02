import os
import sys
import importlib
import argparse
import torch
import torchvision.transforms as T
from collections import deque
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

le_wm_path = os.getenv("LE_WM_PATH")
if le_wm_path is None:
    raise ValueError("LE_WM_PATH environment variable not set")
if le_wm_path not in sys.path:
    sys.path.insert(0, le_wm_path)
swm = importlib.import_module("stable_worldmodel")
# ------------------------------------------------------------------

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.envs import make_pusht_env

def load_stats(stats_path, device):
    stats = torch.load(stats_path, map_location=device)
    if 'action_min' in stats:
        stats['action_min'] = stats['action_min'].to(device)
    if 'action_max' in stats:
        stats['action_max'] = stats['action_max'].to(device)
    return stats

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
        
    # 2. Setup Image Preprocessing (Resize to 224x224 to match ViT-Tiny)
    resize = T.Resize((224, 224), antialias=True)

    def encode_observation(obs_pixels):
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        obs_tensor = torch.tensor(obs_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_tensor = resize(obs_tensor).unsqueeze(0).to(device)
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

    if frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if action_chunk_size < 1:
        raise ValueError("action_chunk_size must be at least 1")

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
    
    for ep in range(args.episodes):
        print(f"--- Starting Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_return = 0.0
        
        # Keep enough step-level latents to select a dilated history ending at
        # the current observation.
        max_history_len = (frame_stack - 1) * frame_stride + 1
        latent_history = deque(maxlen=max_history_len)
        latent_history.append(encode_observation(obs))

        def build_stacked_latents():
            history = list(latent_history)
            oldest_latent = history[0]
            selected = []
            for offset in range(frame_stack - 1, -1, -1):
                history_idx = len(history) - 1 - offset * frame_stride
                selected.append(history[history_idx] if history_idx >= 0 else oldest_latent)
            return torch.stack(selected, dim=1)
        
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
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent BC Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved policy weights (.pth)")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to the saved _stats.pth file")
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--render", action='store_true', help="Render the environment visually")
    
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the BC MLP")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of frames to stack (must match training)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Environment steps between stacked history frames")
    parser.add_argument("--latent_dim", type=int, default=192, help="LeWM encoder hidden size")
    parser.add_argument("--action_dim", type=int, default=2, help="Per-step PushT action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=5, help="Number of future actions predicted from one observation")

    args = parser.parse_args()
    evaluate(args)
