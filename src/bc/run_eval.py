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
    
    # 3. Load training metadata
    stats = load_stats(args.stats_path, device)
    frame_stack = int(stats.get('frame_stack', args.frame_stack))
    hidden_dim = int(stats.get('hidden_dim', args.hidden_dim))
    latent_dim = int(stats.get('latent_dim', args.latent_dim))
    action_space = stats.get('action_space')

    if frame_stack != args.frame_stack:
        print(f"Using frame_stack={frame_stack} from stats file instead of CLI value {args.frame_stack}.")
    if hidden_dim != args.hidden_dim:
        print(f"Using hidden_dim={hidden_dim} from stats file instead of CLI value {args.hidden_dim}.")
    if action_space != 'swm_relative':
        print(
            "WARNING: stats file does not declare action_space='swm_relative'. "
            "Old checkpoints trained on absolute pixel actions should be retrained."
        )
    
    # 4. Initialize Latent BC Policy
    policy = LatentBCPolicy(
        latent_dim=latent_dim, 
        frame_stack=frame_stack, 
        action_dim=2, 
        hidden_dim=hidden_dim
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
        
        # Deque to hold the temporal history of latents
        latent_deque = deque(maxlen=frame_stack)
        
        while not done and step_count < args.max_steps:
            # The wrapped env returns RGB pixels with shape (96, 96, 3).
            obs_pixels = obs
            
            # 2. Convert to PyTorch tensor (H, W, C) -> (C, H, W)
            obs_tensor = torch.tensor(obs_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_tensor = resize(obs_tensor).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Extract observation features and isolate the CLS token
                encoder_outputs = lewm_encoder(obs_tensor)
                current_latent = encoder_outputs.last_hidden_state[:, 0, :] # Shape: (1, 192)
            
            # Initialization logic: if step 0, duplicate the first frame to fill the history
            if step_count == 0:
                for _ in range(frame_stack):
                    latent_deque.append(current_latent)
            else:
                latent_deque.append(current_latent)
            
            # Stack the deque elements into a single tensor: (Batch, Frame_Stack, Latent_Dim) -> (1, F, 192)
            stacked_latents = torch.stack(list(latent_deque), dim=1)
            
            with torch.no_grad():
                # Predict SWM PushT relative action in [-1, 1].
                norm_action = policy(stacked_latents)
                norm_action = torch.clamp(norm_action, -1.0, 1.0).squeeze(0)
            
            action_array = norm_action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_array)
            episode_return += float(reward)
            if args.render:
                import cv2
                # OpenCV expects BGR color format, so we reverse the RGB channels
                cv2.imshow("PushT Latent BC Evaluation", obs_pixels[..., ::-1])
                cv2.waitKey(1)

            done = terminated or truncated
            step_count += 1
                
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
    parser.add_argument("--latent_dim", type=int, default=192, help="LeWM encoder hidden size")

    args = parser.parse_args()
    evaluate(args)
