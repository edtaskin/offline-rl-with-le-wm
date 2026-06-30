import os
import sys
import argparse
import torch
import numpy as np
import gymnasium as gym
import torchvision.transforms as T
from collections import deque
from pathlib import Path
import stable_worldmodel as swm
from dotenv import load_dotenv

load_dotenv()

le_wm_path = os.getenv("LE_WM_PATH")
if le_wm_path is None:
    raise ValueError("LE_WM_PATH environment variable not set")
if le_wm_path not in sys.path:
    sys.path.append(le_wm_path)
# ------------------------------------------------------------------

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy

def load_stats(stats_path, device):
    stats = torch.load(stats_path, map_location=device)
    # The PushTLeWMDataset only saves action statistics
    return stats['action_min'].to(device), stats['action_max'].to(device)

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
    
    # 3. Load Action Normalization Stats
    a_min, a_max = load_stats(args.stats_path, device)
    
    # 4. Initialize Latent BC Policy
    latent_dim = 192 # Must match the LeWM ViT-Tiny hidden size
    policy = LatentBCPolicy(
        latent_dim=latent_dim, 
        frame_stack=args.frame_stack, 
        action_dim=2, 
        hidden_dim=args.hidden_dim
    ).to(device)
    
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    # 5. Initialize PushT Environment
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    
    for ep in range(args.episodes):
        print(f"--- Starting Episode {ep + 1}/{args.episodes} ---")
        obs, info = env.reset()
        done = False
        step_count = 0
        
        # Deque to hold the temporal history of latents
        latent_deque = deque(maxlen=args.frame_stack)
        
        while not done and step_count < args.max_steps:
            # The env returns 'pixels' with shape (96, 96, 3). Convert to (1, 3, 224, 224)
            obs_pixels = env.render() 
            
            # 2. Convert to PyTorch tensor (H, W, C) -> (C, H, W)
            obs_tensor = torch.tensor(obs_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_tensor = resize(obs_tensor).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Extract observation features and isolate the CLS token
                encoder_outputs = lewm_encoder(obs_tensor)
                current_latent = encoder_outputs.last_hidden_state[:, 0, :] # Shape: (1, 192)
            
            # Initialization logic: if step 0, duplicate the first frame to fill the history
            if step_count == 0:
                for _ in range(args.frame_stack):
                    latent_deque.append(current_latent)
            else:
                latent_deque.append(current_latent)
            
            # Stack the deque elements into a single tensor: (Batch, Frame_Stack, Latent_Dim) -> (1, F, 192)
            stacked_latents = torch.stack(list(latent_deque), dim=1)
            
            with torch.no_grad():
                # Predict normalized action
                norm_action = policy(stacked_latents)
                norm_action = torch.clamp(norm_action, -1.0, 1.0).squeeze(0)
            
            # Un-normalize the action for the environment
            """ raw_action = ((norm_action + 1) / 2) * (a_max - a_min) + a_min
            action_array = raw_action.cpu().numpy()
            action_array = np.clip(np.squeeze(action_array), env.action_space.low, env.action_space.high) """

            action_array = norm_action.cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action_array)
            if args.render:
                import cv2
                # OpenCV expects BGR color format, so we reverse the RGB channels
                cv2.imshow("PushT Latent BC Evaluation", obs_pixels[..., ::-1])
                cv2.waitKey(1)

            done = terminated or truncated
            step_count += 1
                
        print(f"Episode {ep + 1} finished after {step_count} steps. Reward: {reward}")
        
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

    args = parser.parse_args()
    evaluate(args)