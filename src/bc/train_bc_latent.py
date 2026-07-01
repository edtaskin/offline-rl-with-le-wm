import os
import sys
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from src.bc.dataset import PushTLeWMDataset
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from dotenv import load_dotenv

load_dotenv()

le_wm_path = os.getenv("LE_WM_PATH")
if le_wm_path is None:
    raise ValueError("LE_WM_PATH environment variable not set")
if le_wm_path not in sys.path:
    sys.path.insert(0, le_wm_path)
swm = importlib.import_module("stable_worldmodel")

def train_latent_bc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load Dataset with temporal frame history
    dataset = PushTLeWMDataset(args.data_path, frame_stack=args.frame_stack)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Load the Pre-Trained LeWM Encoder (Frozen)
    print("Loading official LeWM object checkpoint...")

    # Get the path to where your conversion script just saved the checkpoint
    ckpt_path = Path(swm.data.utils.get_cache_dir(), "checkpoints", "pusht", "lewm_object.ckpt")
    print(ckpt_path)

    # Because the conversion script used `torch.save(model, out)`, 
    # the file contains the fully initialized and mapped PyTorch model object!
    lewm_model = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract just the encoder for Behavior Cloning
    lewm_encoder = lewm_model.encoder.to(device)
    lewm_encoder.eval()

    # Freeze the encoder so we only train the BC policy
    for param in lewm_encoder.parameters():
        param.requires_grad = False

    print("Successfully loaded and frozen the LeWM Encoder from the official checkpoint!")

    print("Successfully loaded and frozen the LeWM Encoder via the official API!")
    # 3. Initialize Latent BC Policy
    latent_dim = 192
    policy = LatentBCPolicy(
        latent_dim=latent_dim, 
        frame_stack=args.frame_stack,
        action_dim=2, 
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("Starting Latent-Space Behavior Cloning training loop...")

    policy.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for batch_obs_seq, batch_actions in dataloader:
            batch_obs_seq = batch_obs_seq.to(device) # Shape: (Batch, FrameStack, C, H, W)
            batch_actions = batch_actions.to(device)

            with torch.no_grad():
                # Reshape to treat frames as a larger batch: (Batch * FrameStack, C, H, W)
                b, f, c, h, w = batch_obs_seq.shape
                flat_obs = batch_obs_seq.reshape(b * f, c, h, w)
                
                # Extract the Hugging Face output object
                encoder_outputs = lewm_encoder(flat_obs) 
                
                # Extract the CLS token (the 0th token) from the last hidden state
                # last_hidden_state shape: (Batch * 5, Sequence_Length, Hidden_Dim)
                flat_latents = encoder_outputs.last_hidden_state[:, 0, :]
                
                # Reshape back to (Batch, FrameStack, LatentDim)
                stacked_latents = flat_latents.reshape(b, f, latent_dim)

            # Predict action and calculate loss
            predicted_actions = policy(stacked_latents)
            loss = criterion(predicted_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Average MSE Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_name = args.checkpoint_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(policy.state_dict(), checkpoint_name)

    torch.save(policy.state_dict(), args.checkpoint_path)
    torch.save(
        {
            **dataset.stats,
            'frame_stack': args.frame_stack,
            'hidden_dim': args.hidden_dim,
            'latent_dim': latent_dim,
            'action_dim': 2,
        },
        args.checkpoint_path.replace('.pth', '_stats.pth')
    )
    print(f"Latent BC Training Complete! Saved to: {args.checkpoint_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latent-Space Behavior Cloning (BC) Prior for LeWorldModel")
    
    # Data and Logging Paths
    parser.add_argument("--data_path", type=str, default="data/expert_trajectories/pusht_expert.npz", 
                        help="Path to the converted expert dataset .npz file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/trained_policies/pusht_latent_bc.pth", 
                        help="Path to save the final trained policy weights")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer")
    
    # Architecture and Context
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the BC MLP policy")
    parser.add_argument("--frame_stack", type=int, default=3, help="Number of LeWM latents to stack for temporal context")
    
    # Logging and Saving Intervals
    parser.add_argument("--log_interval", type=int, default=10, help="Epochs to wait before logging loss metrics")
    parser.add_argument("--save_interval", type=int, default=10, help="Save policy checkpoint every N epochs")

    args = parser.parse_args()
    
    # Execute the training loop
    train_latent_bc(args)
