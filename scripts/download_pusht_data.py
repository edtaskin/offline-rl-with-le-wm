import os
import urllib.request
import zipfile
import zarr
import numpy as np

def obtain_expert_trajectories():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "expert_trajectories")
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "pusht.zip")
    zarr_dir = os.path.join(data_dir, "pusht", "pusht_cchi_v7_replay.zarr")
    npz_path = os.path.join(data_dir, "pusht_expert.npz")
    
    url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
    
    if not os.path.exists(zip_path) and not os.path.exists(npz_path):
        print(f"Downloading expert PushT dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    if os.path.exists(zarr_dir) and not os.path.exists(npz_path):
        dataset_root = zarr.open(zarr_dir, 'r')
        
        # Existing extractions
        states = dataset_root['data']['state'][:]
        actions = dataset_root['data']['action'][:]
        
        images = dataset_root['data']['img'][:] 
        
        episode_ends = dataset_root['meta']['episode_ends'][:]
        
        # Save the images array alongside the rest of your data
        np.savez(npz_path, 
                 states=states, 
                 actions=actions, 
                 images=images, 
                 episode_ends=episode_ends)
        print(f"Successfully saved converted NPZ dataset with images to: {npz_path}")

if __name__ == "__main__":
    obtain_expert_trajectories()