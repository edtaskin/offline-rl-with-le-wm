"""Ablation-local latent BC-head training."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.bc.dataset import PushTLatentDataset
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy

from .cache import cache_path


DEFAULT_CONTRACT = {
    "frame_stack": 3,
    "frame_stride": 5,
    "action_chunk_size": 5,
    "hidden_dim": 256,
    "action_dim": 2,
}


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def model_paths(output_root, encoder_name, condition_name, seed):
    directory = (
        Path(output_root)
        / "models"
        / encoder_name
        / condition_name
        / f"seed_{int(seed)}"
    )
    return directory / "policy.pth", directory / "metadata.json"


def _existing_training_matches(
    metadata_path,
    *,
    data_path,
    encoder_name,
    condition_name,
    seed,
    contract,
    epochs,
    batch_size,
    lr,
    num_workers,
    cache_metadata,
):
    try:
        metadata = json.loads(Path(metadata_path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    identity_matches = (
        metadata.get("encoder") == encoder_name
        and metadata.get("train_condition") == condition_name
        and metadata.get("seed") == int(seed)
        and metadata.get("data_path") == str(Path(data_path).resolve())
    )
    saved_contract = metadata.get("contract", {})
    contract_matches = all(
        saved_contract.get(key) == value for key, value in contract.items()
    )
    saved_training = metadata.get("training", {})
    training_matches = (
        saved_training.get("epochs") == int(epochs)
        and saved_training.get("batch_size") == int(batch_size)
        and saved_training.get("learning_rate") == float(lr)
        and saved_training.get("num_workers") == int(num_workers)
    )
    return (
        identity_matches
        and contract_matches
        and training_matches
        and metadata.get("cache_metadata") == cache_metadata
    )


def train_policy(
    *,
    data_path,
    output_root,
    encoder_name,
    condition_name,
    seed,
    device="auto",
    epochs=100,
    batch_size=64,
    lr=1e-3,
    num_workers=0,
    contract=None,
    force=False,
):
    if epochs < 1 or batch_size < 1 or lr <= 0 or num_workers < 0:
        raise ValueError(
            "epochs/batch_size/lr must be positive and num_workers non-negative"
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    checkpoint_path, metadata_path = model_paths(
        output_root, encoder_name, condition_name, seed
    )
    selected_contract = {**DEFAULT_CONTRACT, **(contract or {})}
    selected_cache = cache_path(output_root, encoder_name, condition_name)
    if not selected_cache.exists():
        raise FileNotFoundError(f"latent cache does not exist: {selected_cache}")
    cache_payload = torch.load(selected_cache, map_location="cpu", weights_only=False)
    current_cache_metadata = json.loads(json.dumps(cache_payload.get("metadata", {})))
    del cache_payload
    if checkpoint_path.exists() and metadata_path.exists() and not force:
        if not _existing_training_matches(
            metadata_path,
            data_path=data_path,
            encoder_name=encoder_name,
            condition_name=condition_name,
            seed=seed,
            contract=selected_contract,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            cache_metadata=current_cache_metadata,
        ):
            raise RuntimeError(
                f"existing trained head is incompatible with the requested settings: "
                f"{checkpoint_path}; use --force to replace it"
            )
        print(f"Using existing trained head: {checkpoint_path}")
        return checkpoint_path

    seed_everything(seed)
    dataset = PushTLatentDataset(
        data_path,
        selected_cache,
        frame_stack=selected_contract["frame_stack"],
        frame_stride=selected_contract["frame_stride"],
        action_chunk_size=selected_contract["action_chunk_size"],
    )
    latent_dim = int(dataset.stats["latent_dim"])
    selected_contract["latent_dim"] = latent_dim
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if not len(loader):
        raise ValueError("dataset is smaller than one drop-last training batch")
    policy = LatentBCPolicy(
        latent_dim=latent_dim,
        frame_stack=selected_contract["frame_stack"],
        action_dim=selected_contract["action_dim"],
        hidden_dim=selected_contract["hidden_dim"],
        action_chunk_size=selected_contract["action_chunk_size"],
    ).to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_epoch = 0
    policy.train()
    for epoch in range(epochs):
        running = 0.0
        for latents, actions in loader:
            prediction = policy(latents.to(resolved_device))
            loss = criterion(prediction, actions.to(resolved_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        average = running / len(loader)
        history.append(average)
        if average < best_loss:
            best_loss = average
            best_epoch = epoch + 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"  {encoder_name}/{condition_name}/seed={seed} "
                f"epoch={epoch + 1}/{epochs} loss={average:.6f}"
            )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_path)
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder_name,
        "train_condition": condition_name,
        "seed": int(seed),
        "data_path": str(Path(data_path).resolve()),
        "cache_path": str(selected_cache.resolve()),
        "cache_metadata": current_cache_metadata,
        "contract": selected_contract,
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "num_workers": int(num_workers),
            "deterministic": True,
            "loss_history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return checkpoint_path


def load_policy(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_name("metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"model metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    contract = metadata["contract"]
    policy = LatentBCPolicy(
        latent_dim=int(contract["latent_dim"]),
        frame_stack=int(contract["frame_stack"]),
        action_dim=int(contract["action_dim"]),
        hidden_dim=int(contract["hidden_dim"]),
        action_chunk_size=int(contract["action_chunk_size"]),
    ).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy, metadata

