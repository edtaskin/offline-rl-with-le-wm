"""Chunk-level latent PPO trained *inside* the LeWM world model (imagination).

Instead of stepping ``swm/PushT-v1``, rollouts happen in LeWM latent space:
one PPO transition = one predictor step = ``wm_frameskip`` env steps, so the
agent contract (frame_stack=3, frame_stride=5, action_chunk_size=5) maps 1:1
onto the world model (history=3, frameskip=5). Per imagined step:

1. the agent picks an action chunk ``[k, 2]`` from its dilated stack of raw
   CLS latents (same interface as :mod:`src.ppo.ppo`, so BC init and
   ``src/ppo/evaluate.py`` keep working);
2. the chunk is converted to the dataset action space, z-scored with dataset
   stats, and fed to ``wm.action_encoder`` + ``wm.predict`` to imagine the next
   *projected* latent (the space LeWM dynamics run in);
3. the latent image decoder (``scripts/decoder/train_decoder_pusht.py``)
   renders the imagined latent to an RGB frame, which is re-encoded to the raw
   CLS latent the policy consumes -- the decoder bridges LeWM's projected
   latent space back to the BC policy's input space;
4. reward and success come from frozen probes/classifiers on the imagined
   projected latent: ``sparse`` = 1.0 on success, ``dense`` = sparse success
   plus learned time-to-success classifier shaping, and ``pose_dense`` = the
   legacy block-pose distance reward.

Episodes start from ground-truth context windows sampled from the expert h5
dataset (respecting ``block_start_near_goal``/``block_start_radius`` and
skipping already-solved frames) and truncate after ``dream_episode_steps``
predictor steps with a value bootstrap, exactly like time-limit truncation in
the real-env trainer. ``--record-real-eval`` runs the inherited held-out eval
in the *real* env at the dream-eval cadence, measuring dream-to-real transfer
while leaving checkpoint selection based on ``--selection``.

Requires (defaults match the decoder/probe scripts; decoder and probes are
also published under hf.co/offline-rl-with-le-wm):

* LeWM weights:      ``le-wm/models/checkpoints/hf_pusht/weights.pt``
  (falls back to the converted ``lewm_object.ckpt`` in the swm cache)
* expert dataset:    ``le-wm/models/datasets/pusht_expert_train.h5``
* image decoder:     ``models/latent_decoder/pusht_lewm/decoder_lewm_pusht.pt``
* state probes under ``models/probes/pusht_lewm/``: the ``objective_met``
  classifier (success + sparse reward) and/or the ``block_rel_objective``
  regression probe (distances, required for ``pose_dense`` reward)
* dense reward:      ``hf://offline-rl-with-le-wm/dense_reward_classifier/``
  ``dense_reward_classifier.pt`` (downloaded automatically for ``dense`` reward)

Runnable either way::

    python -m src.ppo.train_lewm --smoke
    python src/ppo/train_lewm.py --fixed_target --eval_interval 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from src.ppo.agent import build_latent_agent
from src.ppo.config import LatentConfig
from src.ppo.dense_reward import DenseRewardShaper
from src.ppo.env import LatentHistory
from src.ppo.lewm_encoder import LeWMLatentEncoder
from src.ppo.ppo import LatentPPOTrainer, _configure_logging, build_bc_ref_policy, logger
from src.ppo.train import (
    _NULLABLE_STR_FIELDS,
    SMOKE_OVERRIDES,
    _flag_names,
    _push_checkpoint_to_hf,
    _stats_contract,
)
from src.ppo.utils import RewardNormalizer, get_device, set_seed

# The dataset stores absolute pointer targets; the env/BC/PPO action space is
# SWM-relative: env target = agent_xy + action * PUSHT_ACTION_SCALE.
from src.bc.dataset import PUSHT_ACTION_SCALE
from src.envs import PUSHT_FIXED_TARGET_POSE

PUSHT_COORD_LOW = 0.0
PUSHT_COORD_HIGH = 512.0

DREAM_SMOKE_OVERRIDES = {**SMOKE_OVERRIDES, "dream_episode_steps": 6}

DENSE_REWARD_CHECKPOINT_HF = (
    "hf://offline-rl-with-le-wm/dense_reward_classifier/dense_reward_classifier.pt"
)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass
class DreamConfig(LatentConfig):
    """:class:`LatentConfig` plus the world-model / imagination knobs."""

    exp_name: str = "latent_ppo_pusht_lewm_dream"

    # ----- world model + offline artifacts -----
    wm_checkpoint: str = "hf_pusht/weights.pt"
    wm_cache_dir: str = "le-wm/models"
    decoder_checkpoint: str = "models/latent_decoder/pusht_lewm/decoder_lewm_pusht.pt"
    probe_dir: str = "models/probes/pusht_lewm"
    dataset_path: str = "le-wm/models/datasets/pusht_expert_train.h5"

    # How an imagined (projected) latent is returned to the policy's raw-CLS
    # input space. "decoder" renders a frame and re-encodes it with the ViT;
    # "deprojector" maps latent to latent directly (see scripts/deprojector/).
    # The decoder is still loaded on demand for frame diagnostics either way.
    bridge: str = "decoder"
    deprojector_checkpoint: str = "models/deprojector/pusht_lewm/deprojector.pt"

    # One predictor step covers this many env steps; must equal both
    # frame_stride and action_chunk_size so agent and WM tick together.
    wm_frameskip: int = 5

    # Imagined episode length in predictor steps (x wm_frameskip env steps).
    # Kept short: compounding prediction error grows with horizon (the decoder
    # diagnostics in scripts/decoder cover ~10 steps); success ends episodes early.
    dream_episode_steps: int = 20

    # ----- interaction-free checkpoint selection -----
    # Which metric ranks best.pt: "dream" = imagined success on held-out expert
    # anchors (zero env steps -- the RQ1 headline); "real" = held-out success in
    # the actual simulator (costs interaction, kept for the contrast); "rolling"
    # = the training-window average, the LatentConfig default.
    selection: str = "dream"
    # Imagined held-out eval every N iterations (0 = off).
    dream_eval_interval: int = 10
    dream_eval_episodes: int = 96
    dream_eval_seed: int = 12345
    # Imagined eval horizon in predictor steps; 0 = reuse dream_episode_steps.
    dream_eval_steps: int = 0
    # Record the real held-out metric alongside dream validation. When the
    # inherited eval_interval is 0, this synchronizes it to dream_eval_interval
    # so selection_log.jsonl contains paired dream/real measurements. Real eval
    # is diagnostic unless selection="real", but its env steps are always charged.
    record_real_eval: bool = False
    # Fraction of expert *episodes* reserved for evaluation anchors. Held out at
    # episode granularity, not frame granularity, so a validation start state is
    # never a few frames away from a training start state.
    dream_val_fraction: float = 0.1
    dream_val_split_seed: int = 0

    # Success thresholds, mirroring PushTAlignSampledGoalToFixedTargetWrapper.
    success_pos_tol: float = 20.0
    success_angle_tol: float = float(np.pi / 9)

    # Learned dense reward from scripts/probes/train_dense_reward_pusht.py.
    # Active when reward_mode == "dense". The classifier is evaluated on
    # projected LeWM dynamics latents, not raw CLS policy latents.
    dense_reward_checkpoint: str = DENSE_REWARD_CHECKPOINT_HF
    dense_reward_coef: float = 0.05
    dense_reward_weights: str = "1 0.75 0.4 0.1"
    dense_reward_clip: float = 0.5
    dense_reward_mode: str = "potential"  # "potential" | "delta" | "score"
    dense_reward_positive_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.record_real_eval and self.eval_interval <= 0:
            self.eval_interval = self.dream_eval_interval
        if self.record_real_eval and self.eval_interval <= 0:
            raise ValueError(
                "record_real_eval requires eval_interval > 0 or "
                "dream_eval_interval > 0"
            )
        if self.frame_stride != self.wm_frameskip:
            raise ValueError(
                f"frame_stride ({self.frame_stride}) must equal wm_frameskip "
                f"({self.wm_frameskip}): one dilated-stack hop = one predictor step"
            )
        if self.action_chunk_size != self.wm_frameskip:
            raise ValueError(
                f"action_chunk_size ({self.action_chunk_size}) must equal "
                f"wm_frameskip ({self.wm_frameskip}): one chunk = one predictor step"
            )
        if self.action_dim != 2:
            raise ValueError("PushT dream rollouts require action_dim == 2")
        if self.bridge not in ("decoder", "deprojector"):
            raise ValueError(f"bridge must be 'decoder' or 'deprojector', got {self.bridge!r}")
        if self.selection not in ("dream", "real", "rolling"):
            raise ValueError(
                f"selection must be 'dream', 'real' or 'rolling', got {self.selection!r}"
            )
        if self.selection == "dream" and self.dream_eval_interval <= 0:
            raise ValueError("selection='dream' requires dream_eval_interval > 0")
        if self.selection == "real" and self.eval_interval <= 0:
            raise ValueError("selection='real' requires eval_interval > 0")
        if not 0.0 < self.dream_val_fraction < 1.0:
            raise ValueError("dream_val_fraction must be in (0, 1)")
        if self.reward_mode not in ("sparse", "dense", "pose_dense"):
            raise ValueError("dream reward_mode must be 'sparse', 'dense' or 'pose_dense'")
        if self.dense_reward_mode not in ("potential", "delta", "score"):
            raise ValueError("dense_reward_mode must be 'potential', 'delta' or 'score'")
        if self.dense_reward_coef < 0.0:
            raise ValueError("dense_reward_coef must be non-negative")
        if self.dense_reward_clip < 0.0:
            raise ValueError("dense_reward_clip must be non-negative")
        if self.reward_mode == "dense":
            if not self.dense_reward_checkpoint:
                raise ValueError(
                    "reward_mode='dense' requires a dense_reward_checkpoint"
                )
            if self.dense_reward_coef <= 0.0:
                raise ValueError("reward_mode='dense' requires dense_reward_coef > 0")


class _StateProbe(nn.Module):
    """Frozen MLP probe: projected LeWM latent ``[B, 192]`` -> state feature.

    Handles both payload flavors saved by ``train_probes_pusht.py``: regression
    probes (de-standardized with ``y_mean``/``y_std``) and binary classifiers
    (sigmoid probability plus the F1-selected decision ``threshold``).
    """

    def __init__(self, payload: dict):
        super().__init__()
        from scripts.probes.train_probes_pusht import ProbeMLP

        model = ProbeMLP(
            payload["input_dim"],
            payload["output_dim"],
            payload["hidden_dim"],
            payload["depth"],
        )
        model.load_state_dict(payload["model"])
        self.model = model.eval().requires_grad_(False)
        self.is_classifier = payload.get("task") == "binary_classification"
        self.threshold = float(payload.get("threshold", 0.5))
        out_dim = int(payload["output_dim"])
        for name, default in (
            ("x_mean", None),
            ("x_std", None),
            ("y_mean", np.zeros((1, out_dim))),
            ("y_std", np.ones((1, out_dim))),
        ):
            value = payload.get(name, default)
            self.register_buffer(
                name, torch.as_tensor(np.asarray(value), dtype=torch.float32)
            )

    @classmethod
    def find(
        cls, probe_dir: Path, candidates: tuple[str, ...], device: torch.device
    ) -> "_StateProbe | None":
        """Load the first existing probe among ``candidates`` (relative paths)."""
        for rel in candidates:
            path = probe_dir / rel
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                logger.info("Loaded state probe %s", path)
                return cls(payload).to(device)
        return None

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pred = self.model((z - self.x_mean) / self.x_std)
        if self.is_classifier:
            return torch.sigmoid(pred)
        return pred * self.y_std + self.y_mean


def _load_world_model(cfg: DreamConfig, device: torch.device) -> nn.Module:
    """Load the full (frozen) LeWM world model for this run's config."""
    from src.representations.lewm import load_lewm_world_model

    return load_lewm_world_model(
        cfg.wm_checkpoint,
        cache_dir=repo_path(cfg.wm_cache_dir),
        device=device,
        logger=logger,
    )


def _resolve_decoder_reference(reference: str | Path) -> Path:
    """Accept either a Hub reference or a repo-relative path.

    Runs configure ``decoder_checkpoint`` both ways, and ``repo_path`` would
    silently turn ``hf://owner/repo/file.pt`` into a bogus relative path.
    """
    from src.utils.hf_hub import parse_hf_artifact_reference, resolve_artifact

    if parse_hf_artifact_reference(str(reference)) is not None:
        return resolve_artifact(str(reference))
    return repo_path(reference)


def _load_decoder(path: Path, device: torch.device) -> nn.Module:
    """Load the frozen latent image decoder (same payload as decode_rollouts)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing decoder checkpoint: {path}. Train one with "
            "scripts/decoder/train_decoder_pusht.py or download it from "
            "hf.co/offline-rl-with-le-wm/decoder_lewm_pusht."
        )
    from scripts.decoder.train_decoder_pusht import LatentImageDecoder

    payload = torch.load(path, map_location="cpu", weights_only=False)
    decoder = LatentImageDecoder(**payload["config"])
    decoder.load_state_dict(payload["decoder"])
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder


class LeWMDreamWorld:
    """Batched imagination "vector env" over the frozen LeWM world model.

    Holds, per parallel dream env, the predictor's latent/action history plus
    an agent-position anchor used to convert SWM-relative action chunks into
    the absolute dataset action space the WM was trained on. ``step`` advances
    every env one predictor step; finished envs are re-seeded by the trainer
    via :meth:`reset_env` from fresh expert-dataset context windows.
    """

    def __init__(self, cfg: DreamConfig, device: torch.device):
        # The expert h5 stores pixels Blosc/Zstd-compressed. Without the
        # hdf5plugin filters registered, the first pixel read fails with a bare
        # "can't open directory /usr/local/lib/plugin" OSError.
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import h5py

        self.cfg = cfg
        self.device = device

        self.wm = _load_world_model(cfg, device)
        self.history_size = int(getattr(self.wm.predictor, "num_frames", 3))

        # The decoder is loaded on first use, so a de-projector run needs no
        # decoder checkpoint at all -- but frame diagnostics
        # (scripts/rq2/dream_success_gallery.py) can still ask for one.
        self._decoder = None
        self._deprojector = None
        self.capture_frames = cfg.bridge == "decoder"
        if cfg.bridge == "deprojector":
            from src.representations.deprojector import load_deprojector

            self._deprojector = load_deprojector(repo_path(cfg.deprojector_checkpoint), device)
            logger.info("Dream bridge: de-projector %s", cfg.deprojector_checkpoint)
        else:
            logger.info("Dream bridge: decoder %s", cfg.decoder_checkpoint)
        # Shared frozen ViT: raw CLS latents for the agent (and real-env eval);
        # the WM's projector lifts the same CLS into the dynamics latent space.
        # Pass the device explicitly: LeWMEncoder defaults to CPU and moves the
        # module it wraps, which would strand the WM's ViT off-device.
        self.cls_encoder = LeWMLatentEncoder(
            self.wm.encoder, device=device, latent_dim=cfg.latent_dim
        )

        dataset_path = repo_path(cfg.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Expert dataset not found: {dataset_path}. The dream trainer samples "
                "episode-start context windows from it."
            )
        self._h5 = h5py.File(dataset_path, "r")
        self._rng = np.random.default_rng(cfg.seed)
        self.context_steps = max(cfg.frame_stack, self.history_size)

        action = self._h5["action"][:]
        mean = action.mean(axis=0).astype(np.float32)
        std = action.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.action_mean = torch.as_tensor(mean, device=device)
        self.action_std = torch.as_tensor(std, device=device)
        # The diffusion-policy PushT data stores absolute pointer targets
        # (coordinate-scale mean); tolerate a relative-action re-export too.
        self.absolute_actions = bool(np.abs(mean).max() > 10.0)
        logger.info(
            "Dream world | history %d | context %d frames | dataset actions %s "
            "(mean %s)",
            self.history_size,
            self.context_steps,
            "absolute" if self.absolute_actions else "relative",
            np.round(mean, 2).tolist(),
        )

        # Reward/success probes. Success (and the sparse reward) prefers the
        # objective_met classifier -- trained on exactly the fixed-target
        # success condition, with an F1-calibrated threshold; the
        # block_rel_objective regression probe supplies pose distances for the
        # legacy pose_dense reward and is used for success when no classifier
        # exists.
        probe_dir = repo_path(cfg.probe_dir)
        self.success_probe = _StateProbe.find(
            probe_dir,
            ("objective_met/mlp_probe.pt", "is_objective_met_probe_baseline.pt"),
            device,
        )
        self.pose_probe = _StateProbe.find(
            probe_dir, ("block_rel_objective/mlp_probe.pt",), device
        )
        if self.success_probe is None and self.pose_probe is None:
            raise FileNotFoundError(
                f"No usable probe under {probe_dir}: need objective_met (success/"
                "sparse reward) and/or block_rel_objective (pose-distance reward). "
                "Train them with scripts/probes/train_probes_pusht.py or download "
                "from hf.co/offline-rl-with-le-wm/probes."
            )
        if cfg.reward_mode == "pose_dense" and self.pose_probe is None:
            raise FileNotFoundError(
                f"reward_mode='pose_dense' needs the block_rel_objective probe under "
                f"{probe_dir} for pose-distance rewards; use --reward-mode sparse "
                "or --reward-mode dense with the learned dense reward classifier."
            )
        self.engagement_probe = None
        if cfg.agent_block_coef > 0.0:
            self.engagement_probe = _StateProbe.find(
                probe_dir, ("block_rel_agent/mlp_probe.pt",), device
            )
            if self.engagement_probe is None:
                raise FileNotFoundError(
                    f"agent_block_coef > 0 needs the block_rel_agent probe under {probe_dir}"
                )
        self.pos_probe = _StateProbe.find(probe_dir, ("agent_pos/mlp_probe.pt",), device)
        if self.absolute_actions and self.pos_probe is None:
            logger.warning(
                "agent_pos probe not found under %s; dream anchors integrate "
                "commanded targets instead of reading the WM's imagined agent position.",
                probe_dir,
            )

        self.dense_reward = None
        self._dense_reward_prev_score = torch.zeros(cfg.num_envs, device=device)
        self._last_dense_reward = np.zeros(cfg.num_envs, dtype=np.float32)
        self._last_dense_score = np.zeros(cfg.num_envs, dtype=np.float32)
        self._last_dense_probs = np.zeros((cfg.num_envs, 0), dtype=np.float32)
        if cfg.reward_mode == "dense":
            self.dense_reward = DenseRewardShaper(
                cfg.dense_reward_checkpoint,
                weights=cfg.dense_reward_weights,
                scale=cfg.dense_reward_coef,
                clip=cfg.dense_reward_clip,
                device=device,
            )
            if self.dense_reward.x_mean.shape[-1] != cfg.latent_dim:
                raise ValueError(
                    f"dense reward checkpoint expects latent dim "
                    f"{self.dense_reward.x_mean.shape[-1]}, cfg.latent_dim={cfg.latent_dim}"
                )
            if self.dense_reward.frameskip != cfg.wm_frameskip:
                raise ValueError(
                    f"dense reward checkpoint frameskip={self.dense_reward.frameskip}, "
                    f"cfg.wm_frameskip={cfg.wm_frameskip}"
                )
            self._last_dense_probs = np.zeros(
                (cfg.num_envs, len(self.dense_reward.horizons)), dtype=np.float32
            )
            logger.info(
                "Loaded dense reward classifier %s | horizons %s wm steps | "
                "mode %s | coef %.4g | clip %.4g | weights %s",
                cfg.dense_reward_checkpoint,
                self.dense_reward.horizons,
                cfg.dense_reward_mode,
                cfg.dense_reward_coef,
                cfg.dense_reward_clip,
                self.dense_reward.weights.detach().cpu().numpy().round(4).tolist(),
            )
        elif cfg.dense_reward_checkpoint or cfg.dense_reward_coef > 0.0:
            logger.warning(
                "Ignoring dense reward classifier options because reward_mode=%r; "
                "use --reward-mode dense to train with the learned dense reward.",
                cfg.reward_mode,
            )

        anchors = self._valid_context_anchors()
        self._anchors, self._val_anchors = self._split_anchors(anchors)
        logger.info(
            "Dream world | %d train / %d held-out episode-start anchors "
            "(%d expert episodes reserved)",
            len(self._anchors),
            len(self._val_anchors),
            self._num_val_episodes,
        )

        # Episode horizon in predictor steps; swapped by :meth:`eval_mode`.
        self._episode_steps = int(cfg.dream_episode_steps)

        # Per-env predictor state: aligned deques of projected latents and the
        # (normalized, flattened) action blocks taken at each of them.
        n = cfg.num_envs
        self._emb_hist = [deque(maxlen=self.history_size) for _ in range(n)]
        self._act_hist = [deque(maxlen=self.history_size) for _ in range(n)]
        self._agent_pos = torch.zeros((n, 2), device=device)
        self._steps = np.zeros(n, dtype=np.int64)
        # Inspection-only state, written by reset_env/step and never read by
        # training: the dataset provenance of each env's current dream episode,
        # and the most recent decoded frames / latents / probe probabilities.
        self.last_anchor: list[dict | None] = [None] * n
        self.last_frames: torch.Tensor | None = None
        self.last_latent: torch.Tensor | None = None
        self.last_success_prob: torch.Tensor | None = None

        # Dense chunk reward ~ within-chunk discounted sum of the per-step
        # reward, matching the real trainer's sum_j gamma**j r_j scale.
        k = cfg.action_chunk_size
        self._dense_chunk_scale = float((1.0 - cfg.gamma**k) / (1.0 - cfg.gamma))

    # -------------------------------------------------------------- sampling
    def _valid_context_anchors(self) -> np.ndarray:
        """``[N, 2]`` (episode, global anchor row) pairs usable as episode starts.

        The anchor is the *last* context frame (the imagined "now"). It must be
        deep enough into its episode to supply the full ground-truth context
        window, not already satisfy the success condition, and (matching the
        real trainer's start-state distribution) respect block_start_near_goal.
        """
        cfg = self.cfg
        lengths = self._h5["ep_len"][:]
        offsets = self._h5["ep_offset"][:]
        states = self._h5["state"][:]
        # The probes were trained against this same objective, so probe reward,
        # anchor filtering and the real fixed-target env all agree.
        target_xy = PUSHT_FIXED_TARGET_POSE[:2]
        target_angle = float(PUSHT_FIXED_TARGET_POSE[2])
        min_local = (self.context_steps - 1) * cfg.wm_frameskip

        anchors = []
        for ep in range(len(lengths)):
            local = np.arange(min_local, int(lengths[ep]), dtype=np.int64)
            if len(local) == 0:
                continue
            rows = int(offsets[ep]) + local
            s = states[rows]
            pos_err = np.linalg.norm(s[:, 2:4] - target_xy, axis=1)
            angle_err = np.abs(
                np.arctan2(np.sin(s[:, 4] - target_angle), np.cos(s[:, 4] - target_angle))
            )
            ok = ~((pos_err <= cfg.success_pos_tol) & (angle_err <= cfg.success_angle_tol))
            if cfg.block_start_near_goal:
                ok &= pos_err <= cfg.block_start_radius
            anchors.append(np.stack([np.full(ok.sum(), ep, dtype=np.int64), rows[ok]], axis=1))

        anchors = (
            np.concatenate(anchors, axis=0) if anchors else np.empty((0, 2), dtype=np.int64)
        )
        if len(anchors) == 0:
            raise ValueError(
                "No valid episode-start anchors in the dataset; relax "
                "--block-start-radius or check --dataset-path."
            )
        return anchors

    def _split_anchors(self, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Partition ``[N, 2]`` (episode, row) anchors into train / held-out pools.

        Splitting on the episode column keeps the held-out pool genuinely unseen:
        anchors are dense within an episode, so a frame-level split would put
        near-duplicate start states on both sides and make imagined validation
        success an optimistic estimate of itself.
        """
        cfg = self.cfg
        episodes = np.unique(anchors[:, 0])
        rng = np.random.default_rng(cfg.dream_val_split_seed)
        shuffled = rng.permutation(episodes)
        n_val = max(1, int(round(len(episodes) * cfg.dream_val_fraction)))
        val_episodes = np.sort(shuffled[:n_val])
        self._num_val_episodes = int(n_val)

        is_val = np.isin(anchors[:, 0], val_episodes)
        train_anchors, val_anchors = anchors[~is_val], anchors[is_val]
        if len(train_anchors) == 0 or len(val_anchors) == 0:
            raise ValueError(
                "anchor split left an empty pool; adjust --dream-val-fraction"
            )
        return train_anchors, val_anchors

    @contextmanager
    def eval_mode(self, seed: int, episode_steps: int):
        """Roll out from held-out anchors on fresh state, then restore training.

        Training rollouts continue across iteration boundaries, so an evaluation
        must not touch the live predictor histories; this swaps in an independent
        set plus a fixed RNG, making imagined evaluation reproducible and
        side-effect free.
        """
        n = self.cfg.num_envs
        saved = (
            self._anchors,
            self._emb_hist,
            self._act_hist,
            self._agent_pos,
            self._steps,
            self._rng,
            self._episode_steps,
            self._dense_reward_prev_score,
            self._last_dense_reward,
            self._last_dense_score,
            self._last_dense_probs,
        )
        self._anchors = self._val_anchors
        self._emb_hist = [deque(maxlen=self.history_size) for _ in range(n)]
        self._act_hist = [deque(maxlen=self.history_size) for _ in range(n)]
        self._agent_pos = torch.zeros((n, 2), device=self.device)
        self._steps = np.zeros(n, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self._episode_steps = int(episode_steps)
        self._dense_reward_prev_score = torch.zeros(n, device=self.device)
        self._last_dense_reward = np.zeros(n, dtype=np.float32)
        self._last_dense_score = np.zeros(n, dtype=np.float32)
        self._last_dense_probs = np.zeros_like(self._last_dense_probs)
        try:
            yield
        finally:
            (
                self._anchors,
                self._emb_hist,
                self._act_hist,
                self._agent_pos,
                self._steps,
                self._rng,
                self._episode_steps,
                self._dense_reward_prev_score,
                self._last_dense_reward,
                self._last_dense_score,
                self._last_dense_probs,
            ) = saved

    def _normalize_blocks(self, raw_blocks: torch.Tensor) -> torch.Tensor:
        """``[..., k, 2]`` dataset-space actions -> flattened z-scored ``[..., k*2]``."""
        norm = (raw_blocks - self.action_mean) / self.action_std
        return norm.reshape(*raw_blocks.shape[:-2], -1)

    # ---------------------------------------------------------------- resets
    def reset_env(self, i: int) -> torch.Tensor:
        """Re-seed dream env ``i`` from a fresh ground-truth context window.

        Returns the raw CLS latents of the context frames ``[context_steps, D]``
        for the trainer to refill the agent's dilated history.
        """
        cfg = self.cfg
        fs = cfg.wm_frameskip
        episode, anchor = self._anchors[self._rng.integers(len(self._anchors))]
        # Remember where this dream episode was seeded from. Training ignores it,
        # but grounding an imagined rollout against the simulator (see
        # scripts/rq2/dream_success_gallery.py) needs the exact start state.
        self.last_anchor[i] = {
            "episode": int(episode),
            "row": int(anchor),
            "state": np.asarray(self._h5["state"][int(anchor)], dtype=np.float64),
        }
        start = int(anchor) - (self.context_steps - 1) * fs
        idx = start + np.arange(self.context_steps) * fs

        pixels = self._h5["pixels"][idx]  # [C, H, W, 3] uint8
        frames = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            cls = self.cls_encoder(frames)  # [C, D] raw CLS
            emb = self.wm.projector(cls)  # [C, D] dynamics latent space

        self._emb_hist[i].clear()
        for t in range(self.context_steps):
            self._emb_hist[i].append(emb[t])

        # Ground-truth action blocks between context frames, aligned so that
        # act_hist[t] is the block taken *at* emb_hist[t].
        self._act_hist[i].clear()
        first_block = self.context_steps - self.history_size
        for t in range(max(first_block, 0), self.context_steps - 1):
            a0 = start + t * fs
            raw = torch.as_tensor(
                self._h5["action"][a0 : a0 + fs], dtype=torch.float32, device=self.device
            )
            self._act_hist[i].append(self._normalize_blocks(raw))

        self._agent_pos[i] = torch.as_tensor(
            self._h5["state"][int(anchor)][:2], dtype=torch.float32, device=self.device
        )
        if self.dense_reward is not None:
            score, probs = self.dense_reward.score(emb[-1].unsqueeze(0))
            self._dense_reward_prev_score[i] = score[0]
            self._last_dense_score[i] = float(score[0].detach().cpu())
            self._last_dense_reward[i] = 0.0
            self._last_dense_probs[i] = probs[0].detach().cpu().numpy()
        self._steps[i] = 0
        return cls

    # ----------------------------------------------------------------- bridge
    @property
    def decoder(self) -> nn.Module:
        """Frozen latent image decoder, loaded on first use."""
        if self._decoder is None:
            self._decoder = _load_decoder(
                _resolve_decoder_reference(self.cfg.decoder_checkpoint), self.device
            )
        return self._decoder

    @decoder.setter
    def decoder(self, module: nn.Module) -> None:
        self._decoder = module

    def _observe(self, pred: torch.Tensor) -> torch.Tensor:
        """Imagined dynamics latent -> the raw CLS latent the policy consumes.

        LeWM's predictor works in the projected space; the BC-initialized policy
        reads raw CLS. ``bridge="decoder"`` crosses that gap through pixels;
        ``bridge="deprojector"`` maps latent to latent. ``last_frames`` is
        inspection-only state read by ``scripts/rq2/dream_success_gallery.py``
        and never by training, so the de-projector path only pays for it when
        ``capture_frames`` is set.
        """
        if self._deprojector is not None:
            self.last_frames = self.decoder(pred).clamp(0.0, 1.0) if self.capture_frames else None
            return self._deprojector(pred)
        frames = self.decoder(pred).clamp(0.0, 1.0)  # [n, 3, 224, 224]
        self.last_frames = frames
        return self.cls_encoder(frames)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """Advance every dream env one predictor step.

        ``actions``: clamped SWM-relative chunks ``[num_envs, k, 2]``. Returns
        ``(cls [n, D] tensor, reward, terminated, truncated, block_state_dist)``
        with the trailing four as numpy arrays; rewards are raw (unnormalized).
        """
        cfg = self.cfg
        n, k = cfg.num_envs, cfg.action_chunk_size

        if self.absolute_actions:
            # Integrate commanded targets from the current anchor position
            # (the pointer approximately tracks its target within one step).
            pos = self._agent_pos
            targets = []
            for j in range(k):
                pos = torch.clamp(
                    pos + actions[:, j] * PUSHT_ACTION_SCALE, PUSHT_COORD_LOW, PUSHT_COORD_HIGH
                )
                targets.append(pos)
            raw_blocks = torch.stack(targets, dim=1)  # [n, k, 2]
            integrated_pos = pos
        else:
            raw_blocks = actions
            integrated_pos = self._agent_pos
        flat_blocks = self._normalize_blocks(raw_blocks)  # [n, k*2]

        for i in range(n):
            self._act_hist[i].append(flat_blocks[i])
        emb = torch.stack([torch.stack(tuple(h), dim=0) for h in self._emb_hist], dim=0)
        act = torch.stack([torch.stack(tuple(h), dim=0) for h in self._act_hist], dim=0)

        act_emb = self.wm.action_encoder(act)
        pred = self.wm.predict(emb, act_emb)[:, -1]  # [n, D] next dynamics latent
        for i in range(n):
            self._emb_hist[i].append(pred[i])

        cls = self._observe(pred)
        self.last_latent = pred

        # Reward/success are read from the imagined projected latent. The
        # objective_met classifier decides success when available, falling back
        # to probe-decoded pose vs tolerances.
        state_dist = None
        if self.pose_probe is not None:
            rel = self.pose_probe(pred)  # [rel_x, rel_y, sin, cos]
            pos_dist = torch.linalg.norm(rel[:, :2], dim=1)
            angle_dist = torch.abs(torch.atan2(rel[:, 2], rel[:, 3]))
            state_dist = torch.sqrt(pos_dist**2 + angle_dist**2)
        if self.success_probe is not None:
            success_prob = self.success_probe(pred)[:, 0]
            success = success_prob >= self.success_probe.threshold
        else:
            success_prob = None
            success = (pos_dist < cfg.success_pos_tol) & (angle_dist < cfg.success_angle_tol)
        self.last_success_prob = success_prob

        if cfg.reward_mode in ("sparse", "dense"):
            reward = success.float()
        elif cfg.reward_mode == "pose_dense":
            reward = -state_dist * self._dense_chunk_scale
            if self.engagement_probe is not None:
                agent_block = torch.linalg.norm(self.engagement_probe(pred)[:, :2], dim=1)
                reward = reward - cfg.agent_block_coef * self._dense_chunk_scale * agent_block

        if cfg.reward_mode == "dense":
            if self.dense_reward is None:
                raise RuntimeError("reward_mode='dense' is enabled without dense reward classifier")
            dense_score, dense_probs = self.dense_reward.score(pred)
            dense_shaping = self.dense_reward.reward(
                dense_score,
                self._dense_reward_prev_score,
                mode=cfg.dense_reward_mode,
                discount=cfg.chunk_gamma,
                positive_only=cfg.dense_reward_positive_only,
            )
            reward = reward + dense_shaping
            self._dense_reward_prev_score = dense_score.detach()
            self._last_dense_reward = dense_shaping.detach().cpu().numpy().astype(np.float32)
            self._last_dense_score = dense_score.detach().cpu().numpy().astype(np.float32)
            self._last_dense_probs = dense_probs.detach().cpu().numpy().astype(np.float32)
        else:
            self._last_dense_reward = np.zeros(n, dtype=np.float32)
            self._last_dense_score = np.zeros(n, dtype=np.float32)

        # Logged as "dist": true probe distance when available, else the
        # classifier's distance-to-success proxy 1 - P(objective_met).
        if state_dist is None:
            state_dist = 1.0 - success_prob

        self._agent_pos = self.pos_probe(pred)[:, :2] if self.pos_probe else integrated_pos
        self._steps += 1
        terminated = success.cpu().numpy()
        truncated = (self._steps >= self._episode_steps) & ~terminated

        return (
            cls,
            reward.cpu().numpy().astype(np.float64),
            terminated,
            truncated,
            state_dist.cpu().numpy(),
        )

    def close(self) -> None:
        self._h5.close()


class LeWMDreamPPOTrainer(LatentPPOTrainer):
    """PPO trainer whose rollouts run in the LeWM dream world.

    Mirrors :class:`LatentPPOTrainer` setup minus the real envs; GAE, the PPO
    update, checkpointing, best-tracking and the (real-env) held-out eval are
    inherited unchanged, since the agent contract is identical.
    """

    def __init__(self, cfg: DreamConfig):
        self.cfg = cfg
        _configure_logging()
        set_seed(cfg.seed, cfg.torch_deterministic)
        self.device = get_device(cfg.device)

        self.envs = []  # no real envs during dream training
        self.world = LeWMDreamWorld(cfg, self.device)
        self.encoder = self.world.cls_encoder  # shared ViT, reused by real-env eval
        if cfg.eval_interval > 0 and not cfg.fixed_target:
            logger.warning(
                "Dream rewards are always fixed-target (probe objective %s), but the "
                "real-env held-out eval runs without --fixed_target; pass it so eval "
                "measures the task the agent is trained on.",
                PUSHT_FIXED_TARGET_POSE.tolist(),
            )

        self.agent = build_latent_agent(
            encoder=self.encoder,
            latent_dim=cfg.latent_dim,
            frame_stack=cfg.frame_stack,
            action_dim=cfg.action_dim,
            action_chunk_size=cfg.action_chunk_size,
            hidden_dim=cfg.hidden_dim,
            init_log_std=cfg.init_log_std,
            bc_checkpoint_path=cfg.bc_checkpoint,
            device=self.device,
        )
        self.bc_ref_policy = build_bc_ref_policy(cfg, self.device)
        if cfg.anneal_log_std:
            self.agent.actor.log_std.requires_grad_(False)
            self._set_log_std(cfg.init_log_std)

        trainable = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=cfg.learning_rate, eps=1e-5)

        self.reward_norm = (
            RewardNormalizer(cfg.num_envs, cfg.chunk_gamma, clip=cfg.reward_clip)
            if cfg.norm_reward
            else None
        )

        # One raw-CLS latent per predictor step; stride 1 over predictor steps
        # equals the real trainer's stride-wm_frameskip selection over env steps.
        self.histories = [LatentHistory(cfg.frame_stack, 1) for _ in range(cfg.num_envs)]

        self.global_step = 0  # env-step equivalents (predictor steps x frameskip)
        # Real interaction, which dream training only incurs if the optional
        # diagnostic real-env eval is switched on (eval_interval > 0).
        self._eval_env_steps = 0
        self.start_time = time.time()
        self._ep_returns = deque(maxlen=100)
        self._ep_lengths = deque(maxlen=100)
        self._ep_success = deque(maxlen=100)
        self._ep_final_dist = deque(maxlen=100)
        self._ep_return_acc = np.zeros(cfg.num_envs, dtype=np.float64)
        self._ep_step_acc = np.zeros(cfg.num_envs, dtype=np.int64)
        self._dense_reward_terms = deque(maxlen=100)
        self._dense_reward_scores = deque(maxlen=100)
        self._dense_reward_head_probs = deque(maxlen=100)

        self._best_success = -float("inf")
        self._second_best_success = -float("inf")
        self._eval_env = None  # real env, built lazily by the inherited eval

        run_stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
        self.run_dir = Path(cfg.save_dir) / f"{cfg.exp_name}__seed{cfg.seed}" / run_stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None

    # ------------------------------------------------------------- selection
    def train_env_steps(self) -> int:
        """Zero: dream rollouts never call ``env.step``.

        ``global_step`` counts *imagined* env-step equivalents, so the base
        implementation would misreport it as an interaction budget.
        """
        return 0

    @torch.no_grad()
    def _evaluate_dream(self) -> dict:
        """Deterministic imagined rollouts from held-out anchors. Zero env steps.

        Every env runs a fixed, pre-allocated number of episodes so the estimate
        is unbiased: sampling until N episodes have merely *finished* would
        over-count the short ones, and imagined episodes end early exactly when
        they succeed.
        """
        cfg = self.cfg
        n = cfg.num_envs
        horizon = cfg.dream_eval_steps or cfg.dream_episode_steps

        # Round-robin allocation, so num_envs need not divide the episode count.
        quota = np.full(n, cfg.dream_eval_episodes // n, dtype=np.int64)
        quota[: cfg.dream_eval_episodes % n] += 1

        cpu_rng_state = torch.get_rng_state()
        successes: list[float] = []
        lengths: list[float] = []
        with self.world.eval_mode(cfg.dream_eval_seed, horizon):
            histories = [LatentHistory(cfg.frame_stack, 1) for _ in range(n)]
            for i in range(n):
                ctx = self.world.reset_env(i)
                for t in range(ctx.shape[0]):
                    histories[i].append(ctx[t])
            steps = np.zeros(n, dtype=np.int64)

            while quota.sum() > 0:
                stacked = torch.stack([h.stacked() for h in histories], dim=0)
                # Mean action: this is the policy that gets deployed.
                action = torch.clamp(self.agent.actor.bc_policy(stacked), -1.0, 1.0)
                cls, _, terminated, truncated, _ = self.world.step(action)
                steps += 1

                for i in range(n):
                    histories[i].append(cls[i])
                    if not (terminated[i] or truncated[i]):
                        continue
                    if quota[i] > 0:
                        successes.append(float(terminated[i]))
                        lengths.append(float(steps[i]))
                        quota[i] -= 1
                    steps[i] = 0
                    ctx = self.world.reset_env(i)
                    histories[i].clear()
                    for t in range(ctx.shape[0]):
                        histories[i].append(ctx[t])
        torch.set_rng_state(cpu_rng_state)

        return {
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "episodes": len(successes),
        }

    def _log_dream_eval(self, iteration: int, stats: dict) -> None:
        logger.info(
            "iter %d/%d | DREAM-VAL | success %4.2f | len %5.1f | (%d eps, seed %d, 0 env steps)",
            iteration,
            self.cfg.num_iterations,
            stats["success_rate"],
            stats["mean_length"],
            stats["episodes"],
            self.cfg.dream_eval_seed,
        )
        if self.writer is not None:
            self.writer.log(
                {
                    "eval/dream_success": stats["success_rate"],
                    "eval/dream_length": stats["mean_length"],
                },
                step=self.global_step,
            )

    def _record_selection(self, iteration: int, dream: dict | None, real: dict | None) -> None:
        """Append one row to ``selection_log.jsonl`` in the run directory.

        Pairing imagined and real success at the same checkpoint is what lets us
        report whether interaction-free selection actually tracks the quantity it
        is standing in for.
        """
        row = {
            "iteration": iteration,
            "imagined_steps": self.global_step,
            "env_steps_consumed": self.real_env_steps(),
            "selection": self.cfg.selection,
            "dream_success": dream["success_rate"] if dream else None,
            "real_success": real["success_rate"] if real else None,
            # Episode lengths, in predictor steps (dream) and env steps (real).
            # The imagined one is how deep into the horizon the probe is actually
            # being trusted, which is what the RQ2 false-positive-vs-horizon
            # curve has to be read against.
            "dream_length_steps": dream["mean_length"] if dream else None,
            "dream_length_env_steps": (
                dream["mean_length"] * self.cfg.wm_frameskip if dream else None
            ),
            "real_length_env_steps": real["mean_length"] if real else None,
            "dream_episode_steps": self.cfg.dream_eval_steps or self.cfg.dream_episode_steps,
            "wm_frameskip": self.cfg.wm_frameskip,
        }
        with (self.run_dir / "selection_log.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")

    def _run_selection(self, iteration: int) -> None:
        cfg = self.cfg
        dream_stats = None
        if cfg.dream_eval_interval > 0 and iteration % cfg.dream_eval_interval == 0:
            dream_stats = self._evaluate_dream()
            self._log_dream_eval(iteration, dream_stats)

        # Real-env eval is opt-in (record_real_eval, or the backwards-compatible
        # eval_interval > 0) and, unless selection is "real", purely diagnostic:
        # it is logged for correlation analysis and charged to the interaction
        # budget, but does not pick best.pt.
        real_stats = None
        if cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0:
            real_stats = self._evaluate_heldout()
            self._log_heldout(iteration, real_stats)

        if dream_stats is not None or real_stats is not None:
            self._record_selection(iteration, dream_stats, real_stats)

        if cfg.selection == "dream":
            if dream_stats is not None:
                self._update_best_checkpoints(dream_stats["success_rate"])
        elif cfg.selection == "real":
            if real_stats is not None:
                self._update_best_checkpoints(real_stats["success_rate"])
        else:
            self._update_best_checkpoints()

    def _finalize_selection(self) -> float | None:
        cfg = self.cfg
        dream_stats = self._evaluate_dream() if cfg.dream_eval_interval > 0 else None
        if dream_stats is not None:
            self._log_dream_eval(cfg.num_iterations, dream_stats)
        real_stats = self._evaluate_heldout() if cfg.eval_interval > 0 else None
        if real_stats is not None:
            self._log_heldout(cfg.num_iterations, real_stats)
        if dream_stats is not None or real_stats is not None:
            self._record_selection(cfg.num_iterations, dream_stats, real_stats)

        selected = dream_stats if cfg.selection == "dream" else real_stats
        if selected is None:
            return None
        self._update_best_checkpoints(selected["success_rate"])
        return selected["success_rate"]

    # -------------------------------------------------------------- rollouts
    def _reset_all(self) -> None:
        for i, hist in enumerate(self.histories):
            ctx = self.world.reset_env(i)
            hist.clear()
            for t in range(ctx.shape[0]):
                hist.append(ctx[t])
        self._ep_return_acc[:] = 0.0
        self._ep_step_acc[:] = 0
        self._dense_reward_terms.clear()
        self._dense_reward_scores.clear()
        self._dense_reward_head_probs.clear()

    def collect_rollout(self, done: np.ndarray):
        cfg = self.cfg
        n, e, k = cfg.num_chunks, cfg.num_envs, cfg.action_chunk_size
        D, F = cfg.latent_dim, cfg.frame_stack

        b_latents = torch.zeros((n, e, F, D), dtype=torch.float32, device=self.device)
        b_actions = torch.zeros(
            (n, e, k, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        b_logprobs = np.zeros((n, e), dtype=np.float32)
        b_rewards = np.zeros((n, e), dtype=np.float32)
        b_dones = np.zeros((n, e), dtype=np.float32)
        b_values = np.zeros((n, e), dtype=np.float32)

        for step in range(n):
            stacked = self._stacked_latents()  # [e, F, D]
            b_latents[step] = stacked
            b_dones[step] = done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value_from_latents(
                    stacked
                )
            b_actions[step] = action
            b_logprobs[step] = logprob.cpu().numpy()
            b_values[step] = value.cpu().numpy()

            cls, chunk_reward, terminated, truncated, state_dist = self.world.step(
                torch.clamp(action, -1.0, 1.0)
            )
            if self.world.dense_reward is not None:
                self._dense_reward_terms.append(float(np.mean(self.world._last_dense_reward)))
                self._dense_reward_scores.append(float(np.mean(self.world._last_dense_score)))
                self._dense_reward_head_probs.append(
                    np.mean(self.world._last_dense_probs, axis=0).astype(np.float32)
                )
            self.global_step += e * k
            self._ep_return_acc += chunk_reward
            self._ep_step_acc += k

            step_done = np.zeros(e, dtype=np.float32)
            bootstrap = np.zeros(e, dtype=np.float32)
            for i in range(e):
                self.histories[i].append(cls[i])
                if not (terminated[i] or truncated[i]):
                    continue

                step_done[i] = 1.0
                if truncated[i] and not terminated[i]:
                    # Bootstrap the imagined terminal observation before reset,
                    # mirroring time-limit truncation in the real-env trainer.
                    with torch.no_grad():
                        v_term = self.agent.get_value_from_latents(
                            self.histories[i].stacked().unsqueeze(0)
                        ).item()
                    bootstrap[i] = cfg.chunk_gamma * v_term

                self._ep_returns.append(float(self._ep_return_acc[i]))
                self._ep_lengths.append(float(self._ep_step_acc[i]))
                self._ep_success.append(float(terminated[i]))
                self._ep_final_dist.append(float(state_dist[i]))
                self._ep_return_acc[i] = 0.0
                self._ep_step_acc[i] = 0

                ctx = self.world.reset_env(i)
                self.histories[i].clear()
                for t in range(ctx.shape[0]):
                    self.histories[i].append(ctx[t])

            if self.reward_norm is not None:
                rewards = self.reward_norm.normalize(chunk_reward, step_done)
            else:
                rewards = chunk_reward.astype(np.float32)
            b_rewards[step] = rewards + bootstrap
            done = step_done

        return b_latents, b_actions, b_logprobs, b_rewards, b_dones, b_values, done

    def _log(self, iteration: int, stats: dict) -> None:
        super()._log(iteration, stats)
        if self.world.dense_reward is None or not self._dense_reward_terms:
            return
        dense_reward = float(np.mean(self._dense_reward_terms))
        dense_score = float(np.mean(self._dense_reward_scores))
        head_probs = np.mean(np.stack(tuple(self._dense_reward_head_probs), axis=0), axis=0)
        logger.info(
            "iter %d/%d | dense_reward %.4f | dense_score %.4f | dense_probs %s",
            iteration,
            self.cfg.num_iterations,
            dense_reward,
            dense_score,
            np.round(head_probs, 3).tolist(),
        )
        if self.writer is not None:
            metrics = {
                "charts/dense_reward": dense_reward,
                "charts/dense_reward_score": dense_score,
            }
            for horizon, prob in zip(self.world.dense_reward.horizons, head_probs):
                metrics[f"charts/dense_reward_prob_{horizon}wm"] = float(prob)
            self.writer.log(metrics, step=self.global_step)

    def train(self) -> None:
        try:
            super().train()
        finally:
            self.world.close()


# ------------------------------------------------------------------- CLI
def _add_args(parser: argparse.ArgumentParser) -> None:
    """One flag per :class:`DreamConfig` init field, as in ``src/ppo/train.py``."""
    defaults = DreamConfig()
    for f in fields(DreamConfig):
        if not f.init:
            continue
        names = _flag_names(f.name)
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(
                *names, dest=f.name, action="store_true", default=argparse.SUPPRESS
            )
            parser.add_argument(
                *_flag_names(f.name, prefix="--no-"),
                dest=f.name,
                action="store_false",
                default=argparse.SUPPRESS,
            )
        elif f.name == "target_kl":
            parser.add_argument(*names, dest=f.name, type=float, default=argparse.SUPPRESS)
        elif f.name in _NULLABLE_STR_FIELDS or default is None:
            parser.add_argument(*names, dest=f.name, type=str, default=argparse.SUPPRESS)
        else:
            parser.add_argument(
                *names, dest=f.name, type=type(default), default=argparse.SUPPRESS
            )

def parse_config() -> DreamConfig:
    parser = argparse.ArgumentParser(
        description="Chunk-level latent PPO inside the LeWM world model"
    )
    _add_args(parser)
    parser.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    args = vars(parser.parse_args())
    smoke = args.pop("smoke", False)

    defaults = {f.name: getattr(DreamConfig(), f.name) for f in fields(DreamConfig) if f.init}
    stats_path = args.get("bc_stats", defaults["bc_stats"])
    stats_overrides = _stats_contract(stats_path)

    # Precedence: CLI (args) > smoke > stats > defaults.
    merged = {**defaults, **stats_overrides}
    if smoke:
        merged.update(DREAM_SMOKE_OVERRIDES)
    merged.update(args)
    return DreamConfig(**merged)


def main() -> None:
    cfg = parse_config()
    if cfg.push_to_hf and not cfg.hf_repo_id:
        raise ValueError("--push_to_hf requires --hf_repo_id (e.g. your-username/pusht-latent-ppo)")
    print("Config:")
    for f in fields(DreamConfig):
        print(f"  {f.name} = {getattr(cfg, f.name)}")

    trainer = LeWMDreamPPOTrainer(cfg)

    if cfg.track:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.exp_name}__seed{cfg.seed}",
            config=cfg.__dict__,
            save_code=True,
        )
        trainer.writer = run

    try:
        trainer.train()
        if cfg.push_to_hf:
            _push_checkpoint_to_hf(cfg, trainer.run_dir / "best.pt")
    finally:
        if cfg.track:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
