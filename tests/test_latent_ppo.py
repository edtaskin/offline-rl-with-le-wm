"""Tests for the latent PPO fine-tuning pipeline.

Runnable without pytest::

    python tests/test_latent_ppo.py

Uses a tiny frozen ``DummyImageEncoder`` (returns ``[B, latent_dim]``) so nothing
here needs the real LeWM checkpoint or expert data. Covers:

* output shapes of ``get_action_and_value_from_latents``;
* determinism of the frozen-encoder latent path;
* the gradient contract (BC policy / log_std / critic get grads; encoder does not);
* ``LatentHistory`` dilated frame selection;
* ``build_latent_agent`` loading a local experimental BC checkpoint;
* an end-to-end trainer run on a fake image env (rollout + GAE + update + save).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from src.ppo.agent import LatentPPOAgent, build_latent_agent  # noqa: E402
from src.ppo.config import LatentConfig  # noqa: E402
from src.ppo.env import LatentHistory  # noqa: E402


class DummyImageEncoder(nn.Module):
    """Frozen stand-in for the LeWM encoder: ``[B, C, H, W]`` -> ``[B, latent_dim]``."""

    def __init__(self, latent_dim: int = 192, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(3, latent_dim)
        self.eval()
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(torch.float32) / 255.0
        x = self.pool(x).flatten(1)  # [B, 3]
        return self.proj(x)  # [B, latent_dim]


def _make_agent(
    latent_dim=192, frame_stack=3, action_dim=2, action_chunk_size=1, hidden_dim=256, seed=0
) -> LatentPPOAgent:
    encoder = DummyImageEncoder(latent_dim=latent_dim, seed=seed)
    return build_latent_agent(
        encoder=encoder,
        latent_dim=latent_dim,
        frame_stack=frame_stack,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        bc_checkpoint_path=None,  # random BC policy; hermetic
        device="cpu",
    )


def _encode_image_stack(agent: LatentPPOAgent, image_stack: torch.Tensor) -> torch.Tensor:
    """``[B, F, C, H, W]`` images -> ``[B, F, latent_dim]`` via the frozen encoder."""
    b, f, c, h, w = image_stack.shape
    flat = agent.actor.encoder(image_stack.reshape(b * f, c, h, w))
    return flat.reshape(b, f, -1)


# --------------------------------------------------------------------------- #
def test_get_action_and_value_shapes():
    """Output shapes of the latent path must match the contract."""
    batch_size, frame_stack, action_dim, action_chunk_size = 4, 3, 2, 1
    latent_dim, height, width, channels = 192, 64, 64, 3

    agent = _make_agent(
        latent_dim=latent_dim,
        frame_stack=frame_stack,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
    )
    image_stack = torch.randn(batch_size, frame_stack, channels, height, width)
    stacked = _encode_image_stack(agent, image_stack)
    action, logprob, entropy, value = agent.get_action_and_value_from_latents(stacked)

    assert tuple(action.shape) == (batch_size, action_chunk_size, action_dim), action.shape
    assert tuple(logprob.shape) == (batch_size,), logprob.shape
    assert tuple(entropy.shape) == (batch_size,), entropy.shape
    assert tuple(value.shape) == (batch_size,), value.shape


def test_chunked_shapes():
    """Chunked BC checkpoint (k=5) shapes."""
    b, f, adim, k, ld = 3, 3, 2, 5, 192
    agent = _make_agent(latent_dim=ld, frame_stack=f, action_dim=adim, action_chunk_size=k)
    image_stack = torch.randn(b, f, 3, 32, 32)
    stacked = _encode_image_stack(agent, image_stack)
    action, logprob, entropy, value = agent.get_action_and_value_from_latents(stacked)
    assert tuple(action.shape) == (b, k, adim), action.shape
    assert tuple(logprob.shape) == (b,)
    assert tuple(value.shape) == (b,)


def test_latent_path_deterministic():
    """Frozen encoder: re-encoding the same images gives identical logprob/entropy/value."""
    b, f, adim, k, ld = 4, 3, 2, 5, 192
    agent = _make_agent(latent_dim=ld, frame_stack=f, action_dim=adim, action_chunk_size=k)
    image_stack = torch.randn(b, f, 3, 48, 48)
    action = torch.randn(b, k, adim)

    _, lp_a, ent_a, v_a = agent.get_action_and_value_from_latents(
        _encode_image_stack(agent, image_stack), action
    )
    _, lp_b, ent_b, v_b = agent.get_action_and_value_from_latents(
        _encode_image_stack(agent, image_stack), action
    )

    assert torch.allclose(lp_a, lp_b, atol=1e-6)
    assert torch.allclose(ent_a, ent_b, atol=1e-6)
    assert torch.allclose(v_a, v_b, atol=1e-6)


def test_training_observation_resolution_is_validated():
    from src.ppo.config import LatentConfig

    assert LatentConfig(observation_resolution=96).observation_resolution == 96
    try:
        LatentConfig(observation_resolution=0)
    except ValueError as exc:
        assert "observation_resolution must be positive" in str(exc)
    else:
        raise AssertionError("expected a non-positive observation resolution to fail")


def test_ppo_checkpoint_path_preserves_legacy_and_supports_explicit_bases():
    from src.ppo.config import LatentConfig
    from src.ppo.ppo import ppo_artifact_path, ppo_output_paths

    legacy = LatentConfig(exp_name="example", seed=7, save_dir="runs/ppo")
    run_dir, checkpoint_base = ppo_output_paths(legacy, run_stamp="01012026-120000")
    assert checkpoint_base is None
    assert run_dir == Path("runs/ppo/example__seed7/01012026-120000")
    assert ppo_artifact_path(run_dir, checkpoint_base, "best") == run_dir / "best.pt"

    explicit = LatentConfig(checkpoint_path="runs/ppo/example.pt")
    run_dir, checkpoint_base = ppo_output_paths(explicit, run_stamp="ignored")
    assert run_dir == Path("runs/ppo")
    assert checkpoint_base == Path("runs/ppo/example.pt")
    assert ppo_artifact_path(run_dir, checkpoint_base, "best") == Path(
        "runs/ppo/example_best.pt"
    )
    assert ppo_artifact_path(
        run_dir, checkpoint_base, "selection_log", suffix=".jsonl"
    ) == Path("runs/ppo/example_selection_log.jsonl")


def test_ppo_checkpoint_path_cli_accepts_dash_and_underscore_spellings():
    import src.ppo.train as train

    for flag in ("--checkpoint-path", "--checkpoint_path"):
        parser = argparse.ArgumentParser()
        train._add_args(parser)
        args = parser.parse_args([flag, "runs/ppo/example.pt"])
        assert args.checkpoint_path == "runs/ppo/example.pt"


def test_projected_representation_is_a_valid_agent_contract():
    cfg = LatentConfig(
        latent_representation="projected",
        num_envs=1,
        num_chunks=1,
    )
    assert cfg.latent_representation == "projected"

    try:
        LatentConfig(
            latent_representation="unknown",
            num_envs=1,
            num_chunks=1,
        )
    except ValueError as exc:
        assert "latent representation" in str(exc)
    else:
        raise AssertionError("unknown latent representation was accepted")


def test_bc_stats_are_authoritative_for_ppo_representation():
    import src.ppo.train as train

    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        projected_stats = root / "projected_stats.pth"
        legacy_stats = root / "legacy_stats.pth"
        torch.save({"latent_representation": "projected"}, projected_stats)
        torch.save({}, legacy_stats)

        projected = train._stats_contract(str(projected_stats))
        legacy = train._stats_contract(str(legacy_stats))

    assert projected["latent_representation"] == "projected"
    assert legacy["latent_representation"] == "raw_cls"

    parser = argparse.ArgumentParser()
    train._validate_latent_representation_contract(
        parser, {"latent_representation": "projected"}, projected
    )
    try:
        train._validate_latent_representation_contract(
            parser, {"latent_representation": "raw_cls"}, projected
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("PPO accepted a representation that conflicts with BC stats")


def test_gradient_contract():
    """BC policy / log_std / critic receive grads; the frozen encoder does not."""
    b, f, adim, k, ld = 4, 3, 2, 5, 192
    agent = _make_agent(latent_dim=ld, frame_stack=f, action_dim=adim, action_chunk_size=k)
    image_stack = torch.randn(b, f, 3, 40, 40)
    action = torch.randn(b, k, adim)

    stacked = _encode_image_stack(agent, image_stack)
    _, logprob, entropy, value = agent.get_action_and_value_from_latents(stacked, action)
    loss = -logprob.mean() - 0.01 * entropy.mean() + value.pow(2).mean()
    loss.backward()

    # Trainable components have real gradients.
    bc_grads = [p.grad for p in agent.actor.bc_policy.parameters()]
    assert all(g is not None for g in bc_grads)
    assert any(g.abs().sum() > 0 for g in bc_grads)
    assert agent.actor.log_std.grad is not None
    assert agent.actor.log_std.grad.abs().sum() > 0
    crit_grads = [p.grad for p in agent.critic.value_net.parameters()]
    assert all(g is not None for g in crit_grads)
    assert any(g.abs().sum() > 0 for g in crit_grads)

    # Frozen encoder: no gradients, shared instance across actor/critic.
    assert agent.actor.encoder is agent.critic.encoder
    for p in agent.actor.encoder.parameters():
        assert p.requires_grad is False
        assert p.grad is None


def test_latent_history_dilated_selection():
    """LatentHistory reproduces run_eval's dilated frame selection, incl. padding."""
    hist = LatentHistory(frame_stack=3, frame_stride=2)
    latents = [torch.full((4,), float(i)) for i in range(5)]

    # Early episode: only two frames -> pad with the oldest.
    hist.append(latents[0])
    hist.append(latents[1])
    stacked = hist.stacked()  # expect [l0, l0, l1]
    assert stacked.shape == (3, 4)
    assert torch.equal(stacked[0], latents[0])
    assert torch.equal(stacked[1], latents[0])
    assert torch.equal(stacked[2], latents[1])

    # Fill to max_len=5 -> dilated by stride 2 ending at the newest.
    for i in range(2, 5):
        hist.append(latents[i])
    stacked = hist.stacked()  # expect [l0, l2, l4]
    assert torch.equal(stacked[0], latents[0])
    assert torch.equal(stacked[1], latents[2])
    assert torch.equal(stacked[2], latents[4])


def test_build_latent_agent_loads_bc_checkpoint():
    """The factory still accepts explicit experiment paths outside checkpoints/."""
    reference_agent = _make_agent(action_chunk_size=5)
    reference_state = reference_agent.actor.bc_policy.state_dict()
    with TemporaryDirectory() as temporary_dir:
        ckpt = Path(temporary_dir) / "bc_prior.pth"
        torch.save(reference_state, ckpt)
        agent = build_latent_agent(
            encoder=DummyImageEncoder(latent_dim=192),
            latent_dim=192,
            frame_stack=3,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=256,
            bc_checkpoint_path=str(ckpt),
            device="cpu",
        )
    assert torch.equal(agent.actor.bc_policy.net[0].weight, reference_state["net.0.weight"])
    assert torch.equal(agent.actor.bc_policy.net[4].bias, reference_state["net.4.bias"])


def test_build_bc_ref_policy_loads_frozen_checkpoint():
    """The BC reference policy is loaded from the same artifact path as the actor."""
    import src.ppo.ppo as latent_ppo

    reference_agent = _make_agent(action_chunk_size=5)
    reference_state = reference_agent.actor.bc_policy.state_dict()
    with TemporaryDirectory() as temporary_dir:
        ckpt = Path(temporary_dir) / "bc_prior.pth"
        torch.save(reference_state, ckpt)
        cfg = LatentConfig(
            bc_checkpoint=str(ckpt),
            bc_penalty=True,
            latent_dim=192,
            frame_stack=3,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=256,
            num_envs=1,
            num_chunks=1,
        )
        ref_policy = latent_ppo.build_bc_ref_policy(cfg, torch.device("cpu"))

    assert ref_policy is not None
    assert ref_policy.training is False
    assert all(not p.requires_grad for p in ref_policy.parameters())
    assert torch.equal(ref_policy.net[0].weight, reference_state["net.0.weight"])
    assert torch.equal(ref_policy.net[4].bias, reference_state["net.4.bias"])


def test_build_bc_ref_policy_requires_checkpoint():
    """Turning on the BC penalty without a BC checkpoint should fail early."""
    import src.ppo.ppo as latent_ppo

    cfg = LatentConfig(
        bc_checkpoint=None,
        bc_penalty=True,
        num_envs=1,
        num_chunks=1,
    )
    try:
        latent_ppo.build_bc_ref_policy(cfg, torch.device("cpu"))
    except ValueError as exc:
        assert "bc_checkpoint" in str(exc)
    else:
        raise AssertionError("bc_penalty without bc_checkpoint did not fail")


def test_lewm_dream_trainer_loads_bc_ref_policy():
    """Dream PPO wires the frozen BC reference needed by inherited update()."""
    import shutil
    import src.ppo.train_lewm as train_lewm

    class FakeDreamWorld:
        def __init__(self, cfg, device):
            self.cls_encoder = DummyImageEncoder(latent_dim=cfg.latent_dim)

    reference_agent = _make_agent(action_chunk_size=5)
    reference_state = reference_agent.actor.bc_policy.state_dict()
    with TemporaryDirectory() as temporary_dir:
        ckpt = Path(temporary_dir) / "bc_prior.pth"
        torch.save(reference_state, ckpt)

        orig_world = train_lewm.LeWMDreamWorld
        train_lewm.LeWMDreamWorld = FakeDreamWorld
        try:
            cfg = train_lewm.DreamConfig(
                exp_name="test_lewm_dream_kl",
                device="cpu",
                bc_checkpoint=str(ckpt),
                bc_penalty=True,
                latent_dim=192,
                frame_stack=3,
                frame_stride=5,
                action_chunk_size=5,
                hidden_dim=256,
                action_dim=2,
                num_envs=1,
                num_chunks=1,
                total_timesteps=5,
                eval_interval=0,
                dream_eval_interval=0,
                selection="rolling",
                save_dir=str(REPO_ROOT / "runs"),
            )
            trainer = train_lewm.LeWMDreamPPOTrainer(cfg)
        finally:
            train_lewm.LeWMDreamWorld = orig_world
            shutil.rmtree(REPO_ROOT / "runs" / "test_lewm_dream_kl__seed1", ignore_errors=True)

    assert trainer.bc_ref_policy is not None
    assert trainer.bc_ref_policy.training is False
    assert all(not p.requires_grad for p in trainer.bc_ref_policy.parameters())


def test_bc_penalty_cli_flags():
    """BC penalty CLI flags populate the config fields."""
    import src.ppo.train as train
    import src.ppo.train_lewm as train_lewm

    for add_args in (train._add_args, train_lewm._add_args):
        parser = argparse.ArgumentParser()
        add_args(parser)
        parsed = parser.parse_args(["--bc-penalty", "--bc-penalty-coef", "0.05"])
        assert parsed.bc_penalty is True
        assert parsed.bc_penalty_coef == 0.05

        parsed = parser.parse_args(["--no-bc_penalty"])
        assert parsed.bc_penalty is False


def test_dense_reward_shaper_scores_projected_latents():
    """Dense reward shaping uses projected latent checkpoints and bounded rewards."""
    from src.ppo.dense_reward import DenseRewardClassifier, DenseRewardShaper

    model = DenseRewardClassifier(input_dim=4, output_dim=2, hidden_dim=8, depth=1)
    with TemporaryDirectory() as temporary_dir:
        ckpt = Path(temporary_dir) / "dense_reward_classifier.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "input_dim": 4,
                "output_dim": 2,
                "hidden_dim": 8,
                "depth": 1,
                "monotonic_outputs": False,
                "horizons": [2, 5],
                "frameskip": 5,
                "x_mean": np.zeros(4, dtype=np.float32),
                "x_std": np.ones(4, dtype=np.float32),
            },
            ckpt,
        )
        shaper = DenseRewardShaper(
            ckpt,
            weights="1 0.5",
            scale=2.0,
            clip=0.25,
            device="cpu",
        )

    z_prev = torch.zeros(3, 4)
    z_curr = torch.ones(3, 4)
    prev_score, _ = shaper.score(z_prev)
    curr_score, probs = shaper.score(z_curr)
    reward = shaper.reward(curr_score, prev_score, mode="potential", discount=0.9)

    assert tuple(prev_score.shape) == (3,)
    assert tuple(probs.shape) == (3, 2)
    assert shaper.horizons == [2, 5]
    assert shaper.frameskip == 5
    assert torch.all(reward <= 0.25)
    assert torch.all(reward >= -0.25)


def test_dream_reward_mode_names():
    """Dream PPO distinguishes learned dense reward from pose-distance reward."""
    import src.ppo.train_lewm as train_lewm

    cfg = train_lewm.DreamConfig(
        reward_mode="pose_dense",
        selection="rolling",
        dream_eval_interval=0,
        num_envs=1,
        num_chunks=1,
    )
    assert cfg.reward_mode == "pose_dense"

    cfg = train_lewm.DreamConfig(
        reward_mode="dense",
        dense_reward_coef=0.1,
        selection="rolling",
        dream_eval_interval=0,
        num_envs=1,
        num_chunks=1,
    )
    assert cfg.reward_mode == "dense"

    try:
        train_lewm.DreamConfig(
            reward_mode="dense",
            dense_reward_checkpoint="models/probes/pusht_dense_reward/dense_reward_classifier.pt",
            dense_reward_coef=0.0,
            selection="rolling",
            dream_eval_interval=0,
            num_envs=1,
            num_chunks=1,
        )
    except ValueError as exc:
        assert "dense_reward_coef" in str(exc)
    else:
        raise AssertionError("reward_mode='dense' without positive coefficient did not fail")


def test_dense_reward_checkpoint_config():
    """Dense reward defaults to HF and accepts an explicit checkpoint override."""
    import src.ppo.train_lewm as train_lewm

    defaults = train_lewm.DreamConfig(
        selection="rolling",
        dream_eval_interval=0,
        num_envs=1,
        num_chunks=1,
    )
    assert defaults.dense_reward_checkpoint == train_lewm.DENSE_REWARD_CHECKPOINT_HF

    override = "hf://owner/custom-reward/checkpoint.pt"
    configured = train_lewm.DreamConfig(
        dense_reward_checkpoint=override,
        selection="rolling",
        dream_eval_interval=0,
        num_envs=1,
        num_chunks=1,
    )
    assert configured.dense_reward_checkpoint == override

    parser = argparse.ArgumentParser()
    train_lewm._add_args(parser)
    cli = parser.parse_args(["--dense-reward-checkpoint", override])
    assert cli.dense_reward_checkpoint == override

    try:
        train_lewm.DreamConfig(
            reward_mode="dense",
            dense_reward_checkpoint="",
            selection="rolling",
            dream_eval_interval=0,
            num_envs=1,
            num_chunks=1,
        )
    except ValueError as exc:
        assert "dense_reward_checkpoint" in str(exc)
    else:
        raise AssertionError("dense reward accepted an empty checkpoint")


def test_projected_dream_observation_bypasses_decoder_and_deprojector():
    import src.ppo.train_lewm as train_lewm

    class MustNotRun(nn.Module):
        def forward(self, value):
            raise AssertionError("a projected policy must not use a bridge")

    world = object.__new__(train_lewm.LeWMDreamWorld)
    world.policy_uses_projected = True
    world.capture_frames = False
    world._decoder = MustNotRun()
    world._deprojector = MustNotRun()
    pred = torch.randn(3, 192)

    observed = world._observe(pred)

    assert observed is pred
    assert world.last_frames is None


def test_dream_dense_reward_adds_sparse_success():
    """Dense dream reward is sparse success plus dense shaping, never shaping alone."""
    from collections import deque
    from types import SimpleNamespace

    import src.ppo.train_lewm as train_lewm

    class FakeWM:
        predictor = SimpleNamespace(num_frames=1)

        def action_encoder(self, act):
            return act

        def predict(self, emb, act_emb):
            return torch.full((emb.shape[0], 1, emb.shape[-1]), 0.5)

    class FakeSuccessProbe(nn.Module):
        threshold = 0.5

        def forward(self, z):
            return torch.ones((z.shape[0], 1), device=z.device)

    class FakeDenseReward:
        def score(self, z):
            return torch.full((z.shape[0],), 2.0, device=z.device), torch.ones((z.shape[0], 2), device=z.device)

        def reward(self, current_score, previous_score, **kwargs):
            return torch.full_like(current_score, 0.25)

    world = object.__new__(train_lewm.LeWMDreamWorld)
    world.cfg = train_lewm.DreamConfig(
        reward_mode="dense",
        dense_reward_coef=0.1,
        selection="rolling",
        dream_eval_interval=0,
        num_envs=2,
        action_chunk_size=5,
        wm_frameskip=5,
        frame_stride=5,
        frame_stack=3,
    )
    world.device = torch.device("cpu")
    world.wm = FakeWM()
    world.history_size = 1
    world.decoder = nn.Identity()
    world.cls_encoder = nn.Identity()
    world.pose_probe = None
    world.success_probe = FakeSuccessProbe()
    world.engagement_probe = None
    world.pos_probe = None
    world.dense_reward = FakeDenseReward()
    world.action_mean = torch.zeros(2)
    world.action_std = torch.ones(2)
    world.absolute_actions = False
    world._agent_pos = torch.zeros((2, 2))
    world._emb_hist = [deque([torch.zeros(192)], maxlen=1) for _ in range(2)]
    world._act_hist = [deque([torch.zeros(10)], maxlen=1) for _ in range(2)]
    world._dense_reward_prev_score = torch.zeros(2)
    world._last_dense_reward = np.zeros(2, dtype=np.float32)
    world._last_dense_score = np.zeros(2, dtype=np.float32)
    world._last_dense_probs = np.zeros((2, 2), dtype=np.float32)
    world._steps = np.zeros(2, dtype=np.int64)
    world._episode_steps = 10

    _, reward, terminated, truncated, _ = world.step(torch.zeros((2, 5, 2)))

    assert np.allclose(reward, np.asarray([1.25, 1.25]))
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]


class _FakeImageEnv(gym.Env):
    """Random-image PushT stand-in: truncates after ``max_steps``; dummy reward."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, img_shape=(64, 64, 3), max_steps=6):
        self.observation_space = spaces.Box(0, 255, img_shape, dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.img_shape = img_shape
        self.max_steps = max_steps
        self.t = 0
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = -float(np.linalg.norm(np.asarray(action)))
        terminated = False
        truncated = self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"is_success": False}

    def _obs(self):
        return self.rng.integers(0, 256, self.img_shape, dtype=np.uint8)

    def render(self):
        return self._obs()


def test_trainer_saves_to_explicit_checkpoint_base():
    """An explicit base writes BC-style sibling artifacts without a timestamp."""
    import src.ppo.ppo as latent_ppo
    from src.ppo.config import LatentConfig

    def fake_make_latent_env(*, seed=0, idx=0, max_episode_steps=6, **kwargs):
        def thunk():
            env = _FakeImageEnv(max_steps=max_episode_steps)
            env.action_space.seed(seed + idx)
            return env

        return thunk

    original = latent_ppo.make_latent_env
    latent_ppo.make_latent_env = fake_make_latent_env
    try:
        with TemporaryDirectory() as temporary_dir:
            checkpoint_base = Path(temporary_dir) / "explicit_ppo.pt"
            cfg = LatentConfig(
                device="cpu",
                bc_checkpoint=None,
                checkpoint_path=str(checkpoint_base),
                num_envs=1,
                num_chunks=1,
            )
            trainer = latent_ppo.LatentPPOTrainer(
                cfg,
                encoder=DummyImageEncoder(latent_dim=cfg.latent_dim),
            )
            saved = trainer.save_checkpoint("final")
            assert saved == checkpoint_base.with_name("explicit_ppo_final.pt")
            assert saved.is_file()
            assert not checkpoint_base.exists()
            for env in trainer.envs:
                env.close()
    finally:
        latent_ppo.make_latent_env = original


def test_trainer_end_to_end_fake_env(monkeypatch=None):
    """Full rollout + GAE + update + checkpoint on a fake env with a dummy encoder."""
    import src.ppo.ppo as latent_ppo
    from src.ppo.config import LatentConfig

    def fake_make_latent_env(*, seed=0, idx=0, max_episode_steps=6, **kwargs):
        def thunk():
            env = _FakeImageEnv(max_steps=max_episode_steps)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed + idx)
            return env

        return thunk

    orig = latent_ppo.make_latent_env
    latent_ppo.make_latent_env = fake_make_latent_env
    try:
        cfg = LatentConfig(
            exp_name="test_latent_ppo",
            device="cpu",
            bc_checkpoint=None,
            latent_dim=192,
            frame_stack=3,
            frame_stride=2,
            action_chunk_size=5,
            hidden_dim=256,
            action_dim=2,
            num_envs=2,
            num_chunks=3,
            total_timesteps=2 * 3 * 5 * 2,  # ~2 iterations
            update_epochs=2,
            num_minibatches=2,
            max_episode_steps=6,
            save_dir=str(REPO_ROOT / "runs"),
            save_interval=1,
            eval_interval=1,
            eval_episodes=2,
        )
        encoder = DummyImageEncoder(latent_dim=192)
        trainer = latent_ppo.LatentPPOTrainer(cfg, encoder=encoder)

        enc_before = [p.detach().clone() for p in trainer.encoder.parameters()]
        logstd_before = trainer.agent.actor.log_std.detach().clone()

        trainer.train()

        assert trainer.global_step > 0
        assert (trainer.run_dir / "final.pt").exists()
        # Best / second-best (by success rate) are saved and carry their score.
        assert (trainer.run_dir / "best.pt").exists()
        assert (trainer.run_dir / "second_best.pt").exists()
        best_ckpt = torch.load(
            trainer.run_dir / "best.pt", map_location="cpu", weights_only=False
        )
        assert "success_rate" in best_ckpt
        assert best_ckpt["config"]["observation_resolution"] == cfg.observation_resolution
        assert trainer._best_success >= trainer._second_best_success
        # Encoder stayed frozen; the policy moved.
        for p, before in zip(trainer.encoder.parameters(), enc_before):
            assert torch.equal(p, before)
        assert not torch.equal(trainer.agent.actor.log_std.detach(), logstd_before)

        # Saved checkpoint excludes the (large, frozen) encoder weights.
        # weights_only=False matches the canonical evaluation loader (checkpoints
        # carry a config dict + normalizer numpy state).
        ckpt = torch.load(trainer.run_dir / "final.pt", map_location="cpu", weights_only=False)
        assert not any(k.startswith(("actor.encoder.", "critic.encoder.")) for k in ckpt["agent"])
    finally:
        latent_ppo.make_latent_env = orig
        import shutil

        # Run dirs are now timestamped ({exp}__seed{seed}__{DDMMYYYY-HHMMSS}); glob them.
        for run_dir in (REPO_ROOT / "runs").glob("test_latent_ppo__seed*"):
            shutil.rmtree(run_dir, ignore_errors=True)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"ERROR {t.__name__}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
