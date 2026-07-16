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
        assert trainer._best_success >= trainer._second_best_success
        # Encoder stayed frozen; the policy moved.
        for p, before in zip(trainer.encoder.parameters(), enc_before):
            assert torch.equal(p, before)
        assert not torch.equal(trainer.agent.actor.log_std.detach(), logstd_before)

        # Saved checkpoint excludes the (large, frozen) encoder weights.
        # weights_only=False matches src/ppo/evaluate.py (checkpoints carry a
        # config dict + normalizer numpy state).
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
