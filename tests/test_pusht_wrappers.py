"""Tests for the PushT env wrappers.

Runnable without pytest::

    python tests/test_pusht_wrappers.py

Covers the block-start-near-goal wrapper and the sparse/dense reward selector
added to ``src/envs/pusht_wrappers.py``. These need the real ``swm/PushT-v1``
env (``stable_worldmodel``); if it is unavailable the tests self-skip.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import stable_worldmodel  # noqa: F401,E402

    from src.envs import (  # noqa: E402
        PUSHT_FIXED_TARGET_POSE,
        block_center,
        green_t_center,
        make_pusht_env,
    )
    from src.envs.pusht_wrappers import (  # noqa: E402
        PushTRewardModeWrapper,
        _polygon_area_centroid,
    )

    _HAVE_ENV = True
except Exception as exc:  # noqa: BLE001
    print(f"  [skip all] stable_worldmodel/env unavailable: {exc!r}")
    _HAVE_ENV = False


def _has_reward_wrapper(env) -> bool:
    while hasattr(env, "env"):
        if isinstance(env, PushTRewardModeWrapper):
            return True
        env = env.env
    return isinstance(env, PushTRewardModeWrapper)


# --------------------------------------------------------------------------- #
def test_polygon_area_centroid_unit_square():
    """Analytic centroid of a unit square centered at the origin is the origin."""
    if not _HAVE_ENV:
        return
    area, centroid = _polygon_area_centroid([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    assert abs(area - 4.0) < 1e-9, area
    assert np.allclose(centroid, [0.0, 0.0]), centroid


def test_green_t_center_matches_pixels():
    """Analytic green-T center matches the centroid of the rendered green pixels."""
    if not _HAVE_ENV:
        return
    env = make_pusht_env(render_obs=False, resolution=512)
    obs, info = env.reset(seed=3)
    center = green_t_center(env)

    frame = env.unwrapped.render()  # 512x512, same frame the obs renders from
    green = np.array([144, 238, 144])  # LightGreen goal color
    mask = np.all(np.abs(frame.astype(int) - green) < 30, axis=-1)
    ys, xs = np.nonzero(mask)
    # Image col -> world x; image row -> world y (no flip in this render path).
    px_center = np.array([xs.mean(), ys.mean()]) / frame.shape[0] * 512
    assert np.linalg.norm(center - px_center) < 5.0, (center, px_center)
    env.close()


def test_render_observation_shape_tracks_resolution():
    """The declared and returned image shapes match an explicit resolution."""
    if not _HAVE_ENV:
        return
    env = make_pusht_env(resolution=128)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (128, 128, 3)
    assert env.observation_space.shape == observation.shape
    env.close()


def test_block_starts_within_radius():
    """Block centroid spawns within ``radius`` of the green T center, both modes.

    The agent is left where the base env sampled it (this wrapper only moves the
    block).
    """
    if not _HAVE_ENV:
        return
    radius = 50.0
    for fixed in (False, True):
        env = make_pusht_env(
            render_obs=False,
            align_sampled_goal_to_fixed_target=fixed,
            block_start_near_goal=True,
            block_start_radius=radius,
            resolution=96,
        )
        for seed in range(30):
            obs, info = env.reset(seed=seed)
            center = green_t_center(env)
            assert np.allclose(center, info["green_t_center"])
            dist = np.linalg.norm(block_center(env) - center)
            # +2px slack for the single physics tick applied when setting state.
            assert dist <= radius + 2.0, f"fixed={fixed} seed={seed} dist={dist:.2f}"
            assert np.isclose(info["block_goal_dist"], dist, atol=2.0)
        env.close()


def test_block_start_reproducible_with_seed():
    """Same reset seed -> same block start position."""
    if not _HAVE_ENV:
        return
    env = make_pusht_env(
        render_obs=False, block_start_near_goal=True, block_start_radius=50.0, resolution=96
    )
    a = env.reset(seed=7)[0]["state"][2:4].copy()
    b = env.reset(seed=7)[0]["state"][2:4].copy()
    assert np.allclose(a, b), (a, b)
    env.close()


def test_sparse_reward_is_binary_and_matches_success():
    """Sparse reward is in {0,1} and equals the success indicator each step."""
    if not _HAVE_ENV:
        return
    for fixed in (False, True):
        env = make_pusht_env(
            render_obs=False,
            align_sampled_goal_to_fixed_target=fixed,
            reward_mode="sparse",
            resolution=96,
        )
        obs, info = env.reset(seed=0)
        for _ in range(25):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert r in (0.0, 1.0), r
            success = float(info.get("success", info.get("is_success", term))) > 0.5
            assert r == float(success), (r, success)
            if term or trunc:
                obs, info = env.reset()
        env.close()


def test_sparse_reward_one_on_success():
    """Forcing the block onto the fixed target yields success and reward 1.0."""
    if not _HAVE_ENV:
        return
    env = make_pusht_env(
        render_obs=False,
        align_sampled_goal_to_fixed_target=True,
        reward_mode="sparse",
        resolution=96,
    )
    obs, info = env.reset(seed=1)
    u = env.unwrapped
    tgt = PUSHT_FIXED_TARGET_POSE
    state = np.asarray(u._get_obs(), dtype=np.float64).copy()
    state[2:4] = tgt[:2]
    state[4] = tgt[2]
    state[:2] = tgt[:2] + np.array([5.0, 0.0])
    u._set_state(state)
    obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info.get("success", 0.0) > 0.5
    assert r == 1.0, r
    env.close()


def test_dense_reward_is_pass_through():
    """Dense mode adds no reward wrapper and keeps the env's (negative) reward."""
    if not _HAVE_ENV:
        return
    env = make_pusht_env(render_obs=False, reward_mode="dense", resolution=96)
    assert not _has_reward_wrapper(env)
    obs, info = env.reset(seed=0)
    _, r, *_ = env.step(env.action_space.sample())
    assert r <= 1e-6, r  # native reward is -distance
    env.close()
    assert _has_reward_wrapper(
        make_pusht_env(render_obs=False, reward_mode="sparse", resolution=96)
    )


def test_invalid_reward_mode_raises():
    """An unknown reward_mode is rejected."""
    if not _HAVE_ENV:
        return
    try:
        make_pusht_env(render_obs=False, reward_mode="bogus", resolution=96)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid reward_mode")


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
