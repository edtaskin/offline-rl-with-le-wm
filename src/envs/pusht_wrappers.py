import gymnasium as gym
import numpy as np
from gymnasium import spaces


PUSHT_ENV_ID = "swm/PushT-v1"
PUSHT_RENDER_SHAPE = (96, 96, 3)
PUSHT_FIXED_TARGET_POSE = np.array([256.0, 256.0, np.pi / 4], dtype=np.float64)
PUSHT_WORKSPACE_LOW = np.array([0.0, 0.0], dtype=np.float64)
PUSHT_WORKSPACE_HIGH = np.array([512.0, 512.0], dtype=np.float64)


def _wrap_angle(angle):
    return float(angle % (2 * np.pi))


def _angle_distance(angle_a, angle_b):
    diff = abs(float(angle_a) - float(angle_b))
    return min(diff, 2 * np.pi - diff)


def _rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _polygon_area_centroid(vertices):
    """Signed area and centroid of a polygon given ordered ``(x, y)`` vertices."""
    verts = np.asarray(vertices, dtype=np.float64)
    x, y = verts[:, 0], verts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-9:
        # Degenerate polygon: fall back to the vertex mean.
        return 0.0, verts.mean(axis=0)
    cx = float(((x + x_next) * cross).sum()) / (6.0 * area)
    cy = float(((y + y_next) * cross).sum()) / (6.0 * area)
    return abs(area), np.array([cx, cy], dtype=np.float64)


def _block_local_centroid(unwrapped):
    """Area-weighted centroid of the block's shapes in body-local coordinates.

    The block's body origin is not its geometric center (for a T it sits at the
    top edge of the bar, ~40px from the centroid), so both the green goal marker
    and the block are best located by this centroid rather than ``position``.
    """
    total_area = 0.0
    weighted = np.zeros(2, dtype=np.float64)
    for shape in unwrapped.block.shapes:
        get_vertices = getattr(shape, "get_vertices", None)
        if get_vertices is not None:  # pymunk.Poly
            area, centroid = _polygon_area_centroid(
                [tuple(v) for v in get_vertices()]
            )
        else:  # pymunk.Circle
            radius = float(getattr(shape, "radius", 0.0))
            area = float(np.pi * radius * radius)
            centroid = np.asarray(tuple(getattr(shape, "offset", (0.0, 0.0))), dtype=np.float64)
        total_area += area
        weighted += area * centroid
    return weighted / total_area if total_area > 0 else np.zeros(2)


def green_t_center(env):
    """World-space centroid of the rendered green goal ("green T") marker.

    The goal is drawn by transforming the block's own shapes by the goal pose
    (``env.goal_pose = [x, y, angle]``), so the green marker occupies exactly the
    block polygon rotated/translated to the goal pose. We return the
    area-weighted centroid of that polygon in the same coordinate frame as the
    agent/block positions (``512``-pixel pymunk world space) -- i.e. the visual
    center of the green T.
    """
    unwrapped = env.unwrapped
    goal_pose = np.asarray(unwrapped.goal_pose, dtype=np.float64)
    return goal_pose[:2] + _rotation_matrix(float(goal_pose[2])) @ _block_local_centroid(unwrapped)


def block_center(env):
    """World-space centroid of the current block pose (matches :func:`green_t_center`)."""
    unwrapped = env.unwrapped
    state = np.asarray(unwrapped._get_obs(), dtype=np.float64)
    block_pos, block_angle = state[2:4], float(state[4])
    return block_pos + _rotation_matrix(block_angle) @ _block_local_centroid(unwrapped)


def _success_from_info(info, terminated):
    """Task-success flag: prefer explicit info keys, else the terminated flag.

    Mirrors ``src.ppo.latent_env.success_from_info`` so the sparse reward agrees
    with how success is tracked elsewhere (block-pose success under
    ``fixed_target``; native ``terminated``-on-success otherwise).
    """
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return float(info[key]) > 0.5
    return bool(terminated)


class PushTGoalPoseFromStateWrapper(gym.Wrapper):
    """Make the rendered PushT goal match the state used for reward/success."""

    def _sync_goal_pose(self, info=None):
        env = self.unwrapped
        goal_state = getattr(env, "goal_state", None)
        if goal_state is None:
            return info

        goal_state = np.asarray(goal_state)
        if goal_state.shape[0] < 5:
            return info

        env.goal_pose = goal_state[2:5].copy()
        if info is not None:
            info = dict(info)
            info["goal_pose"] = env.goal_pose
        return info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = self._sync_goal_pose(info)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = self._sync_goal_pose(info)
        return observation, reward, terminated, truncated, info


class PushTAlignSampledGoalToFixedTargetWrapper(gym.Wrapper):
    """Rigidly align each sampled PushT task to a fixed block target pose."""

    def __init__(
        self,
        env,
        target_pose=PUSHT_FIXED_TARGET_POSE,
        block_success=True,
        block_position_threshold=20.0,
        block_angle_threshold=np.pi / 9,
        max_reset_attempts=100,
        workspace_low=PUSHT_WORKSPACE_LOW,
        workspace_high=PUSHT_WORKSPACE_HIGH,
        agent_block_coef=0.0,
    ):
        super().__init__(env)
        self.target_pose = np.asarray(target_pose, dtype=np.float64)
        if self.target_pose.shape != (3,):
            raise ValueError("target_pose must contain [x, y, angle]")
        self.block_success = bool(block_success)
        # Optional reward shaping: penalize agent->block distance so the policy
        # stays engaged with the block instead of drifting off (helps far/rotation
        # transports). 0.0 = disabled; success/termination are unaffected.
        self.agent_block_coef = float(agent_block_coef)
        self.block_position_threshold = float(block_position_threshold)
        self.block_angle_threshold = float(block_angle_threshold)
        self.max_reset_attempts = int(max_reset_attempts)
        if self.max_reset_attempts < 1:
            raise ValueError("max_reset_attempts must be at least 1")
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)

    def _transform_pose_xy(self, xy, sampled_goal_pose):
        dtheta = self.target_pose[2] - sampled_goal_pose[2]
        rotation = _rotation_matrix(dtheta)
        return self.target_pose[:2] + rotation @ (np.asarray(xy) - sampled_goal_pose[:2])

    def _transform_state(self, state, sampled_goal_pose):
        state = np.asarray(state, dtype=np.float64).copy()
        dtheta = self.target_pose[2] - sampled_goal_pose[2]
        rotation = _rotation_matrix(dtheta)

        state[:2] = self.target_pose[:2] + rotation @ (state[:2] - sampled_goal_pose[:2])
        state[2:4] = self.target_pose[:2] + rotation @ (state[2:4] - sampled_goal_pose[:2])
        state[4] = _wrap_angle(state[4] + dtheta)
        if state.shape[0] >= 7:
            state[-2:] = rotation @ state[-2:]
        return state

    def _is_valid_state(self, state):
        agent_xy = state[:2]
        block_xy = state[2:4]
        return (
            np.all(agent_xy >= self.workspace_low)
            and np.all(agent_xy <= self.workspace_high)
            and np.all(block_xy >= self.workspace_low)
            and np.all(block_xy <= self.workspace_high)
        )

    def _format_observation(self):
        env = self.unwrapped
        state = env._get_obs()
        proprio = np.concatenate((state[:2], state[-2:]))
        return {"proprio": proprio, "state": state}

    def _refresh_goal_image(self, current_state):
        env = self.unwrapped
        if not hasattr(env, "_goal"):
            return

        env._set_state(env.goal_state)
        env._goal = env.render()
        env._set_state(current_state)

    def _block_metrics(self):
        env = self.unwrapped
        state = env._get_obs()
        block_pose = state[2:5]
        goal_pose = np.asarray(env.goal_state[2:5], dtype=np.float64)
        pos_dist = float(np.linalg.norm(goal_pose[:2] - block_pose[:2]))
        angle_dist = _angle_distance(goal_pose[2], block_pose[2])
        state_dist = float(np.linalg.norm([pos_dist, angle_dist]))
        success = (
            pos_dist < self.block_position_threshold
            and angle_dist < self.block_angle_threshold
        )
        return success, pos_dist, angle_dist, state_dist

    def _augment_info(self, info=None):
        env = self.unwrapped
        info = dict(env._get_info())
        info["goal_pose"] = np.asarray(env.goal_pose).copy()
        info["goal_state"] = np.asarray(env.goal_state).copy()
        state = env._get_obs()
        info["agent_block_dist"] = float(np.linalg.norm(state[:2] - state[2:4]))
        if self.block_success:
            success, pos_dist, angle_dist, state_dist = self._block_metrics()
            info.update(
                {
                    "success": float(success),
                    "block_success": float(success),
                    "block_pos_dist": pos_dist,
                    "block_angle_dist": angle_dist,
                    "block_state_dist": state_dist,
                }
            )
        return info

    def _apply_alignment(self, info=None):
        env = self.unwrapped
        current_state = env._get_obs()
        sampled_goal_state = np.asarray(env.goal_state, dtype=np.float64).copy()
        sampled_goal_pose = sampled_goal_state[2:5].copy()

        aligned_state = self._transform_state(current_state, sampled_goal_pose)
        aligned_goal_state = self._transform_state(sampled_goal_state, sampled_goal_pose)
        aligned_goal_state[2:5] = self.target_pose.copy()

        if not self._is_valid_state(aligned_state):
            return None, None

        env._set_goal_state(aligned_goal_state)
        env.goal_pose = self.target_pose.copy()
        env._set_state(aligned_state)
        self._refresh_goal_image(aligned_state)
        return self._format_observation(), self._augment_info(info)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        base_kwargs = dict(kwargs)
        for attempt in range(self.max_reset_attempts):
            reset_kwargs = dict(base_kwargs)
            if seed is not None and attempt > 0:
                reset_kwargs["seed"] = int(seed) + attempt
            observation, info = self.env.reset(**reset_kwargs)
            observation, info = self._apply_alignment(info)
            if observation is not None:
                return observation, info

        raise RuntimeError(
            "Could not sample a valid fixed-target PushT episode after "
            f"{self.max_reset_attempts} reset attempts."
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.unwrapped.goal_pose = self.target_pose.copy()
        info = self._augment_info(info)
        if self.block_success:
            terminated = bool(info["block_success"])
            reward = -float(info["block_state_dist"])
            if self.agent_block_coef:
                reward -= self.agent_block_coef * info["agent_block_dist"]
        return observation, reward, terminated, truncated, info


class PushTRenderObservationWrapper(gym.ObservationWrapper):
    """Return env.render() RGB frames as observations."""

    def __init__(self, env, image_shape=PUSHT_RENDER_SHAPE):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=image_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.asarray(self.env.render(), dtype=np.uint8)


class PushTBlockStartNearGoalWrapper(gym.Wrapper):
    """Start each episode with the block near the green T (goal) center.

    On every ``reset`` the block is repositioned so its centroid lands at a
    uniform-random point inside a disk of radius ``radius`` pixels around the
    green T center (see :func:`green_t_center` / :func:`block_center`); the block
    *angle* and the agent are left untouched. This shortens the block-transport
    distance -- the main driver of PushT difficulty -- into a controllable range.

    The block origin is clipped to stay ``bounds_margin`` pixels inside the
    workspace. Runs *after* any goal alignment/sync, so the disk is centered on
    the final (possibly fixed) goal.
    """

    def __init__(
        self,
        env,
        radius=50.0,
        bounds_margin=20.0,
        min_agent_clearance=0.0,
        max_sample_attempts=100,
        workspace_low=PUSHT_WORKSPACE_LOW,
        workspace_high=PUSHT_WORKSPACE_HIGH,
    ):
        super().__init__(env)
        self.radius = float(radius)
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative")
        self.bounds_margin = float(bounds_margin)
        # Optional: reject samples whose block centroid lands within this distance
        # of the agent (avoids spawning the block on top of the pusher). 0 = off.
        self.min_agent_clearance = float(min_agent_clearance)
        self.max_sample_attempts = max(1, int(max_sample_attempts))
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)
        self._low = self.workspace_low + self.bounds_margin
        self._high = self.workspace_high - self.bounds_margin
        self._rng = np.random.default_rng()

    def _sample_block_pos(self, goal_center, agent_xy, block_angle, local_centroid):
        # Choose where the block *centroid* should land, then back out the block
        # origin for the (unchanged) block angle: pos = centroid - R(angle) @ lc.
        rot = _rotation_matrix(block_angle)
        best = None
        for _ in range(self.max_sample_attempts):
            radius = self.radius * np.sqrt(self._rng.random())
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            target_centroid = goal_center + radius * np.array([np.cos(theta), np.sin(theta)])
            block_pos = np.clip(target_centroid - rot @ local_centroid, self._low, self._high)
            best = block_pos
            resulting_centroid = block_pos + rot @ local_centroid
            if (
                self.min_agent_clearance <= 0.0
                or np.linalg.norm(resulting_centroid - agent_xy) >= self.min_agent_clearance
            ):
                return block_pos
        return best  # clearance unsatisfiable in bounds; use the last sample

    def _reposition_block(self, observation, info):
        env = self.unwrapped
        state = np.asarray(env._get_obs(), dtype=np.float64)
        goal_center = green_t_center(env)
        local_centroid = _block_local_centroid(env)
        new_block_pos = self._sample_block_pos(
            goal_center, state[:2], float(state[4]), local_centroid
        )

        new_state = state.copy()
        new_state[2:4] = new_block_pos
        env._set_state(new_state)

        state = np.asarray(env._get_obs(), dtype=np.float64)
        observation = {
            "proprio": np.concatenate((state[:2], state[-2:])),
            "state": state,
        }
        info = dict(info)
        info["green_t_center"] = goal_center
        info["block_pose"] = np.array(list(state[2:4]) + [state[4]])
        info["block_goal_dist"] = float(np.linalg.norm(block_center(env) - goal_center))
        if "agent_block_dist" in info:
            info["agent_block_dist"] = float(np.linalg.norm(state[:2] - state[2:4]))
        return observation, info

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        observation, info = self.env.reset(**kwargs)
        return self._reposition_block(observation, info)


class PushTRewardModeWrapper(gym.Wrapper):
    """Select between the environment's dense reward and a sparse success reward.

    ``reward_mode="dense"`` passes the underlying reward through unchanged.
    ``reward_mode="sparse"`` replaces it with ``success_reward`` (default ``1.0``)
    on the success step and ``failure_reward`` (default ``0.0``) otherwise, using
    the same success signal tracked elsewhere (see :func:`_success_from_info`).
    """

    def __init__(
        self,
        env,
        reward_mode="dense",
        success_reward=1.0,
        failure_reward=0.0,
    ):
        super().__init__(env)
        if reward_mode not in ("dense", "sparse"):
            raise ValueError("reward_mode must be 'dense' or 'sparse'")
        self.reward_mode = reward_mode
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_mode == "sparse":
            success = _success_from_info(info, terminated)
            reward = self.success_reward if success else self.failure_reward
        return observation, float(reward), terminated, truncated, info


def make_pusht_env(
    *,
    env_id=PUSHT_ENV_ID,
    render_mode="rgb_array",
    render_obs=True,
    sync_goal_pose=True,
    align_sampled_goal_to_fixed_target=False,
    fixed_target_pose=PUSHT_FIXED_TARGET_POSE,
    fixed_target_block_success=True,
    fixed_target_max_reset_attempts=100,
    fixed_target_agent_block_coef=0.0,
    block_start_near_goal=False,
    block_start_radius=50.0,
    block_start_min_agent_clearance=0.0,
    reward_mode="dense",
    **kwargs,
):
    import stable_worldmodel  # noqa: F401

    kwargs.setdefault("resolution", PUSHT_RENDER_SHAPE[0])
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if align_sampled_goal_to_fixed_target:
        env = PushTAlignSampledGoalToFixedTargetWrapper(
            env,
            target_pose=fixed_target_pose,
            block_success=fixed_target_block_success,
            max_reset_attempts=fixed_target_max_reset_attempts,
            agent_block_coef=fixed_target_agent_block_coef,
        )
    if sync_goal_pose:
        env = PushTGoalPoseFromStateWrapper(env)
    # Reposition the block near the green T. This must wrap both the alignment
    # and goal-pose-sync wrappers so ``goal_pose`` already reflects the actually
    # rendered goal when the green T center is computed.
    if block_start_near_goal:
        env = PushTBlockStartNearGoalWrapper(
            env,
            radius=block_start_radius,
            min_agent_clearance=block_start_min_agent_clearance,
        )
    if reward_mode != "dense":
        env = PushTRewardModeWrapper(env, reward_mode=reward_mode)
    if render_obs:
        env = PushTRenderObservationWrapper(env)
    return env
