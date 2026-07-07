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
    ):
        super().__init__(env)
        self.target_pose = np.asarray(target_pose, dtype=np.float64)
        if self.target_pose.shape != (3,):
            raise ValueError("target_pose must contain [x, y, angle]")
        self.block_success = bool(block_success)
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
        )
    if sync_goal_pose:
        env = PushTGoalPoseFromStateWrapper(env)
    if render_obs:
        env = PushTRenderObservationWrapper(env)
    return env
