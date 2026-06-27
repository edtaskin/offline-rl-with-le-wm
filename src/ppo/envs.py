"""Environment construction and wrappers for ``swm/PushT-v1``.

The raw environment returns a dict observation ``{"proprio", "state"}`` and the
episode goal lives in ``info["goal_state"]`` (it is re-randomized every reset).
Because the goal changes per episode, the policy must observe it, so we flatten
the observation to ``concat(state, goal_state)`` (14-D).
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from pymunk.vec2d import Vec2d

# Importing the package registers ``swm/PushT-v1`` (and friends) with gymnasium.
import stable_worldmodel  # noqa: F401
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.utils import DrawOptions


class GoalConditionedFlatten(gym.Wrapper):
    """Flatten the dict obs into ``concat(state, goal_state)`` and expose success.

    ``info["goal_state"]`` is constant within an episode, so we capture it at
    reset. The base env only sets ``terminated=True`` on success, which we
    surface explicitly as ``info["is_success"]``.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        state_space = env.observation_space["state"]
        self._state_dim = int(state_space.shape[0])
        obs_dim = 2 * self._state_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._goal: np.ndarray | None = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._goal = np.asarray(info["goal_state"], dtype=np.float32)
        return self._build_obs(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["is_success"] = bool(terminated)
        return self._build_obs(obs), float(reward), terminated, truncated, info

    def _build_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32)
        return np.concatenate([state, self._goal]).astype(np.float32)


def make_env(
    env_id: str = "swm/PushT-v1",
    *,
    seed: int = 0,
    idx: int = 0,
    max_episode_steps: int = 200,
    render_mode: str = "rgb_array",
    record_stats: bool = True,
    **env_kwargs,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a single wrapped environment instance.

    Uses the environment's native reward (``-||goal_state - state||``) and
    success condition; ``GoalConditionedFlatten`` exposes the goal to the policy.
    """

    def thunk() -> gym.Env:
        env = gym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            **env_kwargs,
        )
        env = GoalConditionedFlatten(env)
        env = gym.wrappers.ClipAction(env)
        if record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Reach2D: a simple goal-reaching task (agent circle must reach a target point).
# Same physics/observation conventions as swm PushT, but with no block — useful
# as an easy sanity environment that plain PPO solves quickly.
# ---------------------------------------------------------------------------

DEFAULT_VARIATIONS = ("agent.start_position",)


class Reach2D(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
        "render_fps": 10,
    }
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        damping=None,
        render_action=False,
        resolution=224,
        with_target=True,
        render_mode="rgb_array",
        relative=True,
        init_value=None,
    ):
        self.seed = None
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = resolution
        self.relative = relative
        self.action_scale = 100

        # physics
        self.control_hz = self.metadata["render_fps"]
        self.k_p, self.k_v = 100, 20
        self.dt = 0.01

        self.goal_state = None

        self.shapes = ["o", "L", "T", "Z", "square", "I", "small_tee", "+"]

        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([ws, ws]),
                    dtype=np.float64,
                ),
                "state": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([ws, ws]),
                    dtype=np.float64,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.variation_space = swm_spaces.Dict(
            {
                "agent": swm_spaces.Dict(
                    {
                        "color": swm_spaces.RGBBox(
                            init_value=np.array(
                                pygame.Color("RoyalBlue")[:3], dtype=np.uint8
                            )
                        ),
                        "scale": swm_spaces.Box(
                            low=20,
                            high=60,
                            init_value=40,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "shape": swm_spaces.Discrete(
                            len(self.shapes), start=0, init_value=0
                        ),
                        "angle": swm_spaces.Box(
                            low=-2 * np.pi,
                            high=2 * np.pi,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                        "start_position": swm_spaces.Box(
                            low=50,
                            high=450,
                            init_value=np.array((256, 400), dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        "velocity": swm_spaces.Box(
                            low=0,
                            high=ws,
                            init_value=np.array((0.0, 0.0), dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "goal": swm_spaces.Dict(
                    {
                        "color": swm_spaces.RGBBox(
                            init_value=np.array(
                                pygame.Color("LightGreen")[:3], dtype=np.uint8
                            )
                        ),
                        "scale": swm_spaces.Box(
                            low=20,
                            high=60,
                            init_value=40,
                            shape=(),
                            dtype=np.float32,
                        ),
                        "position": swm_spaces.Box(
                            low=50,
                            high=450,
                            init_value=np.array([256, 256], dtype=np.float64),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
                "background": swm_spaces.Dict(
                    {
                        "color": swm_spaces.RGBBox(
                            init_value=np.array(
                                np.array([255, 255, 255], dtype=np.uint8)
                            )
                        ),
                    }
                ),
                "rendering": swm_spaces.Dict(
                    {"render_goal": swm_spaces.Discrete(2, init_value=1)}
                ),
            },
            sampling_order=[
                "background",
                "goal",
                "agent",
                "rendering",
            ],
        )

        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        self.damping = damping
        self.render_action = render_action
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.render_buffer = None
        self.latest_action = None

        self.with_target = with_target
        self.coverage_arr = []
        self.env_name = "Reach2D"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.rng = np.random.default_rng(seed)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        ### setup pymunk space
        self._setup()

        if self.damping is not None:
            self.space.damping = self.damping

        ### get the state
        if options is not None and "goal_state" in options:
            goal_state = options["goal_state"]
        else:
            goal_state = (
                self.variation_space["agent"]["start_position"]
                .sample(set_value=False)
                .tolist()
            )

        ### generate goal
        self._set_state(goal_state)
        self._set_goal_state(goal_state)
        self._goal = self.render()

        # restore original pos
        if options is not None and "state" in options:
            state = options["state"]
        else:
            state = self.variation_space["agent"]["start_position"].value.tolist()

        self._set_state(state)

        #### OBS
        state = self._get_obs()
        proprio = state

        observation = {"proprio": proprio, "state": state}
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))

        self.latest_action = action

        if self.relative:
            action = self.agent.position + action * self.action_scale
            action = np.clip(action, 0, self.window_size)

        for _ in range(n_steps):
            # Step PD control.
            acceleration = self.k_p * (
                action - self.agent.position
            ) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acceleration * self.dt

            # Step physics.
            self.space.step(self.dt)

        # make the observation
        state = self._get_obs()
        proprio = state
        observation = {"proprio": proprio, "state": state}

        # collect info
        info = self._get_info()

        # compute reward and termination
        terminated, distance = self.eval_state(self.goal_state, state)
        reward = -distance  # the closer the better

        truncated = False
        return observation, reward, terminated, truncated, info

    def eval_state(self, goal_state, cur_state):
        # success if position difference is < 20
        goal_state = np.asarray(goal_state, dtype=np.float64)
        cur_state = np.asarray(cur_state, dtype=np.float64)
        pos_diff = np.linalg.norm(goal_state - cur_state)
        success = pos_diff < 20
        state_dist = np.linalg.norm(goal_state - cur_state)

        return success, state_dist

    def render(self):
        return self._render_frame(self.render_mode)

    def _get_obs(self):
        obs = tuple(self.agent.position)
        return np.array(obs, dtype=np.float64)

    def _get_info(self):
        goal_proprio = self.goal_state

        info = {
            "env_name": self.env_name,
            "pos_agent": np.array(self.agent.position),
            "goal_pose": self.goal_pose,
            "goal_state": self.goal_state,
            "goal_proprio": goal_proprio,
            "goal": self._goal,
        }

        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.variation_space["background"]["color"].value)

        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose (optional).
        render_goal = (
            bool(self.variation_space["rendering"]["render_goal"].value)
            and self.with_target
        )

        # Draw goal as a circle at the goal_state coordinate.
        if render_goal and self.goal_state is not None:
            base_radius = 0.375
            goal_radius = base_radius * self.variation_space["goal"]["scale"].value
            goal = np.asarray(self.goal_state, dtype=np.float64)
            pygame.draw.circle(
                canvas,
                self.variation_space["goal"]["color"].value,
                (int(round(goal[0])), int(round(goal[1]))),
                int(round(goal_radius)),
            )

        # change agent color
        self._set_body_color(
            self.agent, self.variation_space["agent"]["color"].value.tolist()
        )

        # Draw agent.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def _set_body_color(self, body, color):
        color = (
            pygame.Color(*color)
            if not isinstance(color, pygame.Color)
            else color
        )
        for s in body.shapes:
            s.color = color

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        self.agent.position = state

        # Run physics to take effect
        self.space.step(self.dt)

    def _set_goal_state(self, goal_state):
        self.goal_state = goal_state

    def _setup(self):
        ## create the space with physics
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.render_buffer = []

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]

        self.space.add(*walls)

        #### agent ####
        agent_params = {
            "position": self.variation_space["agent"]["start_position"].value.tolist(),
            "angle": self.variation_space["agent"]["angle"].value,
            "scale": self.variation_space["agent"]["scale"].value,
            "color": self.variation_space["agent"]["color"].value.tolist(),
            "shape": self.shapes[self.variation_space["agent"]["shape"].value],
        }

        self.agent = self.add_shape(**agent_params)

        self.goal_pose = np.concatenate(
            [self.variation_space["goal"]["position"].value]
        )

        self.max_score = 50 * 100
        self.success_threshold = 0.95

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def add_circle(self, position, angle=0, scale=1, color="RoyalBlue"):
        base_radius = 0.375
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, base_radius * scale)
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return body

    def add_shape(self, shape, *args, **kwargs):
        # Dispatch method based on the 'shape' parameter. Reach2D only uses the
        # circular agent ('o'); other shapes are intentionally unsupported.
        if shape == "o":
            return self.add_circle(*args, **kwargs)
        raise ValueError(f"Unknown/unsupported shape type for Reach2D: {shape!r}")


if "Reach2D-v0" not in gym.registry:
    gym.register(id="Reach2D-v0", entry_point="src.ppo.envs:Reach2D")
