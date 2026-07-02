import gymnasium as gym
import numpy as np
from gymnasium import spaces


PUSHT_ENV_ID = "swm/PushT-v1"
PUSHT_RENDER_SHAPE = (96, 96, 3)


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
    **kwargs,
):
    kwargs.setdefault("resolution", PUSHT_RENDER_SHAPE[0])
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if render_obs:
        env = PushTRenderObservationWrapper(env)
    return env
