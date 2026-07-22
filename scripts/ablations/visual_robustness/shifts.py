"""Visual interventions used only by the PushT robustness ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np


BASE_BACKGROUND = (255, 255, 255)
BASE_BLOCK = (119, 136, 153)
BASE_GOAL = (144, 238, 144)
TARGET_BACKGROUND = (48, 51, 59)
TARGET_BLOCK = (230, 120, 40)
TARGET_GOAL = (150, 80, 210)
SHIFT_STRENGTHS = (0.33, 0.67, 1.0)
CHECKER_COLORS = ((215, 227, 240), (175, 196, 216))
CHECKER_SIZE = 12


def _interpolate(source, target, strength):
    value = np.asarray(source) + float(strength) * (
        np.asarray(target) - np.asarray(source)
    )
    return tuple(np.rint(value).astype(np.uint8).tolist())


def _strength_slug(strength):
    return f"{strength:.2f}".rstrip("0").rstrip(".").replace(".", "p")


@dataclass(frozen=True)
class VisualShiftSpec:
    name: str
    background: tuple[int, int, int] = BASE_BACKGROUND
    block: tuple[int, int, int] = BASE_BLOCK
    goal: tuple[int, int, int] = BASE_GOAL
    strength: float = 0.0
    component: str = "clean"
    checkerboard: bool = False
    checker_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = CHECKER_COLORS
    checker_size: int = CHECKER_SIZE

    def __post_init__(self):
        for label, color in (
            ("background", self.background),
            ("block", self.block),
            ("goal", self.goal),
        ):
            if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
                raise ValueError(f"{label} must be an RGB triplet, got {color}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if self.checker_size < 1:
            raise ValueError("checker_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def renderer_init_value(self):
        value = {}
        if self.background != BASE_BACKGROUND and not self.checkerboard:
            value["background"] = {
                "color": np.asarray(self.background, dtype=np.uint8)
            }
        if self.block != BASE_BLOCK:
            value["block"] = {
                "color": np.asarray(self.block, dtype=np.uint8),
                "scale": np.float32(40.0),
                "shape": 2,
                "angle": np.float64(0.0),
                "start_position": np.asarray([400.0, 100.0], dtype=np.float64),
            }
        if self.goal != BASE_GOAL:
            value["goal"] = {
                "color": np.asarray(self.goal, dtype=np.uint8),
                "scale": np.float32(40.0),
                "angle": np.float64(np.pi / 4),
                "position": np.asarray([256.0, 256.0], dtype=np.float64),
            }
        return value or None

    def apply_post_render(self, frame):
        if not self.checkerboard:
            return np.asarray(frame, dtype=np.uint8)
        frame = np.asarray(frame, dtype=np.uint8).copy()
        height, width = frame.shape[:2]
        rows, cols = np.indices((height, width))
        cells = ((rows // self.checker_size) + (cols // self.checker_size)) % 2
        texture = np.empty_like(frame)
        texture[cells == 0] = np.asarray(self.checker_colors[0], dtype=np.uint8)
        texture[cells == 1] = np.asarray(self.checker_colors[1], dtype=np.uint8)
        mask = np.all(frame == np.asarray(BASE_BACKGROUND, dtype=np.uint8), axis=-1)
        frame[mask] = texture[mask]
        return frame


def _build_conditions():
    conditions = {"clean": VisualShiftSpec(name="clean")}
    endpoints = {
        "background": (BASE_BACKGROUND, TARGET_BACKGROUND),
        "block": (BASE_BLOCK, TARGET_BLOCK),
        "goal": (BASE_GOAL, TARGET_GOAL),
    }
    for component, (source, target) in endpoints.items():
        for strength in SHIFT_STRENGTHS:
            kwargs = {component: _interpolate(source, target, strength)}
            name = f"{component}_{_strength_slug(strength)}"
            conditions[name] = VisualShiftSpec(
                name=name,
                component=component,
                strength=strength,
                **kwargs,
            )
    conditions["combined"] = VisualShiftSpec(
        name="combined",
        component="combined",
        strength=1.0,
        background=TARGET_BACKGROUND,
        block=TARGET_BLOCK,
        goal=TARGET_GOAL,
    )
    conditions["texture"] = VisualShiftSpec(
        name="texture",
        component="background_texture",
        strength=1.0,
        checkerboard=True,
    )
    return conditions


CONDITIONS = _build_conditions()
CONDITION_NAMES = tuple(CONDITIONS)
ENDPOINT_CONDITIONS = ("background_1", "block_1", "goal_1", "combined")


def get_condition(name):
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown visual condition {name!r}; choose from {', '.join(CONDITION_NAMES)}"
        ) from exc


class PostRenderShiftWrapper(gym.ObservationWrapper):
    """Apply an ablation-only post-render intervention to image observations."""

    def __init__(self, env, spec):
        super().__init__(env)
        # Gymnasium reserves Wrapper.spec as a read-only EnvSpec proxy.
        self.visual_shift = spec

    def observation(self, observation):
        return self.visual_shift.apply_post_render(observation)

