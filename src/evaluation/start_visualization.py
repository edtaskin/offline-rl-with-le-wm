"""Render the deterministic PushT evaluation suite's initial block poses."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.envs.pusht_wrappers import (
    PUSHT_WORKSPACE_HIGH,
    PUSHT_WORKSPACE_LOW,
    block_center,
    green_t_center,
)
from src.evaluation.pusht import (
    CANONICAL_OOD,
    CANONICAL_OOD_STRATA,
    CANONICAL_V2_STRATA,
    PushTEvalConfig,
    difficulty_stratum,
    make_evaluation_env,
    protocol_strata,
    scalar_metrics,
)


START_VISUALIZATION_RESOLUTION = 512
_UNSTRATIFIED = "all_starts"
_STRATUM_COLORS = {
    "near_aligned": (0, 114, 178),
    "near_misaligned": (86, 180, 233),
    "mid_aligned": (0, 158, 115),
    "mid_misaligned": (230, 159, 0),
    "far_aligned": (204, 121, 167),
    "far_misaligned": (213, 94, 0),
    _UNSTRATIFIED: (0, 114, 178),
}
_STRATUM_COLORS.update(
    {
        ood_label: _STRATUM_COLORS[v2_label]
        for v2_label, ood_label in zip(CANONICAL_V2_STRATA, CANONICAL_OOD_STRATA)
    }
)
_STRATUM_DISPLAY_LABELS = {
    label: label for label in CANONICAL_V2_STRATA
} | {
    label: f"OOD {lower}–{upper}px, {angle}"
    for label, (lower, upper, angle) in zip(
        CANONICAL_OOD_STRATA,
        (
            (200, 220, "aligned"),
            (200, 220, "misaligned"),
            (220, 240, "aligned"),
            (220, 240, "misaligned"),
            (240, 260, "aligned"),
            (240, 260, "misaligned"),
        ),
    )
}


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _episode_seeds(config: PushTEvalConfig) -> tuple[int, ...]:
    if config.episode_seeds is not None:
        return tuple(int(seed) for seed in config.episode_seeds)
    return tuple(
        int(config.seed + index * config.seed_stride)
        for index in range(config.episodes)
    )


def _body_pose_from_reset(info, env) -> np.ndarray:
    pose = np.asarray(info.get("block_pose", ()), dtype=np.float64).reshape(-1)
    if pose.shape == (3,) and np.all(np.isfinite(pose)):
        return pose
    state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64).reshape(-1)
    if state.size < 5:
        raise ValueError("PushT reset did not expose a valid block pose")
    return state[2:5].copy()


def _centroid_pose_from_reset(info, env) -> np.ndarray:
    body_pose = _body_pose_from_reset(info, env)
    center = np.asarray(info.get("block_center", ()), dtype=np.float64).reshape(-1)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        center = np.asarray(block_center(env), dtype=np.float64)
    return np.array([center[0], center[1], body_pose[2]], dtype=np.float64)


def _goal_center_from_reset(info, env) -> np.ndarray:
    center = np.asarray(info.get("green_t_center", ()), dtype=np.float64).reshape(-1)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        center = np.asarray(green_t_center(env), dtype=np.float64)
    return center


def _stratum_from_reset(config, info, expected: str | None) -> str:
    if config.distance_thresholds and config.angle_thresholds:
        observed = difficulty_stratum(
            scalar_metrics(info),
            config.distance_thresholds,
            config.angle_thresholds,
            protocol_strata(config.protocol),
        )
        if expected is not None and observed != expected:
            raise RuntimeError(
                "start visualization reset no longer matches its assigned stratum: "
                f"expected={expected}, observed={observed}"
            )
        return observed
    return expected or _UNSTRATIFIED


def _world_to_image(xy, resolution: int) -> tuple[float, float]:
    xy = np.asarray(xy, dtype=np.float64)
    unit = (xy - PUSHT_WORKSPACE_LOW) / (PUSHT_WORKSPACE_HIGH - PUSHT_WORKSPACE_LOW)
    return float(unit[0] * resolution), float(unit[1] * resolution)


def write_start_location_visualization(
    config: PushTEvalConfig,
    output_path: str | Path,
    *,
    env=None,
    resolution: int = START_VISUALIZATION_RESOLUTION,
) -> Path:
    """Overlay every deterministic block start on an averaged PushT render.

    Resetting and rendering the exact episode seeds is policy-independent. The
    averaged background preserves the static workspace and fixed goal while
    fading the individual agent/block bodies that move between reset frames.
    """

    config.validate()
    if resolution < 64:
        raise ValueError("start visualization resolution must be at least 64")

    seeds = _episode_seeds(config)
    expected_strata = config.episode_strata or (None,) * len(seeds)
    owns_env = env is None
    if env is None:
        render_config = replace(config, observation_resolution=int(resolution))
        env = make_evaluation_env(
            render_config,
            render_observations=False,
            record_statistics=False,
        )

    records = []
    background_sum = None
    goal_center_xy = None
    try:
        for seed, expected in zip(seeds, expected_strata):
            _, info = env.reset(seed=int(seed))
            pose = _centroid_pose_from_reset(info, env)
            reset_goal_center = _goal_center_from_reset(info, env)
            if goal_center_xy is None:
                goal_center_xy = reset_goal_center
            elif not np.allclose(goal_center_xy, reset_goal_center, atol=1e-6):
                raise RuntimeError("goal centroid changed between deterministic resets")
            label = _stratum_from_reset(config, info, expected)
            frame = np.asarray(env.render(), dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"PushT render must be RGB, got shape {frame.shape}")
            if background_sum is None:
                background_sum = np.zeros_like(frame, dtype=np.float64)
            if frame.shape != background_sum.shape:
                raise ValueError("PushT render shape changed between deterministic resets")
            background_sum += frame
            records.append((int(seed), pose, label))
    finally:
        if owns_env:
            env.close()

    if not records or background_sum is None or goal_center_xy is None:
        raise ValueError("cannot visualize an empty evaluation suite")

    background = np.rint(background_sum / len(records)).clip(0, 255).astype(np.uint8)
    background_image = Image.fromarray(background).resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    )
    background_image = Image.blend(
        background_image,
        Image.new("RGB", background_image.size, "white"),
        0.20,
    )

    header_height = 48
    observed_labels = {record[2] for record in records}
    labels = [
        label for label in protocol_strata(config.protocol) if label in observed_labels
    ]
    if not labels:
        labels = [_UNSTRATIFIED]
    legend_columns = 1 if config.protocol == CANONICAL_OOD else 2
    legend_rows = (len(labels) + legend_columns - 1) // legend_columns
    footer_height = 66 + legend_rows * 26
    canvas = Image.new("RGB", (resolution, header_height + resolution + footer_height), "white")
    canvas.paste(background_image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"Deterministic block starts ({len(records)} episodes)",
        fill=(20, 20, 20),
        font=_font(22),
    )
    workspace_overlay = Image.new(
        "RGBA", (resolution, resolution), (0, 0, 0, 0)
    )
    workspace_draw = ImageDraw.Draw(workspace_overlay)

    goal_xy = _world_to_image(goal_center_xy, resolution)
    scale = resolution / float(PUSHT_WORKSPACE_HIGH[0] - PUSHT_WORKSPACE_LOW[0])
    ring_specs = [(float(radius), False) for radius in config.distance_thresholds]
    if config.block_start_min_radius > 0 and not any(
        np.isclose(config.block_start_min_radius, radius) for radius, _ in ring_specs
    ):
        ring_specs.append((float(config.block_start_min_radius), True))
    if config.block_start_radius is not None and not any(
        np.isclose(config.block_start_radius, radius) for radius, _ in ring_specs
    ):
        ring_specs.append((float(config.block_start_radius), True))
    for radius, is_cap in ring_specs:
        radius_px = float(radius) * scale
        bounds = (
            goal_xy[0] - radius_px,
            goal_xy[1] - radius_px,
            goal_xy[0] + radius_px,
            goal_xy[1] + radius_px,
        )
        workspace_draw.ellipse(
            bounds,
            outline=(25, 25, 25) if is_cap else (90, 90, 90),
            width=3 if is_cap else 2,
        )

    cross_radius = max(4, int(round(resolution / 100)))
    ray_length = max(9, int(round(resolution / 45)))
    for _, pose, label in records:
        x, y = _world_to_image(pose[:2], resolution)
        color = _STRATUM_COLORS[label]
        cross_lines = (
            (x - cross_radius, y - cross_radius, x + cross_radius, y + cross_radius),
            (x - cross_radius, y + cross_radius, x + cross_radius, y - cross_radius),
        )
        for line in cross_lines:
            workspace_draw.line(line, fill=(20, 20, 20), width=5)
            workspace_draw.line(line, fill=color, width=3)
        endpoint = (
            x + ray_length * float(np.cos(pose[2])),
            y + ray_length * float(np.sin(pose[2])),
        )
        workspace_draw.line((x, y, *endpoint), fill=(20, 20, 20), width=4)
        workspace_draw.line((x, y, *endpoint), fill=color, width=2)

    goal_x, goal_y = goal_xy
    goal_radius = max(6, int(round(resolution / 64)))
    workspace_draw.ellipse(
        (
            goal_x - goal_radius,
            goal_y - goal_radius,
            goal_x + goal_radius,
            goal_y + goal_radius,
        ),
        fill=(144, 238, 144),
        outline=(20, 20, 20),
        width=3,
    )
    workspace_draw.text(
        (goal_x + goal_radius + 4, goal_y - goal_radius),
        "goal",
        fill=(20, 20, 20),
        font=_font(12),
    )
    plotted_workspace = Image.alpha_composite(
        background_image.convert("RGBA"), workspace_overlay
    ).convert("RGB")
    canvas.paste(plotted_workspace, (0, header_height))

    counts = Counter(label for _, _, label in records)
    footer_y = header_height + resolution + 8
    threshold_text = "/".join(str(int(value)) for value in config.distance_thresholds)
    detail = "X = block centroid; ray = orientation; green dot = goal centroid"
    draw.text((12, footer_y), detail, fill=(35, 35, 35), font=_font(13))
    if threshold_text:
        ring_text = f"rings = {threshold_text}px strata"
        if config.block_start_min_radius > 0 and config.block_start_radius is not None:
            ring_text += (
                f"; bounds = {int(config.block_start_min_radius)}-"
                f"{int(config.block_start_radius)}px annulus"
            )
        elif config.block_start_radius is not None:
            ring_text += f"; outer = {int(config.block_start_radius)}px sampling cap"
        draw.text(
            (12, footer_y + 18),
            ring_text,
            fill=(35, 35, 35),
            font=_font(13),
        )
    for index, label in enumerate(labels):
        column = index % legend_columns
        row = index // legend_columns
        x = 12 + column * (resolution // legend_columns)
        y = footer_y + 45 + row * 26
        color = _STRATUM_COLORS[label]
        draw.line((x, y + 7, x + 16, y + 7), fill=(20, 20, 20), width=7)
        draw.line((x, y + 7, x + 16, y + 7), fill=color, width=5)
        draw.text(
            (x + 24, y),
            f"{_STRATUM_DISPLAY_LABELS.get(label, label)} (n={counts[label]})",
            fill=(25, 25, 25),
            font=_font(14),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved start-location visualization to: {output_path}")
    return output_path
