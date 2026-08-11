"""What the policy imagines, next to what actually happens.

Takes held-out expert frames as start states, lets a policy roll forward inside
:class:`~src.ppo.train_lewm.LeWMDreamWorld` -- one action chunk per predictor
step, executed open loop -- and then replays *those exact chunks* in the
simulator from the same start state. Every imagined frame therefore has the real
frame it was supposed to be, and the two are drawn side by side.

This is the policy-driven counterpart to the existing fidelity tools, which all
drive LeWM with ground-truth expert actions:
``scripts/decoder/decode_rollouts_pusht.py`` (decoded rollouts against dataset
frames), ``scripts/deprojector/bridge_horizon.py`` (latent drift per bridge) and
``scripts/rq2/probe_horizon.py`` (probe reliability). Here the actions are the
ones the agent would actually take, so what shows up is the drift the agent
itself walks into -- including states it learned to steer toward *because* the
world model is wrong about them.

Reading the panels:

* the imagined row is a decoded latent, so it carries the decoder's own
  reconstruction loss on top of any world-model error. ``--decoder-floor`` adds a
  third row -- the *simulator* frame encoded and decoded again -- which is that
  loss with no imagination in it. Pixel differences above that floor are drift.
* the simulator row stops on real task success and freezes; imagination keeps
  going, and its caption keeps reporting what the frozen success probe believes.
  A green probe reading beside a frozen, clearly-unsolved simulator frame is a
  hallucinated success.

Usage::

    python -m scripts.rq2.imagined_vs_simulated                       # 4 start states
    python -m scripts.rq2.imagined_vs_simulated --seed 7 --episodes 6
    python -m scripts.rq2.imagined_vs_simulated --agent bc --decoder-floor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# PushT renders through pygame; this script never wants a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.rq_common import INK, SERIES_COLORS, repo_path
from src.evaluation.agents import load_ppo_components, resolve_device
from src.ppo.env import LatentHistory
from src.ppo.train_lewm import DreamConfig, LeWMDreamWorld

# The dream-PPO policy and the BC prior the blog post compares.
PPO_CHECKPOINT = (
    "hf://offline-rl-with-le-wm/ppo/rawcls_bc_best/dream_dense_dense02_10m/seed1/final.pt"
)
BC_CHECKPOINT = "hf://offline-rl-with-le-wm/bc/rawcls-bc/seed42/pusht_raw_cls_bc_best.pth"

# RQ2 colour semantics (see scripts/rq_common.py): imagined is the dream's
# colour, the simulator is the real environment's.
IMAGINED_COLOR = SERIES_COLORS["dream"]
REAL_COLOR = SERIES_COLORS["real"]
FLOOR_COLOR = INK["muted"]
SUCCESS_COLOR, FAILURE_COLOR = "#0ca30c", "#d03b3b"

ROW_TITLES = {
    "imagined": "imagined (LeWM)",
    "simulator": "simulator",
    "floor": "decode floor",
}
ROW_COLORS = {"imagined": IMAGINED_COLOR, "simulator": REAL_COLOR, "floor": FLOOR_COLOR}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", choices=["ppo", "bc"], default="ppo")
    parser.add_argument("--checkpoint", default=None, help="defaults to the arm's published policy")
    parser.add_argument("--bc-stats", default=None, help="BC stats; inferred when omitted")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--episodes", type=int, default=4, help="start states to compare")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="predictor steps to imagine (default: the run's dream episode length)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="draws the held-out anchors; change it to sample different start states",
    )
    parser.add_argument(
        "--decoder-floor",
        action="store_true",
        help="add a row that encodes and decodes the simulator frame, isolating decoder loss",
    )

    parser.add_argument("--strip-steps", type=int, default=8, help="columns in the filmstrip")
    parser.add_argument("--tile", type=int, default=180, help="filmstrip cell size, px")
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write a two-panel animation per start state",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--anim-tile", type=int, default=320, help="animation panel size, px")
    parser.add_argument("--gif-colors", type=int, default=128)
    parser.add_argument("--output-dir", default="runs/rq2/imagined_vs_simulated")
    return parser


# ----------------------------------------------------------------- dream side
def build_world_and_policy(args):
    """Rebuild the dream world the checkpoint was trained in, plus its policy."""
    from dataclasses import fields

    device = resolve_device(args.device)
    checkpoint = args.checkpoint or (PPO_CHECKPOINT if args.agent == "ppo" else BC_CHECKPOINT)

    if args.agent == "ppo":
        from src.utils.hf_hub import resolve_artifact

        payload = torch.load(resolve_artifact(checkpoint), map_location="cpu", weights_only=False)
        saved = payload.get("config", {})
        if "wm_frameskip" not in saved:
            raise SystemExit(
                f"{checkpoint} was not produced by src.ppo.train_lewm; this script "
                "compares a dream-trained policy against the world it was trained in."
            )
        # Derived config fields (batch_size, num_iterations, ...) are init=False.
        init_names = {field_.name for field_ in fields(DreamConfig) if field_.init}
        cfg = DreamConfig(**{key: value for key, value in saved.items() if key in init_names})
    else:
        # No dream run to inherit from: the defaults are the campaign's dream
        # settings, which is the world every arm was trained in.
        cfg = DreamConfig(exp_name="imagined_vs_simulated", frame_stack=3, frame_stride=5)

    cfg.num_envs = args.episodes
    # Rewards never feed back into the predictor, so the imagined frames do not
    # depend on them; sparse keeps the dense-reward classifier out of the run.
    cfg.reward_mode = "sparse"

    world = LeWMDreamWorld(cfg, device)
    # A de-projector run's policy path never renders anything; this script exists
    # to look at the frames, so they are always decoded.
    world.capture_frames = True

    if args.agent == "ppo":
        # The policy must consume exactly the latents the dream world produces,
        # from the same frozen ViT, as it did during training. Passing the
        # world's encoder also avoids the checkpoint's recorded encoder path,
        # which belongs to the machine the run was trained on.
        components = load_ppo_components(checkpoint, device, encoder=world.cls_encoder)

        def policy(stacked):
            # The distribution mean, i.e. the policy that would be deployed.
            return components.agent.actor.bc_policy(stacked)
    else:
        from scripts.deprojector.bridge_horizon import load_raw_cls_bc_policy
        from src.evaluation.agents import infer_bc_stats_path

        stats = args.bc_stats or infer_bc_stats_path(checkpoint)
        bc_policy, _ = load_raw_cls_bc_policy(checkpoint, stats, device)
        policy = bc_policy

    return cfg, world, policy, device, checkpoint


@dataclass
class Episode:
    """One start state, imagined and then re-run in the simulator."""

    index: int
    anchor: dict
    actions: list = field(default_factory=list)
    imagined: list = field(default_factory=list)
    success_prob: list = field(default_factory=list)
    declared_step: int | None = None

    anchor_frame: np.ndarray | None = None
    anchor_pixel_error: float = float("nan")
    sim_chunks: list = field(default_factory=list)
    sim_frames: list = field(default_factory=list)
    sim_success_step: int | None = None
    sim_final_metrics: dict = field(default_factory=dict)
    pixel_error: list = field(default_factory=list)
    floor_frames: list = field(default_factory=list)
    floor_error: list = field(default_factory=list)

    def summary(self):
        return {
            "start_state": {
                "expert_episode": int(self.anchor["episode"]),
                "dataset_row": int(self.anchor["row"]),
                "anchor_pixel_error": float(self.anchor_pixel_error),
            },
            "imagined_success_step": self.declared_step,
            "simulator_success_step": self.sim_success_step,
            "hallucinated_success": bool(
                self.declared_step is not None and self.sim_success_step is None
            ),
            "success_prob": [float(value) for value in self.success_prob],
            "pixel_error": [float(value) for value in self.pixel_error],
            "decoder_floor_error": [float(value) for value in self.floor_error],
            "simulator_final_metrics": {
                key: float(value) for key, value in self.sim_final_metrics.items()
            },
        }


def to_uint8(frames):
    """``[..., 3, H, W]`` float RGB in [0, 1] -> ``[..., H, W, 3]`` uint8."""
    array = frames.detach().cpu().clamp(0, 1).movedim(-3, -1).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def imagine(world, cfg, policy, args, horizon):
    """Roll every dream env ``horizon`` predictor steps from a held-out anchor.

    Unlike the trainer, a finished episode is *not* re-seeded: the point is to
    watch one start state drift, including past the step where the probe first
    declares success.
    """
    n = cfg.num_envs
    episodes = []
    with world.eval_mode(args.seed, horizon):
        histories = [LatentHistory(cfg.frame_stack, 1) for _ in range(n)]
        for i in range(n):
            context = world.reset_env(i)
            for t in range(context.shape[0]):
                histories[i].append(context[t])
            episodes.append(Episode(index=i, anchor=dict(world.last_anchor[i])))

        for step in range(horizon):
            stacked = torch.stack([history.stacked() for history in histories], dim=0)
            action = torch.clamp(policy(stacked), -1.0, 1.0)
            cls, _, terminated, _, _ = world.step(action)
            frames = to_uint8(world.last_frames)
            probs = (
                world.last_success_prob.detach().cpu().numpy()
                if world.last_success_prob is not None
                else np.full(n, np.nan)
            )
            for i, episode in enumerate(episodes):
                histories[i].append(cls[i])
                episode.actions.append(action[i].detach().cpu().numpy())
                episode.imagined.append(frames[i])
                episode.success_prob.append(float(probs[i]))
                if terminated[i] and episode.declared_step is None:
                    episode.declared_step = step + 1
    for episode in episodes:
        declared = episode.declared_step
        print(
            f"  start {episode.index}: expert episode {episode.anchor['episode']} "
            f"row {episode.anchor['row']} | probe declared success at step "
            f"{declared if declared is not None else '-'}"
        )
    return episodes


# ------------------------------------------------------------- simulator side
def replay_in_simulator(episodes, cfg, world, args, horizon):
    """Drive the simulator from each anchor with the chunks imagination produced.

    The dream episode was seeded from a known dataset frame, so the simulator can
    be put into that exact state; the rendered frame is checked against the
    dataset's own pixels before anything is compared, because every panel below
    is meaningless if the two do not start from the same place.
    """
    import h5py

    from src.envs import make_pusht_env

    frameskip = cfg.wm_frameskip
    env = make_pusht_env(
        env_id=cfg.env_id,
        max_episode_steps=horizon * frameskip,
        align_sampled_goal_to_fixed_target=True,
        fixed_target_block_success=True,
        block_start_near_goal=False,
    )
    unwrapped = env.unwrapped
    try:
        with h5py.File(repo_path(cfg.dataset_path), "r") as h5:
            for episode in episodes:
                env.reset(seed=args.seed + episode.index)
                state = np.asarray(episode.anchor["state"], dtype=np.float64)
                full_state = np.asarray(unwrapped._get_obs(), dtype=np.float64).copy()
                full_state[: len(state)] = state
                unwrapped._set_state(full_state)

                start = np.asarray(env.render(), dtype=np.uint8)
                reference = np.asarray(h5["pixels"][episode.anchor["row"]], dtype=np.uint8)
                episode.anchor_frame = reference
                if reference.shape == start.shape:
                    episode.anchor_pixel_error = float(
                        np.abs(start.astype(np.float32) - reference.astype(np.float32)).mean()
                    )
                episode.sim_frames = [start]

                terminated = truncated = False
                for step, chunk in enumerate(episode.actions, start=1):
                    for action in chunk:
                        if terminated or truncated:
                            break
                        observation, _, terminated, truncated, info = env.step(
                            np.clip(action, env.action_space.low, env.action_space.high)
                        )
                        episode.sim_frames.append(np.asarray(observation, dtype=np.uint8))
                        if terminated and episode.sim_success_step is None:
                            episode.sim_success_step = step
                            episode.sim_final_metrics = {
                                key: float(value)
                                for key, value in info.items()
                                if key
                                in ("block_state_dist", "block_pos_dist", "block_angle_dist")
                                and np.isscalar(value)
                            }
                    # Frozen on the terminal frame once the task is solved, which
                    # is what the caption says happened.
                    episode.sim_chunks.append(episode.sim_frames[-1])
                if not episode.sim_final_metrics:
                    episode.sim_final_metrics = {
                        key: float(value)
                        for key, value in info.items()
                        if key in ("block_state_dist", "block_pos_dist", "block_angle_dist")
                        and np.isscalar(value)
                    }

                episode.pixel_error = [
                    float(np.abs(imagined.astype(np.float32) - real.astype(np.float32)).mean())
                    for imagined, real in zip(episode.imagined, episode.sim_chunks)
                ]
                if args.decoder_floor:
                    episode.floor_frames, episode.floor_error = decode_floor(world, episode)
                print(
                    f"  start {episode.index}: anchor match {episode.anchor_pixel_error:5.2f}/255 "
                    f"| simulator success at step "
                    f"{episode.sim_success_step if episode.sim_success_step is not None else '-'} "
                    f"| mean |imagined - real| {np.mean(episode.pixel_error):5.1f}/255"
                )
    finally:
        env.close()

    worst = max(
        (episode.anchor_pixel_error for episode in episodes if np.isfinite(episode.anchor_pixel_error)),
        default=0.0,
    )
    if worst > 12.0:
        print(
            f"WARNING: worst anchor match is {worst:.1f}/255 -- the simulator may not be "
            "starting from the dataset state, which would invalidate every comparison below."
        )
    return episodes


@torch.no_grad()
def decode_floor(world, episode):
    """Simulator frame -> ViT -> projector -> decoder: the decoder's own error.

    The imagined row is a decoded latent, so it can never be sharper than this.
    Subtracting this floor from the imagined error is what separates "the decoder
    is lossy" from "the world model is wrong".

    Frames are returned for the start state *and* every chunk, so the floor row
    lines up column for column with the other two; the errors cover the chunks
    only, so they line up with ``pixel_error``.
    """
    frames = [episode.sim_frames[0]] + list(episode.sim_chunks)
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(world.device)
    latent = world.wm.projector(world.cls_encoder(batch))
    decoded = to_uint8(world.decoder(latent).clamp(0.0, 1.0))
    error = [
        float(np.abs(decoded[i].astype(np.float32) - frames[i].astype(np.float32)).mean())
        for i in range(1, len(frames))
    ]
    return list(decoded), error


# ------------------------------------------------------------------- drawing
_FONTS = {}


def font(size):
    """A real TTF at a chosen size (PIL's built-in bitmap font is fixed at ~11px)."""
    if size not in _FONTS:
        from PIL import ImageFont

        try:
            from matplotlib import font_manager

            _FONTS[size] = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:  # noqa: BLE001 - any failure just means the default font
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def strip_columns(args, horizon):
    """Time indices to show: 0 is the shared start, then imagined steps 1..H."""
    columns = np.unique(np.linspace(0, horizon, max(2, args.strip_steps)).astype(int))
    return [int(value) for value in columns]


def cell_frame(episode, row, column):
    """The frame for one filmstrip cell, or ``None`` when that row has none."""
    if column == 0:
        # Imagination has not run yet: every row shows the same real start state.
        return {
            "imagined": episode.anchor_frame,
            "simulator": episode.sim_frames[0],
            "floor": episode.floor_frames[0] if episode.floor_frames else None,
        }[row]
    step = column - 1
    return {
        "imagined": episode.imagined[step],
        "simulator": episode.sim_chunks[step],
        # floor_frames[0] is the start state, so a chunk sits one index later.
        "floor": episode.floor_frames[column] if episode.floor_frames else None,
    }[row]


def column_caption(episode, column, cfg):
    """One short line per fact, each narrow enough to stay inside its column."""
    if column == 0:
        return [
            ("start state", INK["secondary"]),
            ("shared by both", INK["muted"]),
            (f"anchor {episode.anchor_pixel_error:.1f}/255", INK["muted"]),
            ("", INK["muted"]),
        ]
    step = column - 1
    declared = episode.declared_step is not None and column >= episode.declared_step
    solved = episode.sim_success_step is not None and column >= episode.sim_success_step
    return [
        (f"+{column * cfg.wm_frameskip} env steps", INK["secondary"]),
        (f"|Δ| {episode.pixel_error[step]:.0f}/255", INK["muted"]),
        (
            "probe: SUCCESS" if declared else f"probe p={episode.success_prob[step]:.2f}",
            FAILURE_COLOR if declared and not solved else IMAGINED_COLOR,
        ),
        ("real: solved" if solved else "real: unsolved", SUCCESS_COLOR if solved else INK["muted"]),
    ]


def filmstrip(episode, cfg, args, horizon, path):
    """One start state: rows are imagined / simulator, columns are time."""
    columns = strip_columns(args, horizon)
    rows = ["imagined", "simulator"] + (["floor"] if args.decoder_floor else [])
    tile, pad, gap = args.tile, 16, 10
    header, caption = 62, 74
    # Sized to the longest row label rather than fixed: "imagined (LeWM)" is
    # wider than a guessed column and would be clipped by the first tile.
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = 12 + max(
        int(measure.textlength(ROW_TITLES[name], font=font(13))) for name in rows
    )

    width = label_width + len(columns) * tile + (len(columns) - 1) * gap + 2 * pad
    height = header + len(rows) * (tile + gap) + caption + pad
    canvas = Image.new("RGB", (width, height), INK["surface"])
    draw = ImageDraw.Draw(canvas)

    hallucinated = episode.declared_step is not None and episode.sim_success_step is None
    title = (
        f"Open-loop chunks from expert episode {episode.anchor['episode']}, "
        f"row {episode.anchor['row']}"
    )
    draw.text((pad, 12), title, fill=INK["primary"], font=font(19))
    subtitle = (
        f"same start state · one chunk of {cfg.action_chunk_size} actions per "
        f"predictor step = {cfg.wm_frameskip} env steps · "
        "|Δ| is the mean per-pixel gap between the two rows"
    )
    if hallucinated:
        subtitle += "  ·  probe declared a success the simulator never reached"
    draw.text(
        (pad, 38),
        subtitle,
        fill=FAILURE_COLOR if hallucinated else INK["secondary"],
        font=font(13),
    )

    for row_index, name in enumerate(rows):
        top = header + row_index * (tile + gap)
        draw.text((pad, top + tile // 2 - 8), ROW_TITLES[name], fill=ROW_COLORS[name], font=font(13))
        for column_index, column in enumerate(columns):
            left = pad + label_width + column_index * (tile + gap)
            frame = cell_frame(episode, name, column)
            if frame is None:
                continue
            canvas.paste(Image.fromarray(frame).resize((tile, tile), Image.LANCZOS), (left, top))
            draw.rectangle(
                [left, top, left + tile - 1, top + tile - 1], outline=ROW_COLORS[name], width=2
            )

    bottom = header + len(rows) * (tile + gap)
    for column_index, column in enumerate(columns):
        left = pad + label_width + column_index * (tile + gap)
        for line_index, (line, color) in enumerate(column_caption(episode, column, cfg)):
            draw.text((left, bottom + 2 + line_index * 16), line, fill=color, font=font(12))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f"  wrote {path}")
    return path


def animation_frames(episode, cfg, args):
    """Two panels in lockstep: imagination updates once per chunk, the sim every step.

    The imagined panel holding still for ``wm_frameskip`` frames is not a
    rendering shortcut -- it is the model's actual temporal resolution.
    """
    tile, pad, gap = args.anim_tile, 16, 16
    label, caption, footer = 30, 28, 24
    width = 2 * tile + 2 * pad + gap
    width += width % 2
    height = pad + label + tile + caption + footer + pad
    height += height % 2

    chrome = Image.new("RGB", (width, height), INK["surface"])
    draw = ImageDraw.Draw(chrome)
    for index, name in enumerate(("imagined", "simulator")):
        left = pad + index * (tile + gap)
        draw.text((left, pad - 4), ROW_TITLES[name], fill=ROW_COLORS[name], font=font(17))
        underline = pad + label - 7
        draw.rectangle([left, underline, left + tile - 1, underline + 2], fill=ROW_COLORS[name])
    draw.text(
        (pad, height - pad - footer + 4),
        f"expert episode {episode.anchor['episode']}, row {episode.anchor['row']}  ·  "
        f"chunks of {cfg.action_chunk_size} actions, executed open loop",
        fill=INK["muted"],
        font=font(12),
    )

    frames = []
    top = pad + label
    # Imagination always runs the full horizon; the simulator may have stopped
    # early on real success, so its panel freezes on the terminal frame while the
    # imagined one keeps going. That divergence is the thing worth seeing.
    total_env_steps = len(episode.imagined) * cfg.wm_frameskip
    for env_step in range(total_env_steps + 1):
        chunk_index = max(0, (env_step - 1) // cfg.wm_frameskip)
        canvas = chrome.copy()
        cell = ImageDraw.Draw(canvas)
        panels = (
            episode.anchor_frame if env_step == 0 else episode.imagined[chunk_index],
            episode.sim_frames[min(env_step, len(episode.sim_frames) - 1)],
        )
        for index, (name, frame) in enumerate(zip(("imagined", "simulator"), panels)):
            left = pad + index * (tile + gap)
            canvas.paste(Image.fromarray(frame).resize((tile, tile), Image.LANCZOS), (left, top))
            cell.rectangle([left, top, left + tile - 1, top + tile - 1], outline=INK["grid"])
        step = 0 if env_step == 0 else chunk_index + 1
        declared = episode.declared_step is not None and step >= episode.declared_step
        solved = episode.sim_success_step is not None and step >= episode.sim_success_step
        captions = [
            (
                "start state" if env_step == 0 else f"probe p={episode.success_prob[chunk_index]:.2f}",
                FAILURE_COLOR if declared else INK["secondary"],
            ),
            (
                f"env step {env_step}" + ("  ·  solved" if solved else ""),
                SUCCESS_COLOR if solved else INK["secondary"],
            ),
        ]
        for index, (text, color) in enumerate(captions):
            cell.text((pad + index * (tile + gap), top + tile + 6), text, fill=color, font=font(14))
        frames.append(np.asarray(canvas))
    return frames


def write_gif(frames, path, fps, colors):
    """GIF with one palette sampled across the clip, so late captions keep their colour."""
    images = [Image.fromarray(frame) for frame in frames]
    sample = np.concatenate(
        [frames[index] for index in np.unique(np.linspace(0, len(frames) - 1, 8).astype(int))],
        axis=0,
    )
    palette = Image.fromarray(sample).quantize(colors=colors, method=Image.MEDIANCUT)
    quantized = [image.quantize(palette=palette, dither=Image.NONE) for image in images]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    return path


def write_mp4(frames, path, fps):
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1, quality=8)
    return path


# ---------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    cfg, world, policy, _device, checkpoint = build_world_and_policy(args)
    horizon = int(args.horizon or cfg.dream_eval_steps or cfg.dream_episode_steps)
    output_dir = repo_path(args.output_dir) / f"{args.agent}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{args.agent} policy | {args.episodes} start states | horizon {horizon} predictor "
        f"steps ({horizon * cfg.wm_frameskip} env steps) | bridge {cfg.bridge}"
    )

    try:
        print("Imagining...")
        episodes = imagine(world, cfg, policy, args, horizon)
        print("Replaying the same chunks in the simulator...")
        replay_in_simulator(episodes, cfg, world, args, horizon)
    finally:
        world.close()

    print("Drawing...")
    artifacts = []
    for episode in episodes:
        stem = f"start{episode.index:02d}_ep{episode.anchor['episode']}_row{episode.anchor['row']}"
        files = [filmstrip(episode, cfg, args, horizon, output_dir / f"{stem}.png")]
        if args.animate:
            frames = animation_frames(episode, cfg, args)
            files.append(write_gif(frames, output_dir / f"{stem}.gif", args.fps, args.gif_colors))
            files.append(write_mp4(frames, output_dir / f"{stem}.mp4", args.fps))
            print(f"  wrote {files[-2]} and {files[-1].name}")
        artifacts.append({**episode.summary(), "files": [str(path) for path in files]})

    errors = np.array([episode.pixel_error for episode in episodes], dtype=float)
    per_step = errors.mean(axis=0)
    hallucinated = sum(item["hallucinated_success"] for item in artifacts)
    print(
        f"\nmean |imagined - real| by predictor step: "
        + ", ".join(f"{value:.0f}" for value in per_step)
    )
    print(
        f"{hallucinated}/{len(episodes)} start states ended with a probe-declared success "
        "the simulator never reached"
    )

    manifest = {
        "settings": vars(args),
        "checkpoint": checkpoint,
        "horizon_predictor_steps": horizon,
        "wm_frameskip": cfg.wm_frameskip,
        "bridge": cfg.bridge,
        "mean_pixel_error_by_step": [float(value) for value in per_step],
        "mean_decoder_floor_by_step": (
            [float(value) for value in np.mean([e.floor_error for e in episodes], axis=0)]
            if args.decoder_floor
            else []
        ),
        "hallucinated_successes": int(hallucinated),
        "episodes": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
