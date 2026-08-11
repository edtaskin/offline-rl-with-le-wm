"""Side-by-side BC vs dream-PPO rollouts for the project blog teaser.

Rolls the BC prior and the dream-PPO policy through the *same* PushT episodes --
same seed, same reset, therefore the same start state down to the pixel -- and
writes a two-panel animation of the pair that shows PPO improving on the prior.

The episode suite comes from one master ``--seed`` through the canonical
:func:`sample_episode_seeds`, so re-running with a different seed re-samples a
different set of episodes and nothing else changes. Candidate episodes are first
rolled out without frames (cheap) and ranked; only the clips that get rendered
are replayed with frame capture, which keeps a 300-step 512px rollout out of
memory for every episode that is not used.

Ranking prefers, in order: episodes only PPO solves (the teaser case), episodes
both solve with PPO closer/faster, episodes neither solves. Within a group the
tie-break is how much closer to the goal PPO leaves the block.

Usage::

    python scripts/make_teaser_rollouts.py                      # seed 42, 2 clips
    python scripts/make_teaser_rollouts.py --seed 7 --clips 3
    python scripts/make_teaser_rollouts.py --episode-seeds 1234567   # one known-good episode

The panels are re-rendered from the simulator at ``--render-resolution`` rather
than upscaled from the 224px policy observation: PushT draws on a 512px canvas
and downsamples at the end, so a larger render is the same state at less loss --
this is checked against the actual observation on every episode.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from scripts.rq_common import CANONICAL_EVAL, INK, SERIES_COLORS, repo_path
from src.evaluation.agents import (
    load_bc_components,
    load_ppo_components,
    make_bc_evaluation_agent,
    make_ppo_evaluation_agent,
)
from src.evaluation.evaluate_pusht import sample_episode_seeds
from src.evaluation.pusht import PushTEvalConfig, make_evaluation_env, success_from_info


# The BC prior and the dream-PPO policy the blog post is about.
BC_CHECKPOINT = "hf://offline-rl-with-le-wm/bc/rawcls-bc/seed42/pusht_raw_cls_bc_best.pth"
PPO_CHECKPOINT = (
    "hf://offline-rl-with-le-wm/ppo/rawcls_bc_best/dream_dense_dense02_10m/seed1/final.pt"
)

PANELS = ("bc", "ppo")
PANEL_TITLES = {"bc": "BC policy", "ppo": "Dream-PPO policy"}
PANEL_COLORS = {"bc": SERIES_COLORS["bc"], "ppo": SERIES_COLORS["dream_ppo"]}
SUCCESS_COLOR, FAILURE_COLOR = "#0ca30c", "#d03b3b"

# Ranking groups, best teaser material first.
PPO_ONLY, BOTH, NEITHER, BC_ONLY = 0, 1, 2, 3
CATEGORY_LABELS = {
    PPO_ONLY: "PPO only",
    BOTH: "both",
    NEITHER: "neither",
    BC_ONLY: "BC only",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bc-checkpoint", default=BC_CHECKPOINT)
    parser.add_argument(
        "--bc-stats", default=None, help="inferred from the BC checkpoint when omitted"
    )
    parser.add_argument("--ppo-checkpoint", default=PPO_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--execution-mode",
        choices=["open-loop", "receding-horizon", "temporal-ensemble"],
        default="open-loop",
        help="chunk execution; the canonical evaluation protocol is open-loop",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_EVAL["seed"],
        help="master seed the candidate episodes are sampled from (change this to re-roll)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=16,
        help="episodes to try before picking the clips to render",
    )
    parser.add_argument("--clips", type=int, default=2, help="side-by-side clips to render")
    parser.add_argument(
        "--episode-seeds",
        type=int,
        nargs="+",
        default=None,
        help="explicit episode seeds; skips sampling (use to re-render a known-good episode)",
    )
    parser.add_argument(
        "--only-ppo-wins",
        action="store_true",
        help="render nothing but episodes PPO solves and BC does not",
    )

    parser.add_argument(
        "--block-start-radius", type=float, default=CANONICAL_EVAL["block_start_radius"]
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=CANONICAL_EVAL["max_episode_steps"]
    )
    parser.add_argument("--observation-resolution", type=int, default=224)
    parser.add_argument(
        "--allow-resolution-mismatch",
        action="store_true",
        help="evaluate at a resolution the policies were not trained at",
    )

    parser.add_argument(
        "--render-resolution",
        type=int,
        default=512,
        help="simulator re-render size for the panels (512 is PushT's native canvas)",
    )
    parser.add_argument("--tile", type=int, default=320, help="panel size in the output, px")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="keep every Nth env step; raise it for shorter, smaller clips",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="freeze on the final frame for this long (half of it on the first frame)",
    )
    parser.add_argument("--gif-colors", type=int, default=128, help="GIF palette size")
    parser.add_argument("--formats", nargs="+", choices=["gif", "mp4"], default=["gif", "mp4"])
    parser.add_argument("--output-dir", default="runs/teaser")
    return parser


# --------------------------------------------------------------------- agents
def build_agents(args):
    """Load both policies, sharing one frozen LeWM encoder.

    Sharing is not only an optimization here: PPO checkpoints record the
    ``encoder_checkpoint`` path of the machine they were trained on, which does
    not exist on any other machine, and ``load_ppo_components`` would try to load
    it. Handing it an encoder built from the local checkpoint skips that path.
    """
    bc_components = load_bc_components(args.bc_checkpoint, args.bc_stats, args.device)
    try:
        ppo_components = load_ppo_components(
            args.ppo_checkpoint, args.device, encoder=bc_components.encoder
        )
    except ValueError as exc:  # latent representations disagree -> no sharing possible
        raise SystemExit(
            f"{args.ppo_checkpoint} cannot reuse the BC encoder ({exc}). The two "
            "checkpoints do not read the same LeWM representation, so they cannot "
            "appear side by side as the same-prior comparison this teaser makes."
        ) from exc
    if ppo_components.contract["latent_dim"] != bc_components.contract["latent_dim"]:
        raise SystemExit(
            "checkpoints disagree on latent width: BC="
            f"{bc_components.contract['latent_dim']}, PPO="
            f"{ppo_components.contract['latent_dim']}"
        )

    agents = {
        "bc": make_bc_evaluation_agent(
            checkpoint=args.bc_checkpoint,
            stats_path=args.bc_stats,
            components=bc_components,
            execution_mode=args.execution_mode,
        ),
        "ppo": make_ppo_evaluation_agent(
            checkpoint=args.ppo_checkpoint,
            components=ppo_components,
            execution_mode=args.execution_mode,
            deterministic=True,
        ),
    }
    for name, agent in agents.items():
        training_resolution = agent.metadata.get("training_observation_resolution")
        if (
            training_resolution is not None
            and int(training_resolution) != int(args.observation_resolution)
            and not args.allow_resolution_mismatch
        ):
            raise SystemExit(
                f"{name} was trained at {int(training_resolution)}px but the teaser "
                f"renders observations at {args.observation_resolution}px. Pass the "
                "training resolution, or --allow-resolution-mismatch on purpose."
            )
    return agents, bc_components, ppo_components


# -------------------------------------------------------------------- rollouts
@dataclass
class Rollout:
    success: bool
    length: int
    episode_return: float
    final_metrics: dict
    frames: list = field(default_factory=list)

    @property
    def final_error(self):
        """Distance from the goal pose the block was left at (px, position+angle)."""
        return float(self.final_metrics.get("block_state_dist", float("nan")))

    def summary(self):
        return {
            "success": bool(self.success),
            "length": int(self.length),
            "episode_return": float(self.episode_return),
            "final_metrics": {key: float(value) for key, value in self.final_metrics.items()},
        }


def capture_frame(env, render_size):
    """Re-render the current state at ``render_size`` px.

    PushT draws every frame on a fresh 512px pygame canvas and only then
    downsamples to ``render_size``, so this returns the same state the policy
    saw, sampled less aggressively. Rendering does not touch the simulator, and
    ``render_size`` is restored immediately so the policy keeps its own
    observation resolution.
    """
    unwrapped = env.unwrapped
    original = getattr(unwrapped, "render_size", None)
    if original is None:
        raise RuntimeError(
            "PushT env exposes no render_size; cannot re-render panels. "
            "Run with --render-resolution equal to --observation-resolution."
        )
    unwrapped.render_size = int(render_size)
    try:
        return np.asarray(unwrapped.render(), dtype=np.uint8)
    finally:
        unwrapped.render_size = original


def _check_frame_matches_observation(frame, observation):
    """The rendered panel must be the frame the policy acted on, not another state."""
    import cv2

    observation = np.asarray(observation)
    resized = cv2.resize(frame, (observation.shape[1], observation.shape[0]))
    error = float(np.abs(resized.astype(np.float32) - observation.astype(np.float32)).mean())
    if error > 1.0:
        raise RuntimeError(
            f"re-rendered panel disagrees with the policy observation (mean |diff| "
            f"{error:.2f}/255). The panels would show a different state than the "
            "policy acted on. Re-run with --render-resolution "
            f"{observation.shape[0]} to render exactly the observation."
        )


def rollout(env, agent, episode_seed, *, render_size=None):
    """One deterministic episode. Frames are captured only when ``render_size`` is set."""
    observation, info = env.reset(seed=episode_seed)
    agent.reset(episode_seed)
    frames = []
    if render_size is not None:
        frames.append(capture_frame(env, render_size))
        _check_frame_matches_observation(frames[0], observation)

    episode_return, length = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(agent.act(observation, info), dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{agent.agent_type} returned a non-finite action")
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        length += 1
        if render_size is not None:
            frames.append(capture_frame(env, render_size))

    return Rollout(
        success=bool(success_from_info(info, terminated)),
        length=length,
        episode_return=episode_return,
        final_metrics={
            key: float(value)
            for key, value in info.items()
            if key in ("block_state_dist", "block_pos_dist", "block_angle_dist")
            and np.isscalar(value)
        },
        frames=frames,
    )


@dataclass
class Pair:
    """The same episode played by both policies."""

    seed: int
    bc: Rollout
    ppo: Rollout

    @property
    def category(self):
        if self.ppo.success and not self.bc.success:
            return PPO_ONLY
        if self.ppo.success and self.bc.success:
            return BOTH
        if self.bc.success:
            return BC_ONLY
        return NEITHER

    @property
    def improvement(self):
        """How much closer to the goal PPO leaves the block than BC does (px)."""
        return self.bc.final_error - self.ppo.final_error

    def rank_key(self):
        return (self.category, -self.improvement, self.ppo.length)

    def summary(self):
        return {
            "episode_seed": int(self.seed),
            "category": CATEGORY_LABELS[self.category],
            "improvement_px": float(self.improvement),
            "bc": self.bc.summary(),
            "ppo": self.ppo.summary(),
        }


def collect_pairs(env, agents, episode_seeds):
    """Roll both policies through every candidate episode, frames not kept."""
    pairs = []
    for index, episode_seed in enumerate(episode_seeds):
        rollouts = {
            name: rollout(env, agent, episode_seed) for name, agent in agents.items()
        }
        pair = Pair(seed=episode_seed, bc=rollouts["bc"], ppo=rollouts["ppo"])
        pairs.append(pair)
        print(
            f"  [{index + 1:02d}/{len(episode_seeds)}] seed {episode_seed:<11d} "
            f"BC {'success' if pair.bc.success else ' failed'} in {pair.bc.length:3d} "
            f"(err {pair.bc.final_error:6.1f}) | "
            f"PPO {'success' if pair.ppo.success else ' failed'} in {pair.ppo.length:3d} "
            f"(err {pair.ppo.final_error:6.1f}) | {CATEGORY_LABELS[pair.category]}"
        )
    return pairs


def select_pairs(pairs, clips, only_ppo_wins):
    ranked = sorted(pairs, key=Pair.rank_key)
    if only_ppo_wins:
        ranked = [pair for pair in ranked if pair.category == PPO_ONLY]
        if not ranked:
            raise SystemExit(
                "no episode was solved by PPO alone. Re-run with a different --seed, "
                "more --candidates, or without --only-ppo-wins."
            )
    selected = ranked[:clips]
    weak = [pair for pair in selected if pair.category != PPO_ONLY]
    if weak:
        print(
            f"note: {len(weak)} of {len(selected)} selected clips are not "
            "PPO-only successes; a different --seed may sample a sharper contrast."
        )
    return selected


# ------------------------------------------------------------------- rendering
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


@dataclass(frozen=True)
class Layout:
    tile: int
    pad: int = 18
    gap: int = 18
    label_height: int = 32
    caption_height: int = 30
    footer_height: int = 26

    @property
    def width(self):
        return _even(2 * self.tile + 2 * self.pad + self.gap)

    @property
    def height(self):
        return _even(
            self.pad
            + self.label_height
            + self.tile
            + self.caption_height
            + self.footer_height
            + self.pad
        )

    @property
    def panel_top(self):
        return self.pad + self.label_height

    def panel_left(self, index):
        return self.pad + index * (self.tile + self.gap)


def _even(value):
    """Even canvas dimensions: h.264 cannot encode odd ones."""
    return int(value) + int(value) % 2


def compose_chrome(layout, pair, args):
    """Everything that does not change between frames, drawn once."""
    canvas = Image.new("RGB", (layout.width, layout.height), INK["surface"])
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(PANELS):
        left = layout.panel_left(index)
        color = PANEL_COLORS[name]
        draw.text((left, layout.pad - 2), PANEL_TITLES[name], fill=color, font=font(19))
        underline = layout.pad + layout.label_height - 8
        draw.rectangle([left, underline, left + layout.tile - 1, underline + 2], fill=color)
    footer = (
        f"same start state  ·  episode seed {pair.seed}  ·  block spawns within "
        f"{int(round(args.block_start_radius))} px of the goal  ·  "
        f"{args.max_episode_steps}-step limit"
    )
    draw.text(
        (layout.pad, layout.height - layout.pad - layout.footer_height + 6),
        footer,
        fill=INK["muted"],
        font=_fitting_font(draw, footer, layout.width - 2 * layout.pad),
    )
    return canvas


def _fitting_font(draw, text, max_width, sizes=(13, 12, 11, 10)):
    """Largest of ``sizes`` that keeps ``text`` inside ``max_width``.

    The footer carries the episode seed, whose width varies by several
    characters, so a fixed size silently runs off the edge of the canvas.
    """
    for size in sizes:
        if draw.textlength(text, font=font(size)) <= max_width:
            return font(size)
    return font(sizes[-1])


def panel_caption(rollout, step, max_steps):
    if rollout.success and step >= rollout.length:
        return f"SUCCESS in {rollout.length} steps", SUCCESS_COLOR
    if not rollout.success and step >= rollout.length:
        return f"no success in {rollout.length} steps", FAILURE_COLOR
    return f"step {step:3d} / {max_steps}", INK["secondary"]


def compose_frame(layout, chrome, pair, index, max_steps):
    canvas = chrome.copy()
    draw = ImageDraw.Draw(canvas)
    for panel_index, name in enumerate(PANELS):
        rollout = getattr(pair, name)
        step = min(index, rollout.length)
        left, top = layout.panel_left(panel_index), layout.panel_top
        image = Image.fromarray(rollout.frames[step]).resize(
            (layout.tile, layout.tile), Image.LANCZOS
        )
        canvas.paste(image, (left, top))
        draw.rectangle(
            [left, top, left + layout.tile - 1, top + layout.tile - 1],
            outline=INK["grid"],
        )
        text, color = panel_caption(rollout, step, max_steps)
        draw.text((left, top + layout.tile + 8), text, fill=color, font=font(15))
    return np.asarray(canvas)


def clip_frames(pair, args, layout):
    """Composited frames for one episode, both panels advancing in lockstep.

    The shorter rollout freezes on its final frame -- its caption already says
    whether it stopped because it succeeded or because it ran out of steps.
    """
    steps = max(pair.bc.length, pair.ppo.length)
    indices = list(range(0, steps + 1, max(1, args.frame_stride)))
    if indices[-1] != steps:
        indices.append(steps)
    chrome = compose_chrome(layout, pair, args)
    frames = [compose_frame(layout, chrome, pair, index, args.max_episode_steps) for index in indices]
    hold = max(0, int(round(args.hold_seconds * args.fps)))
    return [frames[0]] * (hold // 2) + frames + [frames[-1]] * hold


def write_gif(frames, path, fps, colors):
    """GIF with one shared palette, so flat PushT colours do not shimmer.

    The palette is built from frames sampled across the whole clip, not from the
    first one: the success and failure captions only appear at the end, and a
    palette that never saw them renders them in the nearest colour it does have.
    """
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


def write_animations(frames, output_dir, stem, args):
    written = []
    for suffix in args.formats:
        path = Path(output_dir) / f"{stem}.{suffix}"
        if suffix == "gif":
            write_gif(frames, path, args.fps, args.gif_colors)
        else:
            write_mp4(frames, path, args.fps)
        written.append(path)
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    return written


# ------------------------------------------------------------------------ main
def main():
    args = build_parser().parse_args()
    if args.clips < 1:
        raise SystemExit("--clips must be at least 1")
    if args.tile < 64:
        raise SystemExit("--tile must be at least 64")

    episode_seeds = args.episode_seeds or sample_episode_seeds(args.seed, args.candidates)
    output_dir = repo_path(args.output_dir) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading policies...")
    agents, bc_components, ppo_components = build_agents(args)

    config = PushTEvalConfig(
        episodes=len(episode_seeds),
        seed=args.seed,
        episode_seeds=tuple(episode_seeds),
        max_episode_steps=args.max_episode_steps,
        observation_resolution=args.observation_resolution,
        block_start_radius=args.block_start_radius,
    )
    env = make_evaluation_env(config)
    try:
        print(f"Scouting {len(episode_seeds)} episodes (both policies, no frames)...")
        pairs = collect_pairs(env, agents, episode_seeds)
        selected = select_pairs(pairs, args.clips, args.only_ppo_wins)
        layout = Layout(tile=args.tile)
        print(f"\nRendering {len(selected)} clip(s) at {layout.width}x{layout.height}...")

        clips, all_frames = [], []
        for index, pair in enumerate(selected, start=1):
            print(f"  replaying seed {pair.seed} with frames...")
            for name, agent in agents.items():
                replay = rollout(env, agent, pair.seed, render_size=args.render_resolution)
                scouted = getattr(pair, name)
                if (replay.success, replay.length) != (scouted.success, scouted.length):
                    raise RuntimeError(
                        f"replay of seed {pair.seed} for {name} diverged from the "
                        f"scouting pass ({scouted.length} steps -> {replay.length}). "
                        "The rollouts are not deterministic, so the two panels "
                        "cannot be trusted to be the episodes that were ranked."
                    )
                setattr(pair, name, replay)
            frames = clip_frames(pair, args, layout)
            all_frames.extend(frames)
            paths = write_animations(
                frames, output_dir, f"clip{index:02d}_seed{pair.seed}", args
            )
            clips.append({**pair.summary(), "files": [str(path) for path in paths]})
            for name in PANELS:  # a 300-step 512px rollout is ~0.2 GB; keep only the metrics
                getattr(pair, name).frames = []

        combined = []
        if len(selected) > 1:
            print("Writing the combined teaser...")
            combined = [str(path) for path in write_animations(all_frames, output_dir, "teaser", args)]
    finally:
        env.close()

    manifest = {
        "settings": vars(args),
        "checkpoints": {
            "bc": args.bc_checkpoint,
            "bc_stats": args.bc_stats or "inferred from the BC checkpoint",
            "ppo": args.ppo_checkpoint,
            # Provenance worth stating in the post: the prior this PPO run was
            # actually initialized from, as recorded in its own config.
            "ppo_was_initialized_from": ppo_components.config.get("bc_checkpoint"),
        },
        "contracts": {"bc": bc_components.contract, "ppo": ppo_components.contract},
        "episode_seeds": [int(seed) for seed in episode_seeds],
        "candidates": [pair.summary() for pair in pairs],
        "clips": clips,
        "combined": combined,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"\nwrote {manifest_path}")
    print(f"outputs in {output_dir}")


if __name__ == "__main__":
    main()
