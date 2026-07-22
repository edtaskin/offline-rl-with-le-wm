from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from PIL import Image

from scripts.ablations.visual_robustness.analysis import (
    adaptation_rows,
    analyze,
    difference_in_differences_rows,
    latent_metric_rows,
    paired_robustness_rows,
    validate_evaluation_protocols,
)
from scripts.ablations.visual_robustness.cache import (
    CounterfactualRenderer,
    build_cache,
    check_cache,
    expected_cache_metadata,
    load_source_arrays,
    save_thumbnail_grid,
)
from scripts.ablations.visual_robustness.cli import build_parser
from scripts.ablations.visual_robustness.evaluation import (
    evaluate_checkpoint,
    make_shifted_evaluation_env,
    save_environment_screenshots,
)
from scripts.ablations.visual_robustness.shifts import (
    ADAPTATION_CONDITIONS,
    BASE_BACKGROUND,
    BASE_BLOCK,
    BASE_GOAL,
    CONDITION_NAMES,
    get_condition,
)
from scripts.ablations.visual_robustness.training import train_policy
from src.evaluation.pusht import PushTEvalConfig


class DummyEncoder(torch.nn.Module):
    def __init__(self, checkpoint_path, feature_dim=4):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.feature_dim = feature_dim
        self.latent_dim = feature_dim
        self.register_parameter(
            "anchor", torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        )

    @property
    def preprocessing_metadata(self):
        return {"image_size": [96, 96], "image_normalization": "dummy"}

    def forward(self, images):
        values = images.float().mean(dim=(1, 2, 3)) / 255.0
        return values[:, None].repeat(1, self.feature_dim)

    def encode(self, images):
        return self(images)


def small_source(path, count=16):
    states = np.zeros((count, 5), dtype=np.float32)
    states[:, 0] = np.linspace(100, 300, count)
    states[:, 1] = np.linspace(400, 250, count)
    states[:, 2] = np.linspace(300, 220, count)
    states[:, 3] = np.linspace(150, 280, count)
    states[:, 4] = np.linspace(0, np.pi / 2, count)
    actions = states[:, :2] + np.array([10.0, -5.0], dtype=np.float32)
    np.savez(
        path,
        states=states,
        actions=actions,
        episode_ends=np.array([count // 2, count], dtype=np.int64),
        images=np.zeros((count, 1, 1, 3), dtype=np.uint8),
    )


def record(encoder, train_condition, training_seed, eval_condition, successes):
    episodes = [
        {"seed": 1000 + index, "success": float(value), "final_metrics": {}}
        for index, value in enumerate(successes)
    ]
    return {
        "encoder": encoder,
        "train_condition": train_condition,
        "training_seed": training_seed,
        "eval_condition": eval_condition,
        "payload": {
            "episodes": episodes,
            "summary": {"success_rate": float(np.mean(successes))},
        },
        "path": "unused",
    }


class VisualRobustnessTests(unittest.TestCase):
    def test_evaluation_defaults_match_canonical_repeated_protocol(self):
        args = build_parser().parse_args(["evaluate"])
        all_args = build_parser().parse_args(["all"])
        analyze_args = build_parser().parse_args(["analyze"])
        self.assertEqual(args.episodes, 50)
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.eval_seed, 42)
        self.assertEqual(args.observation_resolution, 96)
        self.assertEqual(args.encoders, ["lewm"])
        self.assertEqual(all_args.encoders, ["lewm"])
        self.assertEqual(analyze_args.encoders, ["lewm"])

    def test_condition_registry_and_isolated_components(self):
        self.assertEqual(len(CONDITION_NAMES), 16)
        background = get_condition("background_1")
        block = get_condition("block_1")
        goal = get_condition("goal_1")
        self.assertNotEqual(background.background, BASE_BACKGROUND)
        self.assertEqual((background.block, background.goal), (BASE_BLOCK, BASE_GOAL))
        self.assertNotEqual(block.block, BASE_BLOCK)
        self.assertEqual((block.background, block.goal), (BASE_BACKGROUND, BASE_GOAL))
        self.assertNotEqual(goal.goal, BASE_GOAL)
        self.assertEqual(get_condition("resolution_224").observation_resolution, 224)
        self.assertEqual(get_condition("blur_4").blur_sigma, 4.0)
        self.assertIn("blur_4", ADAPTATION_CONDITIONS)
        with self.assertRaises(ValueError):
            get_condition("not-a-condition")

    def test_original_condition_metadata_remains_cache_compatible(self):
        clean = get_condition("clean").to_dict()
        self.assertNotIn("blur_sigma", clean)
        self.assertNotIn("observation_resolution", clean)

    def test_gaussian_blur_is_deterministic_and_preserves_shape_and_dtype(self):
        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
        frame[10:22, 10:22] = BASE_BLOCK
        condition = get_condition("blur_2")
        first = condition.apply_post_render(frame)
        second = condition.apply_post_render(frame)
        self.assertEqual(first.shape, frame.shape)
        self.assertEqual(first.dtype, frame.dtype)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, frame))

    def test_checkerboard_changes_only_exact_background_pixels(self):
        frame = np.full((24, 24, 3), BASE_BACKGROUND, dtype=np.uint8)
        frame[10:14, 10:14] = BASE_BLOCK
        shifted = get_condition("texture").apply_post_render(frame)
        self.assertTrue(np.array_equal(shifted[10:14, 10:14], frame[10:14, 10:14]))
        self.assertFalse(np.array_equal(shifted[0, 0], frame[0, 0]))
        self.assertFalse(np.array_equal(shifted[0, 0], shifted[0, 13]))

    def test_counterfactual_rendering_is_deterministic_and_typed(self):
        state = np.array([150, 350, 320, 180, 0.3], dtype=np.float32)
        with CounterfactualRenderer(get_condition("combined"), 96) as first:
            a = first.render(state)
            b = first.render(state)
        with CounterfactualRenderer(get_condition("combined"), 96) as second:
            c = second.render(state)
        self.assertEqual(a.shape, (96, 96, 3))
        self.assertEqual(a.dtype, np.uint8)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a, c))

    def test_resolution_condition_renders_natively_at_224(self):
        state = np.array([150, 350, 320, 180, 0.3], dtype=np.float32)
        with CounterfactualRenderer(get_condition("resolution_224"), 96) as renderer:
            frame = renderer.render(state)
            self.assertEqual(renderer.resolution, 224)
        self.assertEqual(frame.shape, (224, 224, 3))

    def test_clean_renderer_reproduces_first_expert_frame(self):
        data_path = Path("data/expert_trajectories/pusht_expert.npz")
        if not data_path.exists():
            self.skipTest("expert dataset is unavailable")
        arrays = load_source_arrays(data_path)
        with zipfile.ZipFile(data_path) as archive:
            stream = archive.open("images.npy")
            self.assertEqual(stream.read(6), b"\x93NUMPY")
            major, _minor = stream.read(2)
            size = 2 if major == 1 else 4
            header_length = struct.unpack(
                "<H" if size == 2 else "<I", stream.read(size)
            )[0]
            stream.read(header_length)
            frame = np.frombuffer(
                stream.read(96 * 96 * 3 * 4), dtype="<f4"
            ).reshape(96, 96, 3)
        with CounterfactualRenderer(get_condition("clean"), 96) as renderer:
            rerendered = renderer.render(arrays["states"][0])
        difference = np.abs(frame.astype(np.float32) - rerendered.astype(np.float32))
        self.assertLess(float(difference.mean()), 0.5)
        self.assertGreater(float(np.mean(difference == 0)), 0.98)

    def test_shifted_evaluation_preserves_seeded_physical_state(self):
        config = PushTEvalConfig(
            episodes=1,
            seed=77,
            max_episode_steps=2,
            observation_resolution=96,
            block_start_radius=200.0,
        )
        clean = make_shifted_evaluation_env(config, "clean")
        shifted = make_shifted_evaluation_env(config, "combined")
        try:
            clean.reset(seed=77)
            shifted.reset(seed=77)
            self.assertTrue(np.allclose(clean.unwrapped._get_obs(), shifted.unwrapped._get_obs()))
            self.assertTrue(np.allclose(clean.unwrapped.goal_pose, shifted.unwrapped.goal_pose))
        finally:
            clean.close()
            shifted.close()

    def test_texture_evaluation_wrapper_resets_and_steps(self):
        config = PushTEvalConfig(
            episodes=1,
            seed=81,
            max_episode_steps=2,
            observation_resolution=96,
            block_start_radius=200.0,
        )
        env = make_shifted_evaluation_env(config, "texture")
        try:
            observation, _ = env.reset(seed=81)
            self.assertEqual(observation.shape, (96, 96, 3))
            next_observation, *_ = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(next_observation.shape, observation.shape)
        finally:
            env.close()

    def test_blur_and_resolution_evaluation_observations(self):
        config = PushTEvalConfig(
            episodes=1,
            seed=82,
            max_episode_steps=2,
            observation_resolution=96,
            block_start_radius=200.0,
        )
        clean = make_shifted_evaluation_env(config, "clean")
        blur = make_shifted_evaluation_env(config, "blur_2")
        high_resolution = make_shifted_evaluation_env(config, "resolution_224")
        try:
            clean_frame, _ = clean.reset(seed=82)
            blur_frame, _ = blur.reset(seed=82)
            high_resolution_frame, _ = high_resolution.reset(seed=82)
            self.assertEqual(clean_frame.shape, (96, 96, 3))
            self.assertEqual(blur_frame.shape, clean_frame.shape)
            self.assertFalse(np.array_equal(blur_frame, clean_frame))
            self.assertEqual(high_resolution_frame.shape, (224, 224, 3))
            self.assertTrue(
                np.allclose(clean.unwrapped._get_obs(), blur.unwrapped._get_obs())
            )
            self.assertTrue(
                np.allclose(
                    clean.unwrapped._get_obs(), high_resolution.unwrapped._get_obs()
                )
            )
        finally:
            clean.close()
            blur.close()
            high_resolution.close()

    def test_environment_screenshots_save_individual_frames_and_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_environment_screenshots(
                output_root=directory,
                condition_names=["clean", "combined", "texture", "resolution_224"],
                seeds=[1000, 1001],
                max_episode_steps=2,
            )
            self.assertTrue(artifacts["contact_sheet"].exists())
            self.assertTrue(artifacts["manifest"].exists())
            self.assertEqual(len(artifacts["images"]), 8)
            manifest = json.loads(artifacts["manifest"].read_text())
            self.assertEqual(manifest["observation_resolution"], 96)
            self.assertEqual(manifest["condition_resolutions"]["clean"], 96)
            self.assertEqual(
                manifest["condition_resolutions"]["resolution_224"], 224
            )
            high_resolution = (
                Path(directory)
                / "environment_screenshots"
                / "resolution_224"
                / "seed_1000.png"
            )
            with Image.open(high_resolution) as screenshot:
                self.assertEqual(screenshot.size, (224, 224))

    def test_thumbnail_grid_normalizes_native_resolutions_for_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "expert.npz"
            small_source(data_path)
            path = save_thumbnail_grid(
                data_path=data_path,
                output_root=root,
                condition_names=["clean", "resolution_224", "blur_2"],
                resolution=96,
                count=2,
            )
            with Image.open(path) as grid:
                self.assertEqual(grid.size, (150 + 2 * 96, 3 * 96))

    def test_cache_metadata_detects_condition_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "expert.npz"
            checkpoint = root / "dummy.pth"
            checkpoint.touch()
            small_source(data_path)
            arrays = load_source_arrays(data_path)
            encoder = DummyEncoder(checkpoint)
            clean = expected_cache_metadata(
                data_path, arrays, "lewm", encoder, get_condition("clean")
            )
            shifted = expected_cache_metadata(
                data_path, arrays, "lewm", encoder, get_condition("block_1")
            )
            path = root / "cache.pt"
            torch.save({"latents": torch.zeros(16, 4), "metadata": clean}, path)
            self.assertEqual(check_cache(path, clean), (True, []))
            valid, messages = check_cache(path, shifted)
            self.assertFalse(valid)
            self.assertTrue(any("condition" in message for message in messages))
            for key in ("states_sha256", "actions_sha256", "episode_ends_sha256"):
                self.assertEqual(clean["source"][key], shifted["source"][key])

    def test_resolution_condition_overrides_cache_renderer_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "expert.npz"
            checkpoint = root / "dummy.pth"
            checkpoint.touch()
            small_source(data_path)
            arrays = load_source_arrays(data_path)
            metadata = expected_cache_metadata(
                data_path,
                arrays,
                "lewm",
                DummyEncoder(checkpoint),
                get_condition("resolution_224"),
                resolution=96,
            )
            self.assertEqual(metadata["renderer"]["resolution"], 224)

    def test_latent_metrics_are_reported_from_paired_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache" / "lewm"
            cache_root.mkdir(parents=True)
            clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            blurred = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
            torch.save({"latents": clean}, cache_root / "clean.pt")
            torch.save({"latents": blurred}, cache_root / "blur_1.pt")
            rows = latent_metric_rows(root, encoder_names=("lewm",))
            row = next(item for item in rows if item["eval_condition"] == "blur_1")
            self.assertEqual(row["samples"], 2)
            self.assertIn("cosine_mean", row)
            self.assertIn("normalized_l2_mean", row)
            self.assertIn("variance_ratio", row)

    def test_paired_statistics_and_difference_in_differences(self):
        records = []
        for training_seed in (42, 43):
            records += [
                record("lewm", "clean", training_seed, "clean", [1, 1, 0, 0]),
                record("lewm", "clean", training_seed, "block_1", [1, 1, 0, 0]),
                record("dinov2", "clean", training_seed, "clean", [1, 1, 1, 0]),
                record("dinov2", "clean", training_seed, "block_1", [0, 0, 1, 0]),
            ]
        robust = paired_robustness_rows(records, bootstrap_samples=200)
        lewm = next(row for row in robust if row["encoder"] == "lewm")
        self.assertAlmostEqual(lewm["success_delta"], 0.0)
        comparative = difference_in_differences_rows(records, bootstrap_samples=200)
        self.assertAlmostEqual(comparative[0]["lewm_minus_dinov2_degradation"], 0.5)
        records += [
            record("lewm", "block_1", 42, "block_1", [1, 1, 0, 0]),
            record("lewm", "block_1", 43, "block_1", [1, 1, 0, 0]),
        ]
        adaptation = adaptation_rows(records, bootstrap_samples=200)
        self.assertAlmostEqual(adaptation[0]["matched_delta_vs_clean_baseline"], 0.0)

    def test_analysis_rejects_mixed_evaluation_protocols(self):
        records = [
            {
                **record("lewm", "clean", 42, "clean", [1, 0]),
                "payload": {
                    "config": {
                        "episodes": 50,
                        "repeats": 3,
                        "repeat_seeds": [42, 92, 142],
                        "seed": 42,
                        "max_episode_steps": 300,
                        "block_start_radius": 200.0,
                        "observation_resolution": 96,
                    }
                },
            },
            {
                **record("lewm", "clean", 43, "clean", [1, 0]),
                "payload": {
                    "config": {
                        "episodes": 200,
                        "repeats": None,
                        "repeat_seeds": None,
                        "seed": 1000,
                        "max_episode_steps": 300,
                        "block_start_radius": 200.0,
                        "observation_resolution": None,
                    }
                },
            },
        ]
        with self.assertRaises(RuntimeError):
            validate_evaluation_protocols(records)

    def test_small_cpu_smoke_cache_train_evaluate_analyze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "expert.npz"
            checkpoint = root / "dummy_encoder.pth"
            checkpoint.touch()
            small_source(data_path)
            output_root = root / "ablation"
            encoder = DummyEncoder(checkpoint)
            build_cache(
                data_path=data_path,
                output_root=output_root,
                encoder_name="lewm",
                encoder=encoder,
                condition_name="clean",
                batch_size=4,
            )
            policy_path = train_policy(
                data_path=data_path,
                output_root=output_root,
                encoder_name="lewm",
                condition_name="clean",
                seed=42,
                device="cpu",
                epochs=1,
                batch_size=4,
            )
            self.assertEqual(
                train_policy(
                    data_path=data_path,
                    output_root=output_root,
                    encoder_name="lewm",
                    condition_name="clean",
                    seed=42,
                    device="cpu",
                    epochs=1,
                    batch_size=4,
                ),
                policy_path,
            )
            outputs = evaluate_checkpoint(
                checkpoint_path=policy_path,
                output_root=output_root,
                condition_names=["clean", "resolution_224", "blur_1"],
                device="cpu",
                episodes=1,
                repeats=2,
                eval_seed=1000,
                max_episode_steps=2,
                encoder=encoder,
            )
            self.assertEqual(len(outputs), 3)
            self.assertTrue(outputs[0].exists())
            payload = json.loads(outputs[0].read_text())
            self.assertEqual(payload["config"]["repeats"], 2)
            self.assertEqual(payload["config"]["repeat_seeds"], [1000, 1001])
            self.assertEqual(payload["config"]["observation_resolution"], 96)
            self.assertEqual(len(payload["repeat_summaries"]), 2)
            self.assertEqual(len(payload["episodes"]), 2)
            resolution_payload = json.loads(outputs[1].read_text())
            self.assertEqual(
                resolution_payload["config"]["observation_resolution"], 224
            )
            blur_payload = json.loads(outputs[2].read_text())
            self.assertEqual(blur_payload["ablation"]["eval_condition"], "blur_1")
            with self.assertRaises(RuntimeError):
                evaluate_checkpoint(
                    checkpoint_path=policy_path,
                    output_root=output_root,
                    condition_names=["clean"],
                    device="cpu",
                    episodes=1,
                    repeats=2,
                    eval_seed=1000,
                    max_episode_steps=2,
                    observation_resolution=224,
                    encoder=encoder,
                )
            report = analyze(
                output_root=output_root,
                data_path=data_path,
                bootstrap_samples=20,
                include_action_metrics=False,
            )
            self.assertTrue(report.exists())
            self.assertEqual(json.loads(report.read_text())["bootstrap_samples"], 20)


if __name__ == "__main__":
    unittest.main()
