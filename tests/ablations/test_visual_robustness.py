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

from scripts.ablations.visual_robustness.analysis import (
    adaptation_rows,
    analyze,
    difference_in_differences_rows,
    paired_robustness_rows,
)
from scripts.ablations.visual_robustness.cache import (
    CounterfactualRenderer,
    build_cache,
    check_cache,
    expected_cache_metadata,
    load_source_arrays,
)
from scripts.ablations.visual_robustness.evaluation import (
    evaluate_checkpoint,
    make_shifted_evaluation_env,
    save_environment_screenshots,
)
from scripts.ablations.visual_robustness.shifts import (
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
    def test_condition_registry_and_isolated_components(self):
        self.assertEqual(len(CONDITION_NAMES), 12)
        background = get_condition("background_1")
        block = get_condition("block_1")
        goal = get_condition("goal_1")
        self.assertNotEqual(background.background, BASE_BACKGROUND)
        self.assertEqual((background.block, background.goal), (BASE_BLOCK, BASE_GOAL))
        self.assertNotEqual(block.block, BASE_BLOCK)
        self.assertEqual((block.background, block.goal), (BASE_BACKGROUND, BASE_GOAL))
        self.assertNotEqual(goal.goal, BASE_GOAL)
        with self.assertRaises(ValueError):
            get_condition("not-a-condition")

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
            episodes=1, seed=77, max_episode_steps=2, block_start_radius=200.0
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
            episodes=1, seed=81, max_episode_steps=2, block_start_radius=200.0
        )
        env = make_shifted_evaluation_env(config, "texture")
        try:
            observation, _ = env.reset(seed=81)
            self.assertEqual(observation.shape, (96, 96, 3))
            next_observation, *_ = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(next_observation.shape, observation.shape)
        finally:
            env.close()

    def test_environment_screenshots_save_individual_frames_and_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_environment_screenshots(
                output_root=directory,
                condition_names=["clean", "combined", "texture"],
                seeds=[1000, 1001],
                max_episode_steps=2,
            )
            self.assertTrue(artifacts["contact_sheet"].exists())
            self.assertTrue(artifacts["manifest"].exists())
            self.assertEqual(len(artifacts["images"]), 6)

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
                condition_names=["clean"],
                device="cpu",
                episodes=1,
                eval_seed=1000,
                max_episode_steps=2,
                encoder=encoder,
            )
            self.assertTrue(outputs[0].exists())
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

