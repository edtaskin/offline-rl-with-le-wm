import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.rq1.evaluate_dream_snapshots import (
    _is_dream_config,
    build_parser,
    resolve_settings,
)
from scripts.rq_common import snapshot_checkpoints


class EvaluateDreamSnapshotsTest(unittest.TestCase):
    def test_snapshot_discovery_returns_every_snapshot_in_iteration_order(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "snapshot_step000000010_it00010.pt",
                "snapshot_step000000000_it00000.pt",
                "snapshot_step000000020_it00020.pt",
                "snapshot_bad.pt",
                "final.pt",
            ):
                (run_dir / name).touch()

            snapshots = snapshot_checkpoints(run_dir)

        self.assertEqual([row["iteration"] for row in snapshots], [0, 10, 20])
        self.assertEqual([row["env_steps"] for row in snapshots], [0, 10, 20])

    def test_checkpoint_task_settings_are_used_by_default(self):
        args = build_parser().parse_args(["run"])
        settings = resolve_settings(
            args,
            {
                "env_id": "example/PushT-v2",
                "max_episode_steps": 123,
                "fixed_target_block_success": False,
                "block_start_near_goal": True,
                "block_start_radius": 75.0,
            },
        )

        self.assertEqual(settings.env_id, "example/PushT-v2")
        self.assertEqual(settings.max_episode_steps, 123)
        self.assertFalse(settings.fixed_target_block_success)
        self.assertEqual(settings.block_start_radius, 75.0)
        self.assertTrue(settings.deterministic)
        self.assertEqual(settings.execution_mode, "open-loop")

    def test_explicit_evaluation_flags_override_checkpoint(self):
        args = build_parser().parse_args(
            [
                "run",
                "--env-id",
                "override/PushT-v1",
                "--max-episode-steps",
                "99",
                "--fixed-target-block-success",
                "--unrestricted-block-start",
                "--stochastic",
                "--execution-mode",
                "receding-horizon",
                "--replan-interval",
                "2",
            ]
        )
        settings = resolve_settings(
            args,
            {
                "env_id": "saved/PushT-v1",
                "max_episode_steps": 300,
                "fixed_target_block_success": False,
                "block_start_near_goal": True,
                "block_start_radius": 200.0,
            },
        )

        self.assertEqual(settings.env_id, "override/PushT-v1")
        self.assertEqual(settings.max_episode_steps, 99)
        self.assertTrue(settings.fixed_target_block_success)
        self.assertIsNone(settings.block_start_radius)
        self.assertFalse(settings.deterministic)
        self.assertEqual(settings.replan_interval, 2)

    def test_protocol_id_changes_with_evaluation_arguments(self):
        default_args = build_parser().parse_args(["run"])
        short_args = build_parser().parse_args(["run", "--episodes", "2"])
        config = {"block_start_near_goal": True, "block_start_radius": 200.0}

        default = resolve_settings(default_args, config)
        short = resolve_settings(short_args, config)

        self.assertNotEqual(default.evaluation_id, short.evaluation_id)
        self.assertEqual(
            default.evaluation_id,
            resolve_settings(default_args, config).evaluation_id,
        )

    def test_dream_checkpoint_detection_uses_train_lewm_fields(self):
        self.assertTrue(_is_dream_config({"dream_episode_steps": 20, "wm_frameskip": 5}))
        self.assertFalse(_is_dream_config({"env_id": "swm/PushT-v1"}))


if __name__ == "__main__":
    unittest.main()
