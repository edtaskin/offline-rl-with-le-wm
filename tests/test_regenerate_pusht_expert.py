"""Focused tests for the high-resolution PushT dataset generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.regenerate_pusht_expert import parse_args, regenerate_dataset


class RegeneratePushTExpertTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "expert.npz"
        self.output = self.root / "expert_8.npz"
        np.savez(
            self.source,
            states=np.zeros((3, 5), dtype=np.float32),
            actions=np.zeros((3, 2), dtype=np.float32),
            images=np.linspace(0, 255, 3 * 4 * 4 * 3, dtype=np.float32).reshape(
                3, 4, 4, 3
            ),
            episode_ends=np.array([2, 3], dtype=np.int64),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_upsample_writes_uint8_and_repairs_truncated_episode(self):
        args = parse_args(
            [
                "--dataset",
                str(self.source),
                "--output-dataset",
                str(self.output),
                "--resolution",
                "8",
                "--mode",
                "upsample",
                "--max-frames",
                "1",
                "--verify-n",
                "1",
            ]
        )
        regenerate_dataset(args)

        with np.load(self.output) as generated:
            self.assertEqual(generated["images"].shape, (1, 8, 8, 3))
            self.assertEqual(generated["images"].dtype, np.uint8)
            self.assertEqual(generated["states"].shape, (1, 5))
            self.assertEqual(generated["actions"].shape, (1, 2))
            self.assertTrue(np.array_equal(generated["episode_ends"], [1]))
        self.assertEqual(list(self.root.glob(".*.images.npy")), [])

    def test_input_cannot_be_overwritten(self):
        args = parse_args(
            [
                "--dataset",
                str(self.source),
                "--output-dataset",
                str(self.source),
                "--mode",
                "upsample",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must differ"):
            regenerate_dataset(args)


if __name__ == "__main__":
    unittest.main()
