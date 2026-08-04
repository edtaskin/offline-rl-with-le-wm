import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.bc.train_bc_latent import (
    _hf_run_folder,
    _run_periodic_evaluation,
    build_parser,
    checkpoint_artifact_paths,
    train_latent_bc,
)
from src.utils.hf_hub import HubUploadResult


class BCTrainingArtifactTests(unittest.TestCase):
    def test_checkpoint_base_expands_to_best_and_final_pairs(self):
        paths = checkpoint_artifact_paths("runs/bc/projected.pth")

        self.assertEqual(paths["best"], "runs/bc/projected_best.pth")
        self.assertEqual(paths["best_stats"], "runs/bc/projected_best_stats.pth")
        self.assertEqual(paths["final"], "runs/bc/projected_final.pth")
        self.assertEqual(paths["final_stats"], "runs/bc/projected_final_stats.pth")
        self.assertEqual(
            paths["eval_history"], "runs/bc/projected_eval_history.json"
        )

    def test_hf_folder_prefers_explicit_prefix_then_wandb_name(self):
        args = SimpleNamespace(
            hf_path_prefix="experiments/raw-cls",
            wandb_run_name="ignored",
            checkpoint_path="runs/bc/raw.pth",
        )
        self.assertEqual(_hf_run_folder(args), "experiments/raw-cls")

        args.hf_path_prefix = None
        args.wandb_run_name = "PushT BC / projected CLS"
        self.assertEqual(_hf_run_folder(args), "PushT-BC-projected-CLS")

    def test_periodic_evaluation_uses_canonical_entrypoint_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--checkpoint_path",
                    str(Path(temporary_dir) / "policy.pth"),
                    "--eval_episodes",
                    "2",
                    "--eval_repeats",
                    "1",
                ]
            )
            policy = LatentBCPolicy(
                latent_dim=2,
                frame_stack=1,
                hidden_dim=4,
                action_chunk_size=1,
            )
            policy.train()
            seen = {}

            def fake_evaluate(eval_args):
                seen["args"] = eval_args
                self.assertTrue(Path(eval_args.checkpoint).is_file())
                self.assertTrue(Path(eval_args.stats).is_file())
                random.random()
                np.random.rand()
                torch.rand(1)
                return SimpleNamespace(summary={"success_rate": 0.25})

            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            expected_python = random.Random(7).random()
            expected_numpy = np.random.RandomState(7).rand()
            expected_torch = torch.rand(1, generator=torch.Generator().manual_seed(7))
            with patch(
                "src.evaluation.evaluate_pusht.evaluate_from_args",
                side_effect=fake_evaluate,
            ):
                result = _run_periodic_evaluation(
                    args,
                    policy=policy,
                    run_metadata={"observation_resolution": 224},
                    epoch=50,
                    device=torch.device("cpu"),
                    observation_resolution=224,
                )

            self.assertEqual(result.summary["success_rate"], 0.25)
            self.assertEqual(seen["args"].agent_type, "bc")
            self.assertEqual(seen["args"].episodes, 2)
            self.assertEqual(seen["args"].repeats, 1)
            self.assertEqual(seen["args"].run_name, "policy-epoch-50")
            self.assertFalse(Path(seen["args"].checkpoint).exists())
            self.assertTrue(policy.training)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))

    def test_success_rate_selects_best_and_hf_upload_uses_one_run_folder(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_base = root / "policy.pth"
            args = build_parser().parse_args(
                [
                    "--checkpoint_path",
                    str(checkpoint_base),
                    "--epochs",
                    "2",
                    "--batch_size",
                    "2",
                    "--hidden_dim",
                    "4",
                    "--frame_stack",
                    "1",
                    "--frame_stride",
                    "1",
                    "--action_chunk_size",
                    "1",
                    "--eval_interval",
                    "1",
                    "--eval_episodes",
                    "1",
                    "--eval_repeats",
                    "1",
                    "--save_interval",
                    "99",
                    "--log_interval",
                    "99",
                    "--wandb_run_name",
                    "raw-cls-seed-42",
                    "--push_to_hf",
                    "--hf_repo_id",
                    "owner/bc",
                ]
            )
            dataset = TensorDataset(
                torch.arange(8, dtype=torch.float32).reshape(4, 1, 2),
                torch.zeros(4, 1, 2),
            )
            prepared = {
                "dataset": dataset,
                "dataset_stats": {
                    "latent_dim": 2,
                    "observation_resolution": 224,
                    "source_image_shape": [224, 224],
                    "latent_representation": "raw_cls",
                },
                "extractor": None,
                "latent_dim": 2,
                "use_latent_cache": True,
                "cache_path": str(root / "cache.pt"),
                "cache_rebuilt": False,
                "expected_metadata": {},
            }
            evaluated_states = []
            success_rates = iter([0.75, 0.50])

            def fake_evaluation(*unused_args, policy, epoch, **unused_kwargs):
                evaluated_states.append(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in policy.state_dict().items()
                    }
                )
                return SimpleNamespace(
                    summary={"success_rate": next(success_rates), "episodes": 1},
                    run_dir=f"eval-{epoch}",
                    metrics_path=f"eval-{epoch}/metrics.json",
                )

            upload_result = HubUploadResult(
                repo_id="owner/bc",
                repo_type="model",
                repo_url="https://huggingface.co/owner/bc",
                uploaded_files=(),
            )
            with (
                patch("src.bc.train_bc_latent._prepare_dataset", return_value=prepared),
                patch(
                    "src.bc.train_bc_latent._run_periodic_evaluation",
                    side_effect=fake_evaluation,
                ),
                patch(
                    "src.bc.train_bc_latent.push_files_to_hub",
                    return_value=upload_result,
                ) as push,
            ):
                result = train_latent_bc(args)

            paths = checkpoint_artifact_paths(checkpoint_base)
            for path in paths.values():
                self.assertTrue(Path(path).is_file(), path)
            self.assertFalse(checkpoint_base.exists())
            self.assertEqual(result["best_eval_epoch"], 1)
            self.assertEqual(result["best_eval_success_rate"], 0.75)

            best_state = torch.load(paths["best"], map_location="cpu")
            final_state = torch.load(paths["final"], map_location="cpu")
            for key in best_state:
                self.assertTrue(torch.equal(best_state[key], evaluated_states[0][key]))
                self.assertTrue(torch.equal(final_state[key], evaluated_states[1][key]))

            best_stats = torch.load(paths["best_stats"], map_location="cpu")
            self.assertEqual(best_stats["selection_metric"], "success_rate")
            self.assertEqual(best_stats["selection_epoch"], 1)
            history = json.loads(Path(paths["eval_history"]).read_text())
            self.assertEqual([entry["epoch"] for entry in history], [1, 2])
            self.assertEqual([entry["improved"] for entry in history], [True, False])

            self.assertEqual(push.call_count, 2)
            first_upload = push.call_args_list[0].kwargs
            self.assertEqual(first_upload["path_prefix"], "raw-cls-seed-42")
            self.assertEqual(
                {Path(path).name for path in first_upload["file_paths"]},
                {
                    "policy_best.pth",
                    "policy_best_stats.pth",
                    "policy_final.pth",
                    "policy_final_stats.pth",
                    "policy_eval_history.json",
                    "policy_run_config.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
