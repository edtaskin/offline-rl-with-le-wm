import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import src.bc.train_bc_latent as train_bc


def _args(root: Path, **overrides):
    values = {
        "data_path": str(root / "expert.npz"),
        "observation_resolution": 224,
        "checkpoint_path": str(root / "projected_bc.pth"),
        "latent_representation": "projected",
        "latent_cache_path": None,
        "rebuild_latent_cache": False,
        "disable_latent_cache": False,
        "epochs": 3,
        "batch_size": 2,
        "latent_cache_batch_size": None,
        "lr": 0.0,
        "seed": 7,
        "num_workers": 0,
        "deterministic": True,
        "hidden_dim": 4,
        "frame_stack": 1,
        "frame_stride": 1,
        "action_chunk_size": 1,
        "log_interval": 10,
        "save_interval": 10,
        "eval_interval": 1,
        "eval_episodes": 2,
        "eval_seed": None,
        "eval_max_episode_steps": 300,
        "eval_execution_mode": "open-loop",
        "eval_replan_interval": 1,
        "eval_block_start_radius": 200.0,
        "eval_unrestricted_block_starts": False,
        "eval_output_root": str(root / "evaluations"),
        "wandb": False,
        "wandb_project": "test",
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_group": "test",
        "wandb_tags": None,
        "wandb_mode": "disabled",
        "push_to_hf": False,
        "hf_repo_id": None,
        "hf_repo_type": "model",
        "hf_private": False,
        "hf_token": None,
        "hf_revision": None,
        "hf_path_prefix": None,
        "hf_commit_message": "test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepared_dataset():
    latents = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    actions = torch.zeros(4, 1, 2)
    return {
        "dataset": TensorDataset(latents, actions),
        "dataset_stats": {
            "latent_dim": 3,
            "observation_resolution": 224,
            "latent_representation": "projected",
        },
        "extractor": None,
        "latent_dim": 3,
        "use_latent_cache": True,
        "cache_path": "cache_projected.pt",
        "cache_rebuilt": False,
        "expected_metadata": {"latent_representation": "projected"},
    }


class BCCheckpointTests(unittest.TestCase):
    def test_artifact_names_are_derived_from_base_path(self):
        paths = train_bc.checkpoint_artifact_paths("runs/bc/projected.pth")
        self.assertEqual(paths["best"], "runs/bc/projected_best.pth")
        self.assertEqual(paths["best_stats"], "runs/bc/projected_best_stats.pth")
        self.assertEqual(paths["final"], "runs/bc/projected_final.pth")
        self.assertEqual(paths["final_stats"], "runs/bc/projected_final_stats.pth")
        self.assertEqual(
            paths["eval_history"], "runs/bc/projected_eval_history.json"
        )

    def test_periodic_evaluation_uses_canonical_entrypoint_and_restores_rng(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = train_bc.build_parser().parse_args(
                [
                    "--checkpoint_path",
                    str(root / "policy.pth"),
                    "--eval_episodes",
                    "2",
                ]
            )
            policy = train_bc.LatentBCPolicy(
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
            expected_torch = torch.rand(
                1, generator=torch.Generator().manual_seed(7)
            )
            with patch(
                "src.evaluation.evaluate_pusht.evaluate_from_args",
                side_effect=fake_evaluate,
            ):
                result = train_bc._run_periodic_evaluation(
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
            self.assertEqual(seen["args"].seed, 42)
            self.assertEqual(seen["args"].block_start_radius, 200.0)
            self.assertEqual(seen["args"].run_name, "policy-epoch-50")
            self.assertFalse(Path(seen["args"].checkpoint).exists())
            self.assertTrue(policy.training)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))

    def test_real_env_success_selects_best_and_persists_selection_epoch(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _args(root, epochs=2, lr=1e-2)
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
                    summary={
                        "success_rate": next(success_rates),
                        "episodes": 2,
                        "mean_length": 3.0,
                    },
                    episodes=[SimpleNamespace(length=3), SimpleNamespace(length=3)],
                    run_dir=f"eval-{epoch}",
                    metrics_path=f"eval-{epoch}/metrics.json",
                )

            with (
                patch.object(train_bc, "_prepare_dataset", return_value=_prepared_dataset()),
                patch.object(
                    train_bc,
                    "_run_periodic_evaluation",
                    side_effect=fake_evaluation,
                ),
            ):
                result = train_bc.train_latent_bc(args)

            for key in (
                "best_checkpoint_path",
                "best_stats_path",
                "final_checkpoint_path",
                "final_stats_path",
                "evaluation_history_path",
            ):
                self.assertTrue(Path(result[key]).is_file(), key)

            best_stats = torch.load(result["best_stats_path"], map_location="cpu")
            final_stats = torch.load(result["final_stats_path"], map_location="cpu")
            self.assertEqual(best_stats["checkpoint_role"], "best")
            self.assertEqual(best_stats["checkpoint_epoch"], 1)
            self.assertEqual(best_stats["best_model_epoch"], 1)
            self.assertEqual(best_stats["selection_metric"], "real_env_success_rate")
            self.assertEqual(best_stats["selection_metric_value"], 0.75)
            self.assertEqual(best_stats["selection_epoch"], 1)
            self.assertEqual(final_stats["checkpoint_role"], "final")
            self.assertEqual(final_stats["checkpoint_epoch"], 2)
            self.assertEqual(final_stats["best_model_epoch"], 1)
            self.assertEqual(final_stats["best_eval_epoch"], 1)
            self.assertEqual(final_stats["eval_env_steps_consumed"], 12)
            self.assertEqual(result["best_checkpoint_epoch"], 1)
            self.assertEqual(result["best_eval_success_rate"], 0.75)
            self.assertEqual(result["final_checkpoint_epoch"], 2)
            self.assertEqual(result["eval_env_steps_consumed"], 12)
            self.assertFalse(Path(args.checkpoint_path).exists())

            best_state = torch.load(result["best_checkpoint_path"], map_location="cpu")
            final_state = torch.load(result["final_checkpoint_path"], map_location="cpu")
            for key in best_state:
                self.assertTrue(torch.equal(best_state[key], evaluated_states[0][key]))
                self.assertTrue(torch.equal(final_state[key], evaluated_states[1][key]))

            history = json.loads(Path(result["evaluation_history_path"]).read_text())
            self.assertEqual([record["epoch"] for record in history], [1, 2])
            self.assertEqual([record["improved"] for record in history], [True, False])
            self.assertEqual(history[-1]["best_epoch"], 1)

    def test_wandb_and_hf_receive_both_checkpoint_variants(self):
        class FakeRun:
            id = "run-id"
            url = "https://wandb.invalid/run-id"

            def __init__(self):
                self.summary = {}
                self.logged = []
                self.finished = False

            def log(self, metrics, step=None):
                self.logged.append((dict(metrics), step))

            def finish(self):
                self.finished = True

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = _args(
                root,
                epochs=1,
                wandb=True,
                push_to_hf=True,
                hf_repo_id="owner/repo",
                hf_path_prefix="projected/seed7",
            )
            run = FakeRun()
            upload_result = SimpleNamespace(
                repo_id="owner/repo",
                repo_type="model",
                repo_url="https://huggingface.co/owner/repo",
                uploaded_files=(),
            )
            with (
                patch.object(train_bc, "_prepare_dataset", return_value=_prepared_dataset()),
                patch.object(
                    train_bc,
                    "_run_periodic_evaluation",
                    return_value=SimpleNamespace(
                        summary={
                            "success_rate": 0.5,
                            "episodes": 2,
                            "mean_length": 4.0,
                        },
                        episodes=[
                            SimpleNamespace(length=4),
                            SimpleNamespace(length=4),
                        ],
                        run_dir="eval-1",
                        metrics_path="eval-1/metrics.json",
                    ),
                ),
                patch.object(train_bc, "init_wandb", return_value=run),
                patch.object(train_bc, "push_files_to_hub", return_value=upload_result) as push,
                patch.object(train_bc, "log_wandb_artifact") as log_artifact,
            ):
                result = train_bc.train_latent_bc(args)

            first_upload = push.call_args_list[0].kwargs["file_paths"]
            wandb_files = log_artifact.call_args.kwargs["file_paths"]
            expected = {
                result["best_checkpoint_path"],
                result["best_stats_path"],
                result["final_checkpoint_path"],
                result["final_stats_path"],
                result["evaluation_history_path"],
            }
            self.assertTrue(expected.issubset(set(first_upload)))
            self.assertTrue(expected.issubset(set(wandb_files)))
            self.assertEqual(run.summary["best_train_loss_epoch"], 1)
            self.assertEqual(run.summary["best_eval_epoch"], 1)
            self.assertEqual(run.summary["best_eval_success_rate"], 0.5)
            self.assertTrue(any("eval/success_rate" in values for values, _ in run.logged))
            self.assertTrue(run.finished)


if __name__ == "__main__":
    unittest.main()
