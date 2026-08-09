import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import torch

from src.bc.models.policy.latent_bc_policy import LatentBCPolicy
from src.evaluation.agents import (
    LatentChunkAgent,
    bc_training_observation_resolution,
    load_bc_components,
    load_ppo_components,
)
from src.envs import PUSHT_RENDER_SHAPE
from src.evaluation.evaluate_pusht import (
    _write_metrics,
    build_parser,
    create_run_directory,
    evaluate_from_args,
    sample_episode_seeds,
)
from src.evaluation.pusht import (
    PushTEvalConfig,
    aggregate_evaluation_results,
    make_evaluation_env,
    run_evaluation,
    run_repeated_evaluation,
)


class FakePushTEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.step_index = 0
        self.episode_seed = 0

    def reset(self, seed=None, options=None):
        self.step_index = 0
        self.episode_seed = int(seed or 0)
        observation = np.full((8, 8, 3), self.episode_seed % 255, dtype=np.uint8)
        return observation, {}

    def step(self, action):
        self.step_index += 1
        reward = -float(np.linalg.norm(action))
        terminated = self.step_index == 3
        observation = np.full((8, 8, 3), self.step_index, dtype=np.uint8)
        info = {
            "success": float(terminated),
            "block_state_dist": float(3 - self.step_index),
        }
        return observation, reward, terminated, False, info


class ConstantAgent:
    agent_type = "constant"
    metadata = {
        "source": "test",
        "training_observation_resolution": PUSHT_RENDER_SHAPE[0],
    }

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_seeds = []

    def reset(self, seed):
        self.reset_seeds.append(seed)

    def act(self, observation, info):
        return self.action


class DummyEncoder(torch.nn.Module):
    def forward(self, images):
        return images.float().mean(dim=(2, 3))[:, :2]


class EvaluationRunnerTests(unittest.TestCase):
    def test_cli_defaults_to_one_master_seed_and_150_episodes(self):
        parser = build_parser()
        self.assertEqual(parser.get_default("episodes"), 150)
        self.assertEqual(parser.get_default("seed"), 42)
        self.assertNotIn("repeats", {action.dest for action in parser._actions})
        self.assertNotIn("seed_stride", {action.dest for action in parser._actions})
        self.assertEqual(
            parser.get_default("observation_resolution"), PUSHT_RENDER_SHAPE[0]
        )

    def test_sampled_episode_seeds_are_reproducible_unique_and_well_separated(self):
        seeds = sample_episode_seeds(42, 150)
        self.assertEqual(seeds, sample_episode_seeds(42, 150))
        self.assertNotEqual(seeds, sample_episode_seeds(43, 150))
        self.assertEqual(len(seeds), 150)
        self.assertEqual(len(seeds), len(set(seeds)))
        ordered = sorted(seeds)
        self.assertGreaterEqual(
            min(right - left for left, right in zip(ordered, ordered[1:])),
            7,
        )

    def test_cli_samples_episode_seeds_from_one_master_seed(self):
        with TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--agent-type",
                    "bc",
                    "--checkpoint",
                    "test.pt",
                    "--episodes",
                    "2",
                    "--seed",
                    "7",
                    "--output-root",
                    temporary_dir,
                ]
            )

            def evaluate_fake_env(agent, config, env=None):
                self.assertIsNone(env)
                return run_evaluation(agent, config, env=FakePushTEnv())

            agent = ConstantAgent([0.0, 0.0])
            with (
                patch(
                    "src.evaluation.evaluate_pusht.make_bc_evaluation_agent",
                    return_value=agent,
                ),
                patch(
                    "src.evaluation.evaluate_pusht.run_evaluation",
                    side_effect=evaluate_fake_env,
                ),
            ):
                evaluate_from_args(args)

        self.assertEqual(agent.reset_seeds, sample_episode_seeds(7, 2))

    def test_cli_rejects_unknown_legacy_training_resolution(self):
        args = build_parser().parse_args(
            ["--agent-type", "bc", "--checkpoint", "legacy.pt"]
        )
        agent = ConstantAgent([0.0, 0.0])
        agent.metadata = {"source": "legacy"}
        with (
            patch(
                "src.evaluation.evaluate_pusht.make_bc_evaluation_agent",
                return_value=agent,
            ),
            self.assertRaisesRegex(ValueError, "legacy checkpoint"),
        ):
            evaluate_from_args(args)

    def test_cli_rejects_resolution_mismatch_before_allocating_run(self):
        with TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--agent-type",
                    "bc",
                    "--checkpoint",
                    "test.pt",
                    "--output-root",
                    temporary_dir,
                ]
            )
            agent = ConstantAgent([0.0, 0.0])
            agent.metadata = {"training_observation_resolution": 96}
            with (
                patch(
                    "src.evaluation.evaluate_pusht.make_bc_evaluation_agent",
                    return_value=agent,
                ),
                self.assertRaisesRegex(ValueError, "does not match model training"),
            ):
                evaluate_from_args(args)
            self.assertEqual(list(Path(temporary_dir).iterdir()), [])

    def test_shared_repeated_runner_accepts_an_environment_factory(self):
        agent = ConstantAgent([0.0, 0.0])
        factory_seeds = []

        def factory(config):
            factory_seeds.append(config.seed)
            return FakePushTEnv()

        result = run_repeated_evaluation(
            agent,
            PushTEvalConfig(episodes=2, seed=10),
            repeats=2,
            env_factory=factory,
        )
        payload = result.to_dict()

        self.assertEqual(factory_seeds, [10, 12])
        self.assertEqual(agent.reset_seeds, [10, 11, 12, 13])
        self.assertEqual(payload["config"]["repeat_seeds"], [10, 12])
        self.assertEqual(len(payload["repeat_summaries"]), 2)
        self.assertEqual(len(payload["episodes"]), 4)

    def test_reward_mode_is_not_an_evaluation_option(self):
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("reward_mode", destinations)
        self.assertNotIn("backend", destinations)
        self.assertNotIn("task_mode", destinations)
        self.assertNotIn("output_json", destinations)
        self.assertNotIn("video_dir", destinations)
        self.assertIn("output_root", destinations)
        self.assertFalse(hasattr(PushTEvalConfig(), "reward_mode"))
        self.assertFalse(hasattr(PushTEvalConfig(), "task_mode"))

    def test_eval_artifacts_share_one_timestamped_run_directory(self):
        with TemporaryDirectory() as temporary_dir:
            run_dir = create_run_directory(
                temporary_dir,
                "ppo",
                "checkpoints/pusht.pt",
                run_name="held out",
                timestamp="20260716T120000_000000Z",
            )
            self.assertEqual(
                run_dir.name,
                "20260716T120000_000000Z_ppo_pusht_held-out",
            )
            result = run_evaluation(
                ConstantAgent([0.0, 0.0]),
                PushTEvalConfig(episodes=1),
                env=FakePushTEnv(),
            )
            metrics_path = _write_metrics(result, run_dir)
            self.assertEqual(metrics_path, Path(run_dir) / "metrics.json")
            self.assertTrue(metrics_path.is_file())

    def test_block_start_radius_is_the_complete_near_goal_contract(self):
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("block_start_near_goal", destinations)
        self.assertIsNone(PushTEvalConfig().block_start_radius)
        self.assertEqual(PushTEvalConfig(block_start_radius=200).block_start_radius, 200)

    def test_observation_resolution_is_explicit_and_validated(self):
        self.assertEqual(
            PushTEvalConfig().observation_resolution, PUSHT_RENDER_SHAPE[0]
        )
        with self.assertRaises(ValueError):
            PushTEvalConfig(observation_resolution=0).validate()

    def test_bc_training_resolution_supports_new_and_transitional_stats(self):
        self.assertEqual(
            bc_training_observation_resolution({"observation_resolution": 96}), 96
        )
        self.assertEqual(
            bc_training_observation_resolution({"source_image_shape": [224, 224]}),
            224,
        )
        self.assertIsNone(bc_training_observation_resolution({"image_size": [224, 224]}))

    def test_known_training_resolution_mismatch_is_rejected_unless_explicit(self):
        agent = ConstantAgent([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "does not match model training"):
            run_evaluation(
                agent,
                PushTEvalConfig(episodes=1, observation_resolution=96),
                env=FakePushTEnv(),
            )
        result = run_evaluation(
            agent,
            PushTEvalConfig(
                episodes=1,
                observation_resolution=96,
                allow_resolution_mismatch=True,
            ),
            env=FakePushTEnv(),
        )
        self.assertEqual(result.summary["episodes"], 1)

    @patch("src.evaluation.pusht.make_pusht_env")
    def test_block_start_radius_enables_near_goal_wrapper(self, make_env):
        make_env.return_value = FakePushTEnv()
        env = make_evaluation_env(
            PushTEvalConfig(block_start_radius=200, observation_resolution=96)
        )
        try:
            kwargs = make_env.call_args.kwargs
            self.assertTrue(kwargs["align_sampled_goal_to_fixed_target"])
            self.assertTrue(kwargs["block_start_near_goal"])
            self.assertEqual(kwargs["block_start_radius"], 200)
            self.assertEqual(kwargs["resolution"], 96)
        finally:
            env.close()

    @patch("src.evaluation.pusht.make_pusht_env")
    def test_omitted_block_start_radius_disables_near_goal_wrapper(self, make_env):
        make_env.return_value = FakePushTEnv()
        env = make_evaluation_env(PushTEvalConfig())
        try:
            kwargs = make_env.call_args.kwargs
            self.assertTrue(kwargs["align_sampled_goal_to_fixed_target"])
            self.assertFalse(kwargs["block_start_near_goal"])
        finally:
            env.close()

    def test_same_actions_produce_identical_results(self):
        config = PushTEvalConfig(episodes=2, seed=7, capture_traces=True)
        first = run_evaluation(ConstantAgent([0.25, -0.5]), config, env=FakePushTEnv())
        second = run_evaluation(ConstantAgent([0.25, -0.5]), config, env=FakePushTEnv())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.summary["success_rate"], 1.0)
        self.assertEqual(first.episodes[0].final_metrics["block_state_dist"], 0.0)
        self.assertEqual(first.episodes[0].actions, [[0.25, -0.5]] * 3)

    def test_seed_range_is_owned_by_evaluator(self):
        agent = ConstantAgent([0.0, 0.0])
        config = PushTEvalConfig(episodes=3, seed=41)
        run_evaluation(agent, config, env=FakePushTEnv())
        self.assertEqual(agent.reset_seeds, [41, 42, 43])

    def test_explicit_episode_seed_schedule_overrides_legacy_sequence(self):
        agent = ConstantAgent([0.0, 0.0])
        config = PushTEvalConfig(
            episodes=3,
            seed=41,
            episode_seeds=(700, 70, 7000),
        )
        result = run_evaluation(agent, config, env=FakePushTEnv())
        self.assertEqual(agent.reset_seeds, [700, 70, 7000])
        self.assertEqual([episode.seed for episode in result.episodes], [700, 70, 7000])

    def test_explicit_episode_seed_schedule_must_be_unique_and_complete(self):
        with self.assertRaisesRegex(ValueError, "one seed per episode"):
            PushTEvalConfig(episodes=2, episode_seeds=(7,)).validate()
        with self.assertRaisesRegex(ValueError, "must be unique"):
            PushTEvalConfig(episodes=2, episode_seeds=(7, 7)).validate()

    def test_repeated_results_are_pooled_and_keep_per_repeat_summaries(self):
        first = run_evaluation(
            ConstantAgent([0.0, 0.0]),
            PushTEvalConfig(episodes=2, seed=10),
            env=FakePushTEnv(),
        )
        second = run_evaluation(
            ConstantAgent([0.5, 0.0]),
            PushTEvalConfig(episodes=2, seed=12),
            env=FakePushTEnv(),
        )
        aggregate = aggregate_evaluation_results([first, second])
        payload = aggregate.to_dict()

        self.assertEqual(aggregate.summary["repeats"], 2)
        self.assertEqual(aggregate.summary["episodes_per_repeat"], 2)
        self.assertEqual(aggregate.summary["episodes"], 4)
        self.assertAlmostEqual(
            aggregate.summary["mean_return"],
            (first.summary["mean_return"] + second.summary["mean_return"]) / 2,
        )
        self.assertEqual(payload["config"]["repeat_seeds"], [10, 12])
        self.assertEqual(len(payload["repeat_summaries"]), 2)
        self.assertEqual(
            [episode["repeat"] for episode in payload["episodes"]],
            [0, 0, 1, 1],
        )

    def test_invalid_action_shape_is_rejected(self):
        config = PushTEvalConfig(episodes=1)
        with self.assertRaises(ValueError):
            run_evaluation(ConstantAgent([0.0]), config, env=FakePushTEnv())


class LatentChunkAgentTests(unittest.TestCase):
    def make_agent(self, execution_mode, replan_interval=1):
        calls = []

        def predict(stacked, deterministic):
            calls.append(stacked.clone())
            return torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        agent = LatentChunkAgent(
            agent_type="test",
            encoder=DummyEncoder(),
            predict_chunk=predict,
            contract={
                "frame_stack": 2,
                "frame_stride": 1,
                "action_chunk_size": 3,
                "latent_dim": 2,
                "hidden_dim": 4,
                "action_dim": 2,
            },
            device="cpu",
            execution_mode=execution_mode,
            replan_interval=replan_interval,
        )
        agent.reset(1)
        return agent, calls

    def test_open_loop_queries_once_per_chunk(self):
        agent, calls = self.make_agent("open-loop")
        observation = np.ones((8, 8, 3), dtype=np.uint8)
        actions = [agent.act(observation, {}) for _ in range(3)]
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.allclose(actions[2], [0.5, 0.6]))

    def test_receding_horizon_replans_at_interval(self):
        agent, calls = self.make_agent("receding-horizon", replan_interval=1)
        observation = np.ones((8, 8, 3), dtype=np.uint8)
        for _ in range(3):
            agent.act(observation, {})
        self.assertEqual(len(calls), 3)

    def test_temporal_ensemble_queries_every_step(self):
        agent, calls = self.make_agent("temporal-ensemble")
        observation = np.ones((8, 8, 3), dtype=np.uint8)
        for _ in range(3):
            agent.act(observation, {})
        self.assertEqual(len(calls), 3)

    def test_bc_loader_restores_projected_latent_contract(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "policy.pth"
            stats_path = root / "policy_stats.pth"
            policy = LatentBCPolicy(
                latent_dim=3,
                frame_stack=1,
                action_dim=2,
                hidden_dim=4,
                action_chunk_size=1,
            )
            torch.save(policy.state_dict(), checkpoint)
            torch.save(
                {
                    "latent_dim": 3,
                    "frame_stack": 1,
                    "action_dim": 2,
                    "hidden_dim": 4,
                    "action_chunk_size": 1,
                    "latent_representation": "projected",
                    "image_normalization": "imagenet",
                },
                stats_path,
            )
            frozen_encoder = MagicMock(spec=torch.nn.Module)
            with patch(
                "src.evaluation.agents.LeWMEncoder.from_checkpoint",
                return_value=frozen_encoder,
            ) as load_encoder:
                components = load_bc_components(
                    str(checkpoint), str(stats_path), device="cpu"
                )

            self.assertIs(components.encoder, frozen_encoder)
            self.assertEqual(components.contract["latent_representation"], "projected")
            self.assertEqual(
                load_encoder.call_args.kwargs["latent_representation"], "projected"
            )

    def test_bc_loader_defaults_to_raw_cls_latents(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "policy.pth"
            stats_path = root / "policy_stats.pth"
            policy = LatentBCPolicy(
                latent_dim=3,
                frame_stack=1,
                action_dim=2,
                hidden_dim=4,
                action_chunk_size=1,
            )
            torch.save(policy.state_dict(), checkpoint)
            torch.save(
                {
                    "latent_dim": 3,
                    "frame_stack": 1,
                    "action_dim": 2,
                    "hidden_dim": 4,
                    "action_chunk_size": 1,
                    "image_normalization": "imagenet",
                },
                stats_path,
            )
            frozen_encoder = MagicMock(spec=torch.nn.Module)
            with patch(
                "src.evaluation.agents.LeWMEncoder.from_checkpoint",
                return_value=frozen_encoder,
            ):
                components = load_bc_components(
                    str(checkpoint), str(stats_path), device="cpu"
                )

            self.assertIs(components.encoder, frozen_encoder)

    def test_ppo_loader_restores_projected_latent_contract(self):
        with TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "ppo.pt"
            contract = {
                "frame_stack": 1,
                "frame_stride": 1,
                "action_chunk_size": 1,
                "latent_dim": 3,
                "hidden_dim": 4,
                "action_dim": 2,
                "latent_representation": "projected",
            }
            torch.save(
                {
                    "agent": {},
                    "config": {**contract, "init_log_std": -2.0},
                    "contract": contract,
                },
                checkpoint,
            )
            frozen_encoder = MagicMock(spec=torch.nn.Module)
            fake_agent = MagicMock(spec=torch.nn.Module)
            fake_agent.load_state_dict.return_value = ([], [])
            with (
                patch(
                    "src.evaluation.agents.LeWMEncoder.from_checkpoint",
                    return_value=frozen_encoder,
                ) as load_encoder,
                patch(
                    "src.evaluation.agents.build_latent_agent",
                    return_value=fake_agent,
                ),
            ):
                components = load_ppo_components(str(checkpoint), device="cpu")

            self.assertIs(components.encoder, frozen_encoder)
            self.assertEqual(components.contract["latent_representation"], "projected")
            self.assertEqual(
                load_encoder.call_args.kwargs["latent_representation"], "projected"
            )


if __name__ == "__main__":
    unittest.main()
