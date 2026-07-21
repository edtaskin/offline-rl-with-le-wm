import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from src.evaluation.agents import LatentChunkAgent
from src.evaluation.evaluate_pusht import (
    _write_metrics,
    build_parser,
    create_run_directory,
    evaluate_from_args,
    make_repeat_seeds,
)
from src.evaluation.pusht import (
    PushTEvalConfig,
    aggregate_evaluation_results,
    make_evaluation_env,
    run_evaluation,
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
    metadata = {"source": "test"}

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
    def test_cli_defaults_to_three_repeats_of_fifty_episodes(self):
        parser = build_parser()
        episodes = parser.get_default("episodes")
        repeats = parser.get_default("repeats")
        self.assertEqual(episodes, 50)
        self.assertEqual(repeats, 3)
        self.assertEqual(make_repeat_seeds(42, repeats, episodes), [42, 92, 142])

    def test_repeat_seed_ranges_do_not_overlap(self):
        seeds = make_repeat_seeds(7, repeats=3, episodes=2)
        episode_seeds = [
            seed + episode for seed in seeds for episode in range(2)
        ]
        self.assertEqual(seeds, [7, 9, 11])
        self.assertEqual(len(episode_seeds), len(set(episode_seeds)))

    def test_each_repeat_uses_its_seed_once(self):
        with TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--agent-type",
                    "bc",
                    "--checkpoint",
                    "test.pt",
                    "--episodes",
                    "1",
                    "--repeats",
                    "2",
                    "--seed",
                    "7",
                    "--output-root",
                    temporary_dir,
                ]
            )

            def evaluate_fake_env(agent, config):
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

        self.assertEqual(agent.reset_seeds, [7, 8])

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

    @patch("src.evaluation.pusht.make_pusht_env")
    def test_block_start_radius_enables_near_goal_wrapper(self, make_env):
        make_env.return_value = FakePushTEnv()
        env = make_evaluation_env(PushTEvalConfig(block_start_radius=200))
        try:
            kwargs = make_env.call_args.kwargs
            self.assertTrue(kwargs["align_sampled_goal_to_fixed_target"])
            self.assertTrue(kwargs["block_start_near_goal"])
            self.assertEqual(kwargs["block_start_radius"], 200)
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

if __name__ == "__main__":
    unittest.main()
