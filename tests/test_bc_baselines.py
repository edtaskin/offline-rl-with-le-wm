import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.bc.dataset import (
    PUSHT_STATE_FEATURE_DIM,
    PushTLatentDataset,
    PushTStateDataset,
    pusht_state_features,
)
from src.bc.models.policy.state_bc_policy import StateBCPolicy
from src.evaluation.baseline_agents import (
    StateChunkAgent,
    load_state_bc_components,
    make_state_bc_evaluation_agent,
    pusht_state_from_info,
)


def _write_expert_dataset(path):
    states = np.zeros((7, 5), dtype=np.float32)
    states[:, :2] = np.arange(14, dtype=np.float32).reshape(7, 2)
    states[:, 2:4] = np.arange(14, dtype=np.float32).reshape(7, 2) + 100.0
    states[:, 4] = np.linspace(0.0, 6.2, 7, dtype=np.float32)
    actions = states[:, :2] + np.array([50.0, -150.0], dtype=np.float32)
    np.savez(
        path,
        images=np.zeros((7, 4, 4, 3), dtype=np.uint8),
        states=states,
        actions=actions,
        episode_ends=np.array([4, 7]),
    )
    return states


class StateFeatureTests(unittest.TestCase):
    def test_angle_is_encoded_continuously_across_the_wrap_point(self):
        before = pusht_state_features(torch.tensor([[0.0, 0.0, 0.0, 0.0, 2 * np.pi - 1e-4]]))
        after = pusht_state_features(torch.tensor([[0.0, 0.0, 0.0, 0.0, 1e-4]]))
        self.assertLess(float((before - after).abs().max()), 1e-3)

    def test_positions_pass_through_and_dimension_is_stable(self):
        features = pusht_state_features(torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0]]))
        self.assertEqual(features.shape, (1, PUSHT_STATE_FEATURE_DIM))
        self.assertTrue(torch.equal(features[0, :4], torch.tensor([1.0, 2.0, 3.0, 4.0])))
        self.assertAlmostEqual(float(features[0, 4]), 0.0, places=6)
        self.assertAlmostEqual(float(features[0, 5]), 1.0, places=6)

    def test_short_states_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 5 columns"):
            pusht_state_features(torch.zeros(2, 4))


class StateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.data_path = root / "expert.npz"
        self.cache_path = root / "latents.pt"
        _write_expert_dataset(self.data_path)
        torch.save(
            {"latents": torch.zeros(7, 3), "metadata": {"format_version": 1}},
            self.cache_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_state_dataset_matches_the_latent_temporal_contract(self):
        state_dataset = PushTStateDataset(
            self.data_path, frame_stack=3, frame_stride=2, action_chunk_size=3
        )
        latent_dataset = PushTLatentDataset(
            self.data_path,
            self.cache_path,
            frame_stack=3,
            frame_stride=2,
            action_chunk_size=3,
        )
        self.assertEqual(len(state_dataset), len(latent_dataset))
        for index in range(len(state_dataset)):
            state_history, state_actions = state_dataset[index]
            _, latent_actions = latent_dataset[index]
            self.assertEqual(state_history.shape, (3, PUSHT_STATE_FEATURE_DIM))
            # Only the representation differs; the supervision must not.
            self.assertTrue(torch.equal(state_actions, latent_actions))
        self.assertEqual(state_dataset._get_frame_indices(4, 4), [4, 4, 4])
        self.assertEqual(state_dataset._get_action_indices(3, 4), [3, 3, 3])

    def test_stats_declare_the_state_observation_space(self):
        dataset = PushTStateDataset(self.data_path, frame_stack=2, frame_stride=1)
        self.assertEqual(dataset.stats["observation_space"], "state")
        self.assertEqual(dataset.stats["latent_dim"], PUSHT_STATE_FEATURE_DIM)

    def test_feature_normalization_matches_the_stored_features(self):
        dataset = PushTStateDataset(self.data_path, frame_stack=2, frame_stride=1)
        mean, std = dataset.feature_normalization()
        self.assertTrue(torch.allclose(mean, dataset.features.mean(dim=0)))
        self.assertTrue(torch.allclose(std, dataset.features.std(dim=0)))


class StatePolicyTests(unittest.TestCase):
    def test_normalization_survives_a_checkpoint_round_trip(self):
        mean = torch.arange(PUSHT_STATE_FEATURE_DIM, dtype=torch.float32)
        std = torch.full((PUSHT_STATE_FEATURE_DIM,), 3.0)
        policy = StateBCPolicy(
            frame_stack=3, hidden_dim=8, action_chunk_size=3, feature_mean=mean, feature_std=std
        )
        features = torch.randn(2, 3, PUSHT_STATE_FEATURE_DIM)
        restored = StateBCPolicy(frame_stack=3, hidden_dim=8, action_chunk_size=3)
        restored.load_state_dict(policy.state_dict())
        self.assertTrue(torch.equal(policy(features), restored(features)))
        self.assertEqual(policy(features).shape, (2, 3, 2))

    def test_degenerate_std_cannot_divide_by_zero(self):
        policy = StateBCPolicy(
            frame_stack=1,
            hidden_dim=4,
            action_chunk_size=1,
            feature_std=torch.zeros(PUSHT_STATE_FEATURE_DIM),
        )
        output = policy(torch.zeros(1, 1, PUSHT_STATE_FEATURE_DIM))
        self.assertTrue(torch.isfinite(output).all())

    def test_mismatched_normalization_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalization statistics"):
            StateBCPolicy(feature_mean=torch.zeros(3))


class StateAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint_path = self.root / "state_bc.pth"
        self.stats_path = self.root / "state_bc_stats.pth"
        self.contract = {
            "frame_stack": 3,
            "frame_stride": 5,
            "action_chunk_size": 5,
            "latent_dim": PUSHT_STATE_FEATURE_DIM,
            "hidden_dim": 16,
            "action_dim": 2,
        }
        policy = StateBCPolicy(
            feature_dim=PUSHT_STATE_FEATURE_DIM,
            frame_stack=3,
            hidden_dim=16,
            action_chunk_size=5,
        )
        torch.save(policy.state_dict(), self.checkpoint_path)
        torch.save(
            {**self.contract, "observation_space": "state", "observation_resolution": 224},
            self.stats_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_state_is_assembled_from_the_env_info_dict(self):
        info = {
            "pos_agent": np.array([10.0, 20.0]),
            "block_pose": np.array([30.0, 40.0, 1.5]),
        }
        state = pusht_state_from_info(info)
        self.assertTrue(np.allclose(state, [10.0, 20.0, 30.0, 40.0, 1.5]))

    def test_missing_state_keys_are_reported(self):
        with self.assertRaisesRegex(KeyError, "block_pose"):
            pusht_state_from_info({"pos_agent": np.zeros(2)})

    def test_loader_rejects_a_pixel_checkpoint(self):
        stats_path = self.root / "pixel_stats.pth"
        torch.save({**self.contract, "observation_space": "pixels"}, stats_path)
        with self.assertRaisesRegex(ValueError, "observation_space"):
            load_state_bc_components(str(self.checkpoint_path), str(stats_path), device="cpu")

    def test_agent_acts_from_info_and_reuses_the_chunk_queue(self):
        agent = make_state_bc_evaluation_agent(
            checkpoint=str(self.checkpoint_path),
            stats_path=str(self.stats_path),
            device="cpu",
        )
        self.assertIsInstance(agent, StateChunkAgent)
        self.assertEqual(agent.agent_type, "bc-state")
        self.assertEqual(agent.metadata["training_observation_resolution"], 224)
        agent.reset(0)
        info = {
            "pos_agent": np.array([10.0, 20.0]),
            "block_pose": np.array([30.0, 40.0, 1.5]),
        }
        # The observation is ignored entirely; only ``info`` reaches the policy.
        actions = [agent.act(None, info) for _ in range(6)]
        for action in actions:
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.all(np.abs(action) <= 1.0))
        # One open-loop chunk of five, then a replan on the sixth step.
        self.assertEqual(len(agent.history), 6)

    def test_agent_without_info_fails_loudly(self):
        agent = make_state_bc_evaluation_agent(
            checkpoint=str(self.checkpoint_path),
            stats_path=str(self.stats_path),
            device="cpu",
        )
        agent.reset(0)
        with self.assertRaisesRegex(ValueError, "info dict"):
            agent.act(None, None)


if __name__ == "__main__":
    unittest.main()
