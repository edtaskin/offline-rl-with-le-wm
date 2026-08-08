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
from src.bc.models.policy.cnn_bc_policy import CNNBCPolicy, CNNEncoder, SpatialSoftmax
from src.bc.models.policy.state_bc_policy import StateBCPolicy
from src.bc.train_bc_baseline import random_shift
from src.evaluation.baseline_agents import (
    StateChunkAgent,
    load_cnn_bc_components,
    load_state_bc_components,
    make_cnn_bc_evaluation_agent,
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


class SpatialSoftmaxTests(unittest.TestCase):
    def test_keypoint_tracks_the_activation_peak(self):
        layer = SpatialSoftmax(1, 8, 8)
        for row, col, expected in [(0, 0, (-1.0, -1.0)), (7, 7, (1.0, 1.0))]:
            features = torch.full((1, 1, 8, 8), -20.0)
            features[0, 0, row, col] = 20.0
            keypoint = layer(features)[0]
            self.assertAlmostEqual(float(keypoint[0]), expected[0], places=2)
            self.assertAlmostEqual(float(keypoint[1]), expected[1], places=2)

    def test_output_width_is_two_per_channel(self):
        layer = SpatialSoftmax(5, 4, 4)
        self.assertEqual(layer(torch.randn(3, 5, 4, 4)).shape, (3, 10))


class CNNEncoderTests(unittest.TestCase):
    def test_normalization_is_dtype_driven_not_value_driven(self):
        encoder = CNNEncoder(feature_dim=16, input_resolution=32, num_keypoints=4).eval()
        # A dark frame is where a `max() > 1.5` heuristic would have diverged.
        dark = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        with torch.no_grad():
            from_uint8 = encoder(dark)
            from_float = encoder(dark.float() / 255.0)
        self.assertTrue(torch.allclose(from_uint8, from_float, atol=1e-6))

    def test_uint8_and_scaled_float_inputs_agree(self):
        encoder = CNNEncoder(feature_dim=16, input_resolution=32, num_keypoints=4).eval()
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(encoder(images), encoder(images.float() / 255.0), atol=1e-5)
            )

    def test_rank_3_input_is_rejected(self):
        encoder = CNNEncoder(feature_dim=16, input_resolution=32, num_keypoints=4)
        with self.assertRaisesRegex(ValueError, r"\(B, C, H, W\)"):
            encoder(torch.zeros(3, 32, 32))


class RandomShiftTests(unittest.TestCase):
    def test_one_offset_per_sample_shared_across_frames(self):
        torch.manual_seed(0)
        images = torch.zeros(6, 3, 3, 32, 32)
        images[:, :, :, 16, 16] = 1.0
        shifted = random_shift(images, 4)
        self.assertEqual(shifted.shape, images.shape)
        for sample in range(6):
            locations = [
                (shifted[sample, frame, 0] > 0.5).nonzero().tolist() for frame in range(3)
            ]
            self.assertEqual(locations[0], locations[1])
            self.assertEqual(locations[1], locations[2])

    def test_offsets_differ_across_samples(self):
        torch.manual_seed(0)
        images = torch.zeros(16, 1, 3, 32, 32)
        images[:, :, :, 16, 16] = 1.0
        shifted = random_shift(images, 4)
        offsets = {
            tuple((shifted[sample, 0, 0] > 0.5).nonzero()[0].tolist()) for sample in range(16)
        }
        self.assertGreater(len(offsets), 1)

    def test_non_square_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "square"):
            random_shift(torch.zeros(1, 1, 3, 16, 32), 2)


class CNNAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint_path = self.root / "cnn_bc.pth"
        self.stats_path = self.root / "cnn_bc_stats.pth"
        self.contract = {
            "frame_stack": 3,
            "frame_stride": 5,
            "action_chunk_size": 5,
            "latent_dim": 16,
            "hidden_dim": 8,
            "action_dim": 2,
        }
        policy = CNNBCPolicy(
            feature_dim=16,
            frame_stack=3,
            hidden_dim=8,
            action_chunk_size=5,
            input_resolution=32,
            num_keypoints=4,
        )
        torch.save(policy.state_dict(), self.checkpoint_path)
        torch.save(
            {
                **self.contract,
                "observation_space": "pixels",
                "observation_resolution": 32,
                "cnn_keypoints": 4,
            },
            self.stats_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loader_rejects_a_state_checkpoint(self):
        stats_path = self.root / "state_stats.pth"
        torch.save({**self.contract, "observation_space": "state"}, stats_path)
        with self.assertRaisesRegex(ValueError, "observation_space"):
            load_cnn_bc_components(str(self.checkpoint_path), str(stats_path), device="cpu")

    def test_agent_acts_from_pixels(self):
        agent = make_cnn_bc_evaluation_agent(
            checkpoint=str(self.checkpoint_path),
            stats_path=str(self.stats_path),
            device="cpu",
        )
        self.assertEqual(agent.agent_type, "bc-cnn")
        self.assertEqual(agent.metadata["training_observation_resolution"], 32)
        agent.reset(0)
        observation = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        for _ in range(6):
            action = agent.act(observation, {})
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_training_and_evaluation_encoders_agree(self):
        components = load_cnn_bc_components(
            str(self.checkpoint_path), str(self.stats_path), device="cpu"
        )
        policy = components.policy
        frames = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
        with torch.no_grad():
            # Training path: the trainer scales uint8 to float [0, 1] first.
            train_features = policy.encode_stack(frames.unsqueeze(0).float() / 255.0)[0]
            # Evaluation path: the agent hands the encoder uint8 frames directly.
            eval_features = torch.stack(
                [policy.encoder(frames[index].unsqueeze(0))[0] for index in range(3)]
            )
        self.assertTrue(torch.allclose(train_features, eval_features, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
