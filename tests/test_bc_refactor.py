import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.bc.dataset import (
    PushTImageDataset,
    PushTLatentDataset,
    absolute_to_relative_action,
    npz_array_shape,
    resolve_pusht_observation_resolution,
)
from src.bc.history import (
    FeatureHistory,
    action_chunk_indices,
    history_indices,
    temporal_ensemble_action,
)
from src.bc.latent_cache import check_latent_cache, metadata_matches
from src.representations.lewm import LeWMEncoder
from src.bc.models.policy.latent_bc_policy import LatentBCPolicy


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.data_path = root / "expert.npz"
        self.cache_path = root / "latents.pt"
        images = np.arange(7 * 4 * 4 * 3, dtype=np.uint8).reshape(7, 4, 4, 3)
        states = np.zeros((7, 5), dtype=np.float32)
        states[:, :2] = np.arange(14, dtype=np.float32).reshape(7, 2)
        actions = states[:, :2] + np.array([50.0, -150.0], dtype=np.float32)
        np.savez(
            self.data_path,
            images=images,
            states=states,
            actions=actions,
            episode_ends=np.array([4, 7]),
        )
        torch.save(
            {
                "latents": torch.arange(21, dtype=torch.float32).reshape(7, 3),
                "metadata": {"format_version": 1},
            },
            self.cache_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_image_and_latent_datasets_share_targets_and_boundaries(self):
        image_dataset = PushTImageDataset(
            self.data_path, frame_stack=3, frame_stride=2, action_chunk_size=3
        )
        latent_dataset = PushTLatentDataset(
            self.data_path,
            self.cache_path,
            frame_stack=3,
            frame_stride=2,
            action_chunk_size=3,
        )
        for index in range(len(image_dataset)):
            image_history, image_actions = image_dataset[index]
            latent_history, latent_actions = latent_dataset[index]
            self.assertEqual(image_history.shape[0], 3)
            self.assertEqual(latent_history.shape, (3, 3))
            self.assertTrue(torch.equal(image_actions, latent_actions))
        self.assertEqual(latent_dataset._get_frame_indices(4, 4), [4, 4, 4])
        self.assertEqual(latent_dataset._get_action_indices(3, 4), [3, 3, 3])

    def test_native_observation_resolution_is_inferred_without_loading_images(self):
        self.assertEqual(npz_array_shape(self.data_path, "images"), (7, 4, 4, 3))
        self.assertEqual(resolve_pusht_observation_resolution(self.data_path), 4)
        self.assertEqual(resolve_pusht_observation_resolution(self.data_path, 4), 4)
        with self.assertRaisesRegex(ValueError, "does not match the expert dataset"):
            resolve_pusht_observation_resolution(self.data_path, 8)

    def test_tensor_only_legacy_cache_is_supported(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.pt"
        torch.save(torch.zeros(7, 3), legacy_path)
        dataset = PushTLatentDataset(self.data_path, legacy_path)
        self.assertEqual(dataset.latent_cache_metadata, {})

    def test_absolute_actions_are_converted_and_clamped(self):
        action = torch.tensor([[150.0, -150.0], [20.0, 30.0]])
        position = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        expected = torch.tensor([[1.0, -1.0], [0.1, 0.2]])
        self.assertTrue(torch.allclose(absolute_to_relative_action(action, position), expected))


class HistoryTests(unittest.TestCase):
    def test_index_helpers(self):
        self.assertEqual(history_indices(8, 5, 3, 2), [5, 6, 8])
        self.assertEqual(action_chunk_indices(8, 10, 4), [8, 9, 9, 9])

    def test_feature_history_matches_dataset_padding(self):
        history = FeatureHistory(frame_stack=3, frame_stride=2)
        history.append(torch.tensor([1.0, 2.0]))
        self.assertTrue(
            torch.equal(
                history.stacked(),
                torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]),
            )
        )
        for value in (2.0, 3.0, 4.0, 5.0):
            history.append(torch.tensor([value, -value]))
        expected = torch.tensor([[[1.0, 2.0], [3.0, -3.0], [5.0, -5.0]]])
        self.assertTrue(torch.equal(history.stacked(), expected))

    def test_temporal_ensemble(self):
        predictions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        self.assertTrue(
            torch.equal(
                temporal_ensemble_action(predictions, 0.0),
                torch.tensor([0.5, 0.5]),
            )
        )
        weighted = temporal_ensemble_action(predictions, 1.0)
        self.assertGreater(weighted[0].item(), weighted[1].item())
        with self.assertRaises(ValueError):
            temporal_ensemble_action([], 0.0)


class CacheAndPolicyTests(unittest.TestCase):
    def test_cache_validation_accepts_existing_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.pt"
            metadata = {"num_samples": 4, "latent_dim": 3, "format_version": 1}
            torch.save(
                {"latents": torch.zeros(4, 3), "metadata": metadata}, cache_path
            )
            valid, messages = check_latent_cache(cache_path, metadata)
            self.assertTrue(valid)
            self.assertEqual(messages, [])
            valid, messages = check_latent_cache(
                cache_path, {**metadata, "latent_dim": 2}
            )
            self.assertFalse(valid)
            self.assertIn("shape mismatch", messages[0])

    def test_metadata_mismatch_is_reported(self):
        valid, messages = metadata_matches({"a": 1}, {"a": 2})
        self.assertFalse(valid)
        self.assertEqual(messages, ["a: expected 2, got 1"])

    def test_policy_state_dict_shape_is_stable(self):
        old = LatentBCPolicy(
            latent_dim=3, frame_stack=2, action_dim=2, hidden_dim=4, action_chunk_size=3
        )
        new = LatentBCPolicy(
            latent_dim=3, frame_stack=2, action_dim=2, hidden_dim=4, action_chunk_size=3
        )
        new.load_state_dict(old.state_dict())
        features = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        self.assertTrue(torch.equal(old(features), new(features)))
        self.assertEqual(old(features).shape, (2, 3, 2))

    def test_frozen_extractor_outputs_can_train_policy_head(self):
        class FakeEncoder(torch.nn.Module):
            def forward(self, images):
                pooled = images.mean(dim=(1, 2, 3))
                tokens = pooled[:, None, None].repeat(1, 1, 3)
                return SimpleNamespace(last_hidden_state=tokens)

        encoder = FakeEncoder()
        extractor = LeWMEncoder(
            encoder=encoder,
            device="cpu",
            checkpoint_path="unused.ckpt",
            feature_dim=3,
        )
        features = extractor.encode(torch.rand(2, 3, 8, 8)).unsqueeze(1)
        policy = LatentBCPolicy(
            latent_dim=3, frame_stack=1, action_dim=2, hidden_dim=4
        )
        policy(features).sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
