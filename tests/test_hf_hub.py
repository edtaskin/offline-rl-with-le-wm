import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.utils.hf_hub import (
    parse_hf_artifact_reference,
    resolve_artifact,
    resolve_artifacts,
)


class HuggingFaceArtifactTests(unittest.TestCase):
    def test_explicit_hf_uri_is_parsed(self):
        reference = parse_hf_artifact_reference("hf://owner/repo/models/best.pt")
        self.assertEqual(reference.repo_id, "owner/repo")
        self.assertEqual(reference.filename, "models/best.pt")

    def test_bc_aliases_share_one_cached_snapshot(self):
        with TemporaryDirectory() as temporary_dir:
            snapshot = Path(temporary_dir)
            (snapshot / "pusht_latent_bc.pth").touch()
            (snapshot / "pusht_latent_bc_stats.pth").touch()
            with patch(
                "huggingface_hub.snapshot_download",
                return_value=str(snapshot),
            ) as download:
                weights, stats = resolve_artifacts(
                    [
                        "checkpoints/trained_policies/pusht_latent_bc.pth",
                        "checkpoints/trained_policies/pusht_latent_bc_stats.pth",
                    ]
                )
            self.assertEqual(weights, snapshot / "pusht_latent_bc.pth")
            self.assertEqual(stats, snapshot / "pusht_latent_bc_stats.pth")
            download.assert_called_once()
            self.assertEqual(
                download.call_args.kwargs["allow_patterns"],
                ["pusht_latent_bc.pth", "pusht_latent_bc_stats.pth"],
            )

    def test_unknown_checkpoint_path_never_falls_back_to_local_file(self):
        with self.assertRaisesRegex(ValueError, "Local checkpoint reads are disabled"):
            resolve_artifact("checkpoints/unknown/model.pt")

    def test_non_checkpoint_experiment_path_remains_local(self):
        with TemporaryDirectory() as temporary_dir:
            artifact = Path(temporary_dir) / "model.pt"
            artifact.touch()
            self.assertEqual(resolve_artifact(artifact), artifact)


if __name__ == "__main__":
    unittest.main()
