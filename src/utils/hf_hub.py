import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


HF_URI_PREFIX = "hf://"
CHECKPOINT_HUB_ALIASES = {
    "checkpoints/trained_policies/pusht_latent_bc.pth": (
        "offline-rl-with-le-wm/behavioral-cloning",
        "pusht_latent_bc.pth",
    ),
    "checkpoints/trained_policies/pusht_latent_bc_stats.pth": (
        "offline-rl-with-le-wm/behavioral-cloning",
        "pusht_latent_bc_stats.pth",
    ),
    "checkpoints/trained_policies/pusht_latent_bc_run_config.json": (
        "offline-rl-with-le-wm/behavioral-cloning",
        "pusht_latent_bc_run_config.json",
    ),
    "checkpoints/trained_policies/pusht_latent_ppo.pt": (
        "offline-rl-with-le-wm/ppo",
        "best.pt",
    ),
}


@dataclass(frozen=True)
class HubUploadResult:
    repo_id: str
    repo_type: str
    repo_url: str
    uploaded_files: tuple[str, ...]


@dataclass(frozen=True)
class HubArtifactReference:
    repo_id: str
    filename: str

    @property
    def uri(self) -> str:
        return f"{HF_URI_PREFIX}{self.repo_id}/{self.filename}"


def _load_hf_token(token=None):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return token or os.getenv("HF_TOKEN")


def _checkpoint_key(reference: str) -> str | None:
    normalized = reference.replace("\\", "/")
    marker = "checkpoints/"
    index = normalized.rfind(marker)
    return normalized[index:] if index >= 0 else None


def parse_hf_artifact_reference(reference) -> HubArtifactReference | None:
    value = str(reference)
    if value.startswith(HF_URI_PREFIX):
        parts = value[len(HF_URI_PREFIX) :].strip("/").split("/")
        if len(parts) < 3:
            raise ValueError(
                "Hugging Face artifacts must use hf://<owner>/<repo>/<filename>."
            )
        return HubArtifactReference(
            repo_id="/".join(parts[:2]),
            filename="/".join(parts[2:]),
        )
    checkpoint_key = _checkpoint_key(value)
    if checkpoint_key is None:
        return None
    alias = CHECKPOINT_HUB_ALIASES.get(checkpoint_key)
    if alias is None:
        raise ValueError(
            f"Local checkpoint reads are disabled and no Hugging Face alias is "
            f"registered for {checkpoint_key!r}. Use an hf://<owner>/<repo>/<filename> reference."
        )
    return HubArtifactReference(repo_id=alias[0], filename=alias[1])


def resolve_artifacts(
    references: Sequence[str | Path],
    *,
    token=None,
    revision="main",
    repo_type="model",
) -> tuple[Path, ...]:
    """Resolve local or Hub references, checking Hub revisions before cache reuse."""
    parsed = [parse_hf_artifact_reference(reference) for reference in references]
    resolved: list[Path | None] = [None] * len(references)
    grouped: dict[str, list[tuple[int, HubArtifactReference]]] = {}
    for index, (reference, hub_reference) in enumerate(zip(references, parsed)):
        if hub_reference is None:
            local_path = Path(reference)
            if not local_path.is_file():
                raise FileNotFoundError(f"Artifact not found: {local_path}")
            resolved[index] = local_path
            continue
        grouped.setdefault(hub_reference.repo_id, []).append((index, hub_reference))

    if grouped:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Hugging Face checkpoint loading requires huggingface_hub. "
                "Install requirements.txt before loading model artifacts."
            ) from exc
        token = _load_hf_token(token)
        for repo_id, entries in grouped.items():
            filenames = sorted({entry.filename for _, entry in entries})
            snapshot_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    allow_patterns=filenames,
                    token=token,
                )
            )
            for index, entry in entries:
                artifact_path = snapshot_path / entry.filename
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Hugging Face artifact {entry.uri} was not found in snapshot {snapshot_path}."
                    )
                resolved[index] = artifact_path
                print(f"Resolved {entry.uri} -> {artifact_path}")
    if any(path is None for path in resolved):
        raise RuntimeError("failed to resolve one or more artifacts")
    return tuple(resolved)  # type: ignore[arg-type]


def resolve_artifact(reference, **kwargs) -> Path:
    return resolve_artifacts([reference], **kwargs)[0]


def _repo_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


def _path_in_repo(local_path: Path, path_prefix: Optional[str]) -> str:
    clean_prefix = (path_prefix or "").strip("/")
    if clean_prefix:
        return f"{clean_prefix}/{local_path.name}"
    return local_path.name


def push_files_to_hub(
    *,
    repo_id: str,
    file_paths: Iterable[str],
    repo_type: str = "model",
    private: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    path_prefix: Optional[str] = None,
    commit_message: Optional[str] = None,
    create_repo: bool = True,
) -> HubUploadResult:
    """Create a Hugging Face repo if needed and upload local files to it."""
    if not repo_id:
        raise ValueError("repo_id is required when pushing to the Hugging Face Hub")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Hugging Face upload was requested, but huggingface_hub is not installed. "
            "Install requirements.txt or run without --push_to_hf."
        ) from exc

    local_paths = [Path(path) for path in file_paths]
    missing_paths = [str(path) for path in local_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Cannot upload missing files to Hugging Face Hub: {missing_paths}")

    api = HfApi(token=token)
    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token,
        )

    uploaded_files = []
    for local_path in local_paths:
        remote_path = _path_in_repo(local_path, path_prefix)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            commit_message=commit_message,
        )
        uploaded_files.append(remote_path)

    return HubUploadResult(
        repo_id=repo_id,
        repo_type=repo_type,
        repo_url=_repo_url(repo_id, repo_type),
        uploaded_files=tuple(uploaded_files),
    )
