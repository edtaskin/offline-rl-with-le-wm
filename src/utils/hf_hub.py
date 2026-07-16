from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class HubUploadResult:
    repo_id: str
    repo_type: str
    repo_url: str
    uploaded_files: tuple[str, ...]


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
