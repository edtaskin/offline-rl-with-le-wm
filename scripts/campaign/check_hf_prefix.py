"""Fail before training if a campaign's Hub destinations are already occupied.

``push_files_to_hub`` overwrites silently, and a campaign only discovers the
collision after the run it would clobber has already finished. This checks every
target prefix up front, so a mistyped or reused namespace costs a second rather
than a night of GPU time.

Usage::

    python -m scripts.campaign.check_hf_prefix \
        --repo-id offline-rl-with-le-wm/ppo \
        --prefix rawcls_bc_best/real_sparse/seed1 rawcls_bc_best/real_sparse/seed2
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--prefix", nargs="+", required=True, help="paths that must not exist yet")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="report collisions but exit 0 (for resuming a partially finished campaign)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        existing = set(HfApi().list_repo_files(args.repo_id, repo_type=args.repo_type))
    except RepositoryNotFoundError:
        print(f"{args.repo_id} does not exist yet; nothing can be overwritten")
        return 0

    collisions = {}
    for prefix in args.prefix:
        normalized = prefix.strip("/") + "/"
        hits = sorted(f for f in existing if f.startswith(normalized))
        if hits:
            collisions[prefix] = hits

    if not collisions:
        print(f"{len(args.prefix)} destination(s) free in {args.repo_id}")
        return 0

    print(f"{len(collisions)} destination(s) already occupied in {args.repo_id}:", file=sys.stderr)
    for prefix, hits in collisions.items():
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
    if args.allow_existing:
        print("continuing anyway (--allow-existing)", file=sys.stderr)
        return 0
    print(
        "Refusing to start: pushing would overwrite these. Choose a different "
        "--hf-prefix, or pass --allow-existing to resume deliberately.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
