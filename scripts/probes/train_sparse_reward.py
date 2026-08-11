"""Train the PushT sparse reward classifier.

This is the explicit entrypoint for the ``objective_met`` probe used as the
sparse reward classifier. It delegates to ``scripts.probes.train_state`` and
preserves all of that trainer's arguments; when ``--probes`` is not supplied,
it trains only ``objective_met``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.train_state import main


def force_sparse_probe_arg(argv: list[str]) -> list[str]:
    """Keep all train_state args, but force the trained probe to objective_met."""
    if "--help" in argv or "-h" in argv:
        return argv
    if "--probes" not in argv:
        return [*argv, "--probes", "objective_met"]

    idx = argv.index("--probes")
    end = idx + 1
    while end < len(argv) and not argv[end].startswith("-"):
        end += 1
    return [*argv[: idx + 1], "objective_met", *argv[end:]]


if __name__ == "__main__":
    sys.argv = force_sparse_probe_arg(sys.argv)
    main()
