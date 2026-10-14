"""Deletes HPO sweep checkpoints that have been superseded by a later one.

Checkpoint files are named `budget_{budget_id}_config_{config_id}_fold_{fold}_
{best,final}.pth`. When a config gets promoted to a higher fidelity rung, it
resumes from and then overwrites its own progress with a NEW, higher-
budget_id checkpoint for the same (config_id, fold) - the sweeper's own
resume logic (`_get_load_and_save_path` in hypersweeper) only ever looks at
the most recently seen budget_id for a config, so once a higher-budget_id
checkpoint exists for a given (config_id, fold), any lower-budget_id ones for
that same pair are provably dead: no future job will ever read them again.

This is safe to run at any time, including while a sweep is actively
running, since it never touches a config's current (highest) checkpoint.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from autonnunet.utils.paths import AUTONNUNET_OUTPUT

_PATTERN = re.compile(r"budget_(\d+)_config_(\d+)_fold_(\d+)_(best|final)\.pth")


def find_superseded_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Returns the checkpoint files in `checkpoint_dir` that are safe to delete."""
    groups: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    for path in checkpoint_dir.glob("*.pth"):
        match = _PATTERN.match(path.name)
        if match is None:
            continue
        budget_id, config_id, fold = int(match.group(1)), int(match.group(2)), int(match.group(3))
        groups[(config_id, fold)].append((budget_id, path))

    superseded = []
    for files in groups.values():
        max_budget_id = max(budget_id for budget_id, _ in files)
        superseded.extend(path for budget_id, path in files if budget_id < max_budget_id)
    return superseded


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="hpo_nas", choices=["hpo", "hpo_nas", "hpo_hnas"])
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--hpo_seed", type=int, default=0)
    argparser.add_argument("--dry_run", action="store_true", help="List what would be deleted, without deleting.")
    args = argparser.parse_args()

    sweep_dir = AUTONNUNET_OUTPUT / args.approach / args.dataset / args.configuration / str(args.hpo_seed)
    checkpoint_dir = sweep_dir / "checkpoints"

    superseded = find_superseded_checkpoints(checkpoint_dir)
    total_bytes = sum(p.stat().st_size for p in superseded)

    print(f"{len(superseded)} superseded checkpoint file(s), {total_bytes / 1e9:.1f} GB")
    if args.dry_run:
        for p in superseded:
            print(f"  would delete: {p.name}")
        return

    for p in superseded:
        p.unlink()
    print(f"Deleted {len(superseded)} file(s), freed {total_bytes / 1e9:.1f} GB.")


if __name__ == "__main__":
    main()
