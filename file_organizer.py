"""
file_organizer.py
=================

Responsibility: copy a recording into its destination folder.

Files are COPIED, never moved - the originals in /calls stay untouched
(the source recordings are never modified). Destination folders are created on demand.
Name collisions are avoided by appending a numeric suffix.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import config
from classifier import Route

# Route.SKIPPED has no folder: a recording that was not analysed is not filed.
_ROUTE_DIRS: dict[Route, Path] = {
    Route.SAFE: config.SAFE_DIR,
    Route.SUSPICIOUS: config.SUSPICIOUS_DIR,
}


def reset_output_folders() -> None:
    """Delete and recreate the result folders so each run starts clean.

    Removes any recordings/summaries left from a previous run rather than
    letting them accumulate alongside the new results.
    """
    for dest_dir in _ROUTE_DIRS.values():
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)


def _unique_destination(dest_dir: Path, name: str) -> Path:
    """Return a non-colliding path inside dest_dir for `name`."""
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_to(route: Route, source: Path) -> Path:
    """Copy `source` into the folder for `route`. Returns the new path."""
    dest_dir = _ROUTE_DIRS[route]
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(dest_dir, source.name)
    shutil.copy2(source, destination)
    return destination
