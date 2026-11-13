"""
audio_loader.py
===============

Responsibility: find the call recordings to process.

Scans the configured `calls/` folder for `.wav` files and yields their
paths. It does not open, decode, or transcribe audio - that is the
transcriber's job.
"""

from __future__ import annotations

from pathlib import Path

import config


def find_wav_files() -> list[Path]:
    """Return a sorted list of .wav files in the calls folder.

    Sorted so runs are deterministic and the log is easy to follow.
    Matching is case-insensitive (.wav / .WAV).
    """
    files = [p for p in config.CALLS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    return sorted(files, key=lambda p: p.name.lower())
