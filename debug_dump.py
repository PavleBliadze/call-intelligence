"""
debug_dump.py
=============

Responsibility: write debugging artifacts for a run, under `debug/`.

  - debug/transcripts/<recording-stem>.txt : full transcript of each call.
  - debug/last_prompt.txt                  : the exact classification prompt
                                             sent for the FIRST classified
                                             call of the run.

The whole `debug/` folder is wiped and recreated at the start of each run
(see reset_debug), so artifacts never accumulate across runs and
`save_prompt_if_first` naturally writes only on the first classification.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import config

# Serialises the shared last_prompt.txt write. Under the web console several
# request threads can hit save_prompt_if_first at once; without this, concurrent
# writes to the one file can raise (a sharing violation on Windows) and bubble up
# to make that call fail and be skipped.
_prompt_lock = threading.Lock()


def reset_debug() -> None:
    """Delete and recreate the debug folder so each run starts clean."""
    if config.DEBUG_DIR.exists():
        shutil.rmtree(config.DEBUG_DIR)
    config.DEBUG_TRANSCRIPTS_DIR.mkdir(parents=True)


def save_transcript(wav_path: Path, text: str) -> None:
    """Write one recording's full transcript to debug/transcripts/<stem>.txt."""
    dest = config.DEBUG_TRANSCRIPTS_DIR / f"{wav_path.stem}.txt"
    dest.write_text(text, encoding="utf-8")


def save_prompt_if_first(prompt: str) -> None:
    """Write the classification prompt once per run (first call only).

    reset_debug() removed last_prompt.txt at run start, so it is absent only
    until the first classified call writes it; later calls find it and skip.

    Best-effort and thread-safe: it's a debug nicety, so a race or write failure
    must never break the analysis request that called it.
    """
    try:
        with _prompt_lock:
            if not config.DEBUG_PROMPT_FILE.exists():
                config.DEBUG_PROMPT_FILE.write_text(prompt, encoding="utf-8")
    except OSError:
        pass
