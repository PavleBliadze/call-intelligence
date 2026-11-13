"""
summary_writer.py
=================

Responsibility: write a `<recording-name>.summary.txt` beside each copied
recording, whatever its route:

  - suspicious : factual summary and the flags raised, with explanations
  - safe       : factual summary and the transcript

Skipped (not analysed) recordings are not filed, so they get no summary.
"""

from __future__ import annotations

from pathlib import Path

from analyzer import Verdict


def _summary_path(copied_recording: Path) -> Path:
    return copied_recording.with_suffix(copied_recording.suffix + ".summary.txt")


def write_suspicious_summary(copied_recording: Path, verdict: Verdict) -> Path:
    """Write `<name>.summary.txt` beside the copied suspicious recording.

    Lists every flag (triggered rule) with the reason it was raised. `copied_recording` is the path of the file already placed in
    /suspicious. Returns the summary file path.
    """
    summary_path = _summary_path(copied_recording)

    flags = "\n".join(f"- {rule.name}: {rule.reason}".rstrip() for rule in verdict.triggered_rules)

    content = (
        f"Recording: {copied_recording.name}\n"
        f"Classification: SUSPICIOUS\n"
        f"\nSummary:\n{verdict.summary}\n"
        f"\nFlags:\n"
        f"{flags}\n"
    )
    summary_path.write_text(content, encoding="utf-8")
    return summary_path


def write_safe_summary(copied_recording: Path, transcript: str, summary: str = "") -> Path:
    """Write `<name>.summary.txt` beside a recording copied to /safe.

    No flags; include the factual summary and transcript.
    Returns the summary file path.
    """
    summary_path = _summary_path(copied_recording)
    content = (
        f"Recording: {copied_recording.name}\n"
        f"Classification: SAFE\n"
        f"\nSummary:\n{summary}\n"
        f"\nTranscript:\n{transcript.strip()}\n"
    )
    summary_path.write_text(content, encoding="utf-8")
    return summary_path
