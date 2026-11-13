"""
classifier.py
=============

Responsibility: turn an analyzer Verdict into a final routing decision.

Pure mapping, no I/O. Every judged call is safe or suspicious. A call that
cannot be judged ("uncertain": crucial words missing) is SKIPPED - not
analysed, never filed - rather than guessed at either way.
"""

from __future__ import annotations

from enum import Enum

from analyzer import Verdict


class Route(str, Enum):  # noqa: UP042 - StrEnum would change str(route)
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    # Not analysed (no conversation, no speech, unjudgeable, or an error):
    # logged / shown with a reason, but never copied into a result folder.
    SKIPPED = "skipped"


def classify(verdict: Verdict) -> Route:
    """Map a model verdict to a route."""
    if verdict.classification == "safe":
        return Route.SAFE
    if verdict.classification == "suspicious":
        return Route.SUSPICIOUS
    # "uncertain" or anything unexpected -> not analysed.
    return Route.SKIPPED
