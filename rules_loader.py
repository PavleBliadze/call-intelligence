"""
rules_loader.py
===============

Responsibility: load a rule set from `rules/<ruleset>.json` and expose it as
typed objects. One JSON file per rule set; the workflow is identical for every
rule set - only the system prompt and the rules differ. To add a new kind of
detection, add or edit a JSON file - no code change needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Rule:
    """One detection rule.

    `description` is the *trigger condition* - what makes the rule count as
    triggered. Any triggered rule flags the call as suspicious.

    `source` is where the rule is judged:
      - "transcript" (default): from the transcript text, by the analysis model.
      - "audio": from the audio, by the transcriber, which appends the exact
        `marker` string to the transcript when the behaviour occurs; the
        analyzer then detects that marker. (Shouting, screaming and the sounds
        of a struggle cannot be judged from text alone.)
    """

    id: int
    name: str
    description: str
    source: str = "transcript"
    marker: str = ""
    fallback: bool = False


@dataclass(frozen=True)
class RuleSet:
    """All rules of one rule set, plus its system prompt."""

    ruleset: str
    display_name: str
    system_prompt: str
    rules: list[Rule]


def load(ruleset: str) -> RuleSet:
    """Load and validate `rules/<ruleset>.json`. Fail fast on any problem."""
    path = config.RULES_DIR / f"{ruleset}.json"
    if not path.is_file():
        raise RuntimeError(
            f"Rule set not found: '{ruleset}': {path}\n"
            f"Create it (one JSON file per rule set under {config.RULES_DIR})."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    for key in ("ruleset", "system_prompt", "rules"):
        if key not in raw:
            raise RuntimeError(f"{path} is missing required key '{key}'.")
    if not raw["rules"]:
        raise RuntimeError(f"{path} defines no rules.")

    rules: list[Rule] = []
    seen_ids: set[int] = set()
    for entry in raw["rules"]:
        for key in ("id", "name", "description"):
            if key not in entry:
                raise RuntimeError(f"{path}: rule {entry.get('id', '?')} is missing key '{key}'.")
        rule_id = int(entry["id"])
        if rule_id in seen_ids:
            raise RuntimeError(f"{path}: duplicate rule id {rule_id}.")
        seen_ids.add(rule_id)
        source = str(entry.get("source", "transcript"))
        if source not in ("transcript", "audio"):
            raise RuntimeError(
                f"{path}: rule {rule_id} has invalid source '{source}' "
                f"(must be 'transcript' or 'audio')."
            )
        marker = str(entry.get("marker", "")).strip()
        fallback = entry.get("fallback", False)
        if not isinstance(fallback, bool) or (fallback and source != "transcript"):
            raise RuntimeError(f"{path}: fallback must be a boolean on a transcript rule.")
        if source == "audio" and not marker:
            raise RuntimeError(f"{path}: audio rule {rule_id} must define a non-empty 'marker'.")
        rules.append(
            Rule(
                id=rule_id,
                name=str(entry["name"]),
                description=str(entry["description"]),
                source=source,
                marker=marker,
                fallback=fallback,
            )
        )

    if sum(rule.fallback for rule in rules) > 1:
        raise RuntimeError(f"{path}: only one fallback rule is allowed.")
    return RuleSet(
        ruleset=str(raw["ruleset"]),
        display_name=str(raw.get("display_name", raw["ruleset"])),
        system_prompt=str(raw["system_prompt"]),
        rules=rules,
    )


_active: RuleSet | None = None


def get_active() -> RuleSet:
    """Load the configured rule set once and reuse it.

    Shared by analyzer.py (transcript-source rules) and transcriber.py
    (audio-source rules) so the active rule set is parsed only once per run.
    """
    global _active
    if _active is None:
        _active = load(config.RULESET)
    return _active
