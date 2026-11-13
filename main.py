"""
main.py
=======

Orchestrator. Runs the full pipeline sequentially:

    scan calls/  ->  transcribe (OpenAI audio model, any language)
      ->  speaker-label repair  ->  relevance guard
      ->  analyze transcript with OpenAI  ->  classify  ->  copy file
      ->  write summary  ->  log

Sequential by design: simplest to audit and it respects OpenAI rate
limits. A recording that is not worth analysing (no speech, no real
conversation, unjudgeable) - or that fails - is SKIPPED: logged with the
reason and not exported into batch result folders, without stopping the run.
Every valid WAV is retained in the independent data/ library.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import analyzer
import audio_loader
import config
import debug_dump
import file_organizer
import ingestion
import library
import summary_writer
from classifier import Route
from logger_setup import get_logger

log = get_logger()


def _process_one(wav_path, force=False) -> Route:
    """Ingest one recording and export safe/suspicious results with a summary."""
    result = ingestion.ingest(wav_path.name, wav_path.read_bytes(), force=force)
    debug_dump.save_transcript(wav_path, result["transcript"])
    route = Route(result["status"])
    if route is not Route.SKIPPED:
        destination = file_organizer.copy_to(route, wav_path)
        if route is Route.SUSPICIOUS:
            verdict = analyzer.Verdict(
                "suspicious",
                result["reason"],
                [analyzer.TriggeredRule(**rule) for rule in result["triggered_rules"]],
                result["summary"],
            )
            summary_writer.write_suspicious_summary(destination, verdict)
        else:
            summary_writer.write_safe_summary(destination, result["transcript"], result["summary"])
    log.info(
        "%s -> %s (index=%s, cached=%s) %s",
        wav_path.name,
        route.value.upper(),
        result["index_state"],
        result["cached"],
        result["reason"][:100],
    )
    return route


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and index the persistent recording library"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reprocess", action="store_true", help="Force full audio analysis")
    group.add_argument(
        "--reindex", action="store_true", help="Rebuild embeddings from stored summaries"
    )
    args = parser.parse_args()
    config.validate(require_calls=not args.reindex)
    library.initialize()
    if args.reindex:
        pending = 0
        for record in library.searchable_records():
            result = ingestion.index_record(record, force=True)
            pending += result["index_state"] != "ready"
            print(f"{result['file']}: {result['index_state']}")
        if pending:
            raise RuntimeError(f"{pending} recording(s) still need indexing; retry --reindex")
        return
    # Start each run with clean output + debug folders (old results removed).
    file_organizer.reset_output_folders()
    debug_dump.reset_debug()
    files = audio_loader.find_wav_files()

    if not files:
        log.info("No .wav files found in %s - nothing to do.", config.CALLS_DIR)
        return

    log.info(
        "Found %d recording(s). Models: %s / %s | Rule set: %s",
        len(files),
        config.AUDIO_MODEL,
        config.TEXT_MODEL,
        config.RULESET,
    )

    counts: Counter[str] = Counter()
    for wav_path in files:
        try:
            route = _process_one(wav_path, force=args.reprocess)
            counts[route.value] += 1
        except Exception as exc:  # keep the batch alive
            log.error("%s -> SKIPPED (error: %s)", wav_path.name, exc)
            counts[Route.SKIPPED.value] += 1

    log.info(
        "Done. safe=%d  suspicious=%d  skipped=%d  (total=%d)",
        counts.get("safe", 0),
        counts.get("suspicious", 0),
        counts.get("skipped", 0),
        len(files),
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
