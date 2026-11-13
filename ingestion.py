"""One persistent ingestion pipeline for web uploads and batch CLI."""

import functools
import hashlib
import io
import json
import logging
import threading
import wave
from dataclasses import asdict
from pathlib import Path

import analyzer
import classifier
import config
import embeddings
import library
import relevance_guard
import speaker_repair
import transcriber

# Serialize identical uploads without holding a database transaction over an API call.
_locks = [threading.Lock() for _ in range(32)]


@functools.cache
def analysis_version():
    """Fingerprint of everything that shapes an analysis (code, rules, models, settings).

    Computed once per process: code and rules are loaded at startup, so the
    server must be restarted after editing them anyway.
    """
    fingerprint = hashlib.sha256()
    for name in (
        "analyzer.py",
        "transcriber.py",
        "speaker_repair.py",
        "relevance_guard.py",
        "rules_loader.py",
        "classifier.py",
        "openai_json.py",
    ):
        fingerprint.update((config.PROJECT_ROOT / name).read_bytes())
    fingerprint.update((config.RULES_DIR / f"{config.RULESET}.json").read_bytes())
    fingerprint.update(
        json.dumps(
            [
                config.AUDIO_MODEL,
                config.TEXT_MODEL,
                config.REPORT_LANGUAGE,
                config.SPEAKER_REPAIR_ENABLED,
                config.RELEVANCE_GATE_ENABLED,
                config.TRANSCRIBE_CHUNK_SECONDS,
            ]
        ).encode()
    )
    return fingerprint.hexdigest()


def validate_wav(audio):
    """Check size, PCM encoding and completeness; return the duration in seconds."""
    if len(audio) > config.MAX_UPLOAD_BYTES:
        raise ValueError("Recording exceeds the 100 MB upload limit")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getnframes() == 0:
                raise ValueError("Upload a non-empty PCM WAV recording")
            duration = wav.getnframes() / wav.getframerate()
            expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
            if len(wav.readframes(wav.getnframes())) != expected:
                raise ValueError("The WAV recording is truncated")
            return duration
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise ValueError("Upload a valid PCM WAV recording") from exc


def index_record(record, force=False):
    """Embed a record's summary unless a current vector exists; failures stay pending."""
    if record["status"] == "skipped":
        return record
    if (
        not force
        and record["index_state"] == "ready"
        and record.get("embedding_model") == config.EMBEDDING_MODEL
        and record.get("embedding_dimensions") == config.EMBEDDING_DIMENSIONS
        and record.get("embedding_text_hash") == library.summary_hash(record["summary"])
    ):
        return record
    try:
        library.save_embedding(record["id"], record["summary"], embeddings.embed(record["summary"]))
    except Exception as exc:
        library.index_failed(
            record["id"], f"Indexing failed ({type(exc).__name__}); retry indexing."
        )
    return library.get(record["id"])


def ingest(filename, audio, force=False):
    """Analyze one WAV (or reuse a current analysis), persist it, and index it."""
    duration = validate_wav(audio)
    recording_id = hashlib.sha256(audio).hexdigest()
    library.initialize()
    with _locks[int(recording_id[:2], 16) % len(_locks)]:
        version = analysis_version()
        cached = library.get(recording_id)
        path = library.audio_path(recording_id)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(audio)
            temporary.replace(path)
        if (
            cached
            and not force
            and not cached["analysis_error"]
            and cached["analysis_version"] == version
        ):
            return dict(index_record(cached), cached=True)
        name = Path(filename.replace("\\", "/")).name
        name = "".join(c for c in name if c.isprintable())[:180] or "recording.wav"
        out = {
            "id": recording_id,
            "file": name,
            "duration": duration,
            "status": "skipped",
            "triggered_rules": [],
            "summary": "",
            "transcript": "",
            "reason": "",
            "analysis_version": version,
            "analysis_error": False,
        }
        try:
            result = transcriber.transcribe(path)
            out.update(transcript=result.text, language=result.language)
            if not result.text.strip():
                out["reason"] = "No intelligible speech or supported audio event."
            else:
                if config.SPEAKER_REPAIR_ENABLED:
                    out["transcript"] = speaker_repair.repair(result.text).text
                relevance = (
                    relevance_guard.check(out["transcript"])
                    if config.RELEVANCE_GATE_ENABLED
                    else None
                )
                if relevance is not None and not relevance.is_relevant:
                    out["reason"] = relevance.reason or "No meaningful conversation."
                else:
                    verdict = analyzer.analyze(out["transcript"])
                    out.update(
                        status=classifier.classify(verdict).value,
                        summary=verdict.summary,
                        reason=verdict.reason,
                        triggered_rules=[asdict(r) for r in verdict.triggered_rules],
                    )
        except Exception as exc:
            logging.getLogger(__name__).exception("Analysis failed for %s", out["file"])
            failure = f"Analysis failed ({type(exc).__name__}); re-upload to retry."
            if cached and not cached["analysis_error"]:
                # A failed retry must not replace the last successful analysis.
                return dict(index_record(cached), cached=True, retry_error=failure)
            out.update(status="skipped", analysis_error=True, reason=failure)
        library.save(out)
        return dict(index_record(library.get(recording_id)), cached=False)
