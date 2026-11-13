"""
web/app.py
==========

Web console for the call threat-analysis pipeline.

Serves a single-page UI (web/index.html) and an /api/analyze endpoint that runs
ONE uploaded recording through the existing pipeline:

    transcribe  ->  speaker-label repair  ->  relevance guard
      ->  analyze  ->  classify

and returns a JSON result the UI renders. Processing one file per request keeps
the server simple and lets the UI show real per-file progress (the pipeline is
sequential by design anyway).

The shared ingestion service persists audio and analysis in data/. Library,
hybrid search, conversational answers and audio playback use this same catalog.

Run:
    python web/app.py        # serves on WEB_HOST:WEB_PORT via waitress

Served with waitress (a production WSGI server) so several people can use it at
once. Requests share the persistent catalog, deduplicate identical uploads,
and WEB_THREADS caps how many analyses run in parallel. Configure via
WEB_HOST / WEB_PORT / WEB_THREADS.

SECURITY: this sends call AUDIO and transcript text to OpenAI's cloud. The
console is UNAUTHENTICATED: anyone who can reach the port can use it and see
transcripts. Only share the link with people you trust.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

# Make the project-root pipeline modules importable when run as `python web/app.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from waitress import serve

import config
import ingestion
import library
import rules_loader
import search

WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES + 1024 * 1024


def _ensure_ready() -> None:
    """Fail fast with a clear message if config is missing (uploads need no calls/ folder)."""
    config.validate(require_calls=False)
    # analyzer.analyze -> debug_dump.save_prompt_if_first writes the first prompt
    # under debug/; make sure the folder exists so that write succeeds.
    config.DEBUG_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def _analyze_one(filename: str, audio_bytes: bytes, *, force=False) -> dict:
    return ingestion.ingest(filename, audio_bytes, force=force)


@app.before_request
def prepare_library():
    if request.path.startswith("/api/"):
        library.initialize()


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify(error="Recording exceeds the 100 MB upload limit"), 413


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/static/<path:filename>")
def static_file(filename):
    return send_from_directory(WEB_DIR / "static", filename)


@app.post("/api/analyze")
def api_analyze():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify(error="No file uploaded"), 400
    try:
        return jsonify(
            _analyze_one(uploaded.filename or "recording.wav", uploaded.read(), force=True)
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        return jsonify(error="Unable to save the recording. Please retry."), 500


def _filters(status, categories):
    if status not in (None, "", "safe", "suspicious", "skipped"):
        raise ValueError("Invalid status filter")
    known = {r.id for r in rules_loader.get_active().rules}
    if any(type(c) is not int or c not in known for c in categories):
        raise ValueError("Invalid category filter")
    return status or None, list(dict.fromkeys(categories))


@app.get("/api/recordings")
def api_recordings():
    try:
        status, categories = _filters(
            request.args.get("status"), [int(c) for c in request.args.getlist("category")]
        )
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 24))
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("Invalid pagination")
        return jsonify(library.list_recordings(status, categories, page, page_size))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/recordings/<recording_id>")
def api_recording(recording_id):
    record = library.get(recording_id)
    if record is None:
        abort(404)
    return jsonify(record)


@app.get("/api/recordings/<recording_id>/audio")
def api_audio(recording_id):
    record = library.get(recording_id)
    if record is None or not library.audio_path(recording_id).is_file():
        abort(404)
    return send_file(
        library.audio_path(recording_id),
        mimetype="audio/wav",
        conditional=True,
        as_attachment=request.args.get("download") == "1",
        download_name=record["file"],
    )


@app.get("/api/categories")
def api_categories():
    return jsonify(library.category_stats())


@app.post("/api/search")
def api_search():
    body = request.get_json(silent=True)
    try:
        if not isinstance(body, dict):
            raise ValueError("Send a JSON search request")
        query = body.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Enter a question of 1 to 2,000 characters")
        categories = body.get("categories", [])
        if not isinstance(categories, list):
            raise ValueError("Categories must be a list")
        status, categories = _filters(body.get("status"), categories)
        history = body.get("history", [])
        references = body.get("reference_ids", [])
        if not isinstance(history, list) or not isinstance(references, list):
            raise ValueError("Invalid conversation context")
        if any(not library.valid_id(r) for r in references):
            raise ValueError("Invalid recording reference")
        clean_history = []
        for message in history[-6:]:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Invalid conversation message")
            clean_history.append({"role": message["role"], "content": message["content"][:2000]})
        use_reranker = body.get("rerank", True)
        if not isinstance(use_reranker, bool):
            raise ValueError("rerank must be a boolean")
        return jsonify(
            search.search(
                query.strip(),
                clean_history,
                references[:5],
                status=status,
                categories=categories,
                use_reranker=use_reranker,
            )
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        return jsonify(
            error="Search could not be completed. Your recordings are still in the library."
        ), 500


def _port_in_use(host: str, port: int) -> bool:
    """Whether something is already serving on this port.

    On Windows a second process CAN bind an address another process is already
    listening on (waitress sets SO_REUSEADDR, which on Windows permits it
    instead of failing): both consoles then report "serving" and incoming
    requests land on one or the other unpredictably. Connecting first turns
    that silent hazard into a loud refusal to start.
    """
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe, port)) == 0


def _lan_ip() -> str:
    """Best-effort primary LAN IPv4 for this host (for the shareable URL).

    Opens a UDP socket toward a public address to learn which local interface
    the OS would route out of; no packet is actually sent. Falls back to
    127.0.0.1 if that can't be resolved (e.g. no network).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    _ensure_ready()
    if _port_in_use(config.WEB_HOST, config.WEB_PORT):
        print(
            f"Port {config.WEB_PORT} is already in use - another console is "
            f"running.\n"
            f"Stop it first (Ctrl+C in its window), or set WEB_PORT to a free "
            f"port in .env.",
            file=sys.stderr,
        )
        sys.exit(1)
    # channel_timeout is generous: one request holds the connection for the whole
    # transcribe+analyze (30-120s+, longer when a long call falls back to chunked
    # transcription), so it must not be cut off mid-request.
    # When bound to all interfaces (0.0.0.0/""), print the LAN IP others should
    # open plus the host-only localhost URL; otherwise just show the bound host.
    if config.WEB_HOST in ("0.0.0.0", ""):
        print(
            f"Call Threat Analysis console -> http://{_lan_ip()}:{config.WEB_PORT}"
            f"  (share this on your network)"
        )
        print(f"  local: http://127.0.0.1:{config.WEB_PORT}")
        print(f"  ({config.WEB_THREADS} threads, Ctrl+C to stop)")
    else:
        print(
            f"Call Threat Analysis console -> http://{config.WEB_HOST}:{config.WEB_PORT}"
            f"  ({config.WEB_THREADS} threads, Ctrl+C to stop)"
        )
    serve(
        app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        threads=config.WEB_THREADS,
        channel_timeout=600,
    )
