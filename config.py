"""
config.py
=========

Single source of configuration for the call threat-analysis pipeline.

Responsibility: load the `.env` file and expose every setting (API key,
model names, folder paths, toggles) as plain module constants. No other
module should read `os.environ` or hard-code a path.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env (created manually by the user; see README).
load_dotenv(PROJECT_ROOT / ".env")

# --- OpenAI ---------------------------------------------------------------
# NOTE: raw call AUDIO is sent to OpenAI (transcription), in addition to the
# transcript text (analysis).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Pinned in code, not read from the environment, so a run is reproducible.
# Transcription needs a model that HEARS the audio (tone, shouting, background
# sounds - the audio rules), not just one that converts speech to words.
AUDIO_MODEL = "gpt-audio-1.5"
# Every text stage (speaker_repair, relevance_guard, analyzer) uses this one.
TEXT_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Persistent library; deliberately outside the disposable batch result folders.
DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DB = DATA_DIR / "library.sqlite3"
LIBRARY_AUDIO_DIR = DATA_DIR / "audio"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# --- Rule set -------------------------------------------------------------
# Which rule set to apply. Selects rules/<ruleset>.json. The workflow is
# identical for every rule set; only the rules/system prompt differ.
RULESET = os.getenv("RULESET", "threat_detection").strip()

# --- Report language ------------------------------------------------------
# Calls may be in ANY language; the transcript always stays in the language
# actually spoken. The model's explanations (violation reasons, gate reasons)
# are written in this language, so the reviewer can read every report.
REPORT_LANGUAGE = os.getenv("REPORT_LANGUAGE", "English").strip()

# --- Transcription chunking (fallback for long recordings) ----------------
# A recording is first transcribed whole (one OpenAI request). Only if that
# fails (e.g. token-limit truncation) or the file is too large to send inline
# does the transcriber fall back to splitting the audio into chunks of this
# many seconds and transcribing each separately. 300s (5 min) keeps each chunk
# well under the inline size cap and each generation bounded.
TRANSCRIBE_CHUNK_SECONDS = int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "300"))

# --- Relevance gating -----------------------------------------------------
# Kill switch for the relevance gate (relevance_guard.py), which routes calls
# with no real conversation in them (silence, voicemail, dropped line) as
# skipped instead of analysing them. Set to false to disable the gate (every
# non-empty transcript goes straight to analysis) without a code change.
RELEVANCE_GATE_ENABLED = os.getenv("RELEVANCE_GATE_ENABLED", "true").strip().lower() != "false"

# --- Speaker-label repair -------------------------------------------------
# Kill switch for the speaker-label repair pass (speaker_repair.py), which asks
# the text model to re-check who said each turn before the transcript is judged.
# Set to false to skip it (the transcriber's own labels are used as-is)
# without a code change.
SPEAKER_REPAIR_ENABLED = os.getenv("SPEAKER_REPAIR_ENABLED", "true").strip().lower() != "false"

# --- Folders --------------------------------------------------------------
RULES_DIR = PROJECT_ROOT / "rules"
CALLS_DIR = PROJECT_ROOT / "calls"
SAFE_DIR = PROJECT_ROOT / "safe"
SUSPICIOUS_DIR = PROJECT_ROOT / "suspicious"

# --- Debug artifacts (cleared and recreated each run) ---------------------
DEBUG_DIR = PROJECT_ROOT / "debug"
DEBUG_TRANSCRIPTS_DIR = DEBUG_DIR / "transcripts"
DEBUG_PROMPT_FILE = DEBUG_DIR / "last_prompt.txt"

# Run log (no full transcripts written here; see logger_setup.py).
LOG_FILE = PROJECT_ROOT / "run.log"

# --- Web console (web/app.py) ---------------------------------------------
# Served with waitress so several people can use it at once (concurrency comes
# from the thread pool below, not the bind address). WEB_HOST default 127.0.0.1
# runs on localhost only; set WEB_HOST=0.0.0.0 for direct LAN access.
# WEB_THREADS is the concurrency cap: at most this many analyses (hence OpenAI
# calls) run at once; raise it (and mind the OpenAI rate limit) for more users.
# Default port 5050 avoids the common 8000/8080.
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1").strip()
WEB_PORT = int(os.getenv("WEB_PORT", "5050"))
WEB_THREADS = int(os.getenv("WEB_THREADS", "6"))


def validate(*, require_calls: bool = True) -> None:
    """Fail fast with a clear message if required config is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file in the project "
            "root containing:  OPENAI_API_KEY=your_key_here"
        )
    if require_calls and not CALLS_DIR.is_dir():
        raise RuntimeError(
            f"Calls folder not found: {CALLS_DIR}\n"
            "Create it and add .wav recordings before running."
        )
    rules_file = RULES_DIR / f"{RULESET}.json"
    if not rules_file.is_file():
        raise RuntimeError(
            f"Rule set not found for RULESET='{RULESET}': "
            f"{rules_file}\nExpected a JSON rule file at that path."
        )
