"""
openai_json.py
==============

Responsibility: one robust "call OpenAI, get a JSON object back" function,
shared by transcriber.py (audio -> transcript), speaker_repair.py,
relevance_guard.py and analyzer.py (transcript -> verdict).

Why it exists: a bare `json.loads(message.content)` makes regular calls fail
(and be skipped) with opaque errors whenever a response is refused, cut off
mid-generation or empty. This module centralizes:

  - Structured Outputs (a pydantic schema) for the text stages, so their JSON is
    always well-formed. The audio model does not support Structured Outputs, so
    for it the prompt asks for JSON and the reply is parsed leniently (code
    fences and raw newlines inside the transcript string are tolerated).
  - explicit max_completion_tokens, so long transcripts are not truncated.
  - response validation: empty/refused/truncated/unparseable responses raise a
    RuntimeError that names the finish_reason instead of a bare JSON error.
  - retries with backoff for transient failures (empty/truncated output, 5xx,
    connection errors, and 429 rate limits - important when several web users
    run at once).
  - a per-request deadline, so one hung request fails and gets retried instead
    of stalling the whole batch.
"""

from __future__ import annotations

import json
import time

import openai
from openai import OpenAI
from pydantic import BaseModel

import config

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2

# Per-attempt deadline. A degenerate transcription request can hang for many
# minutes before returning; fail it and retry.
_TIMEOUT_SECONDS = 180

# Output caps. Text models count reasoning tokens against this too, and speaker
# repair echoes the whole transcript back, so the text cap is generous. The
# audio model's maximum output is 16,384 tokens; a transcript longer than that
# comes back truncated and the transcriber falls back to chunks.
_TEXT_MAX_OUTPUT_TOKENS = 65536
_AUDIO_MAX_OUTPUT_TOKENS = 16384

# Default for repair and relevance. The analyzer requests medium reasoning
# explicitly because repeated evaluations exposed missed category exclusions
# at low effort (for example, jokes and disclosure-only threats).
_TEXT_REASONING_EFFORT = "low"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create the OpenAI client once and reuse it.

    The SDK's own retries are disabled so the loop below is the only retry
    policy (otherwise each attempt could silently retry several times more).
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _client


def _parse_lenient(raw: str) -> dict:
    """Parse a JSON object from free text (audio model, no Structured Outputs).

    Takes the outermost {...} so a ```json fence or a stray sentence around it
    does not matter; strict=False accepts literal newlines inside strings, which
    a multi-line transcript value often contains. Slicing to the braces also
    unwraps the object if the model wrapped it in a JSON array.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return json.loads(raw[start : end + 1], strict=False)


def generate_json(
    model: str,
    content,
    schema: type[BaseModel] | None = None,
    *,
    reasoning_effort: str = _TEXT_REASONING_EFFORT,
) -> dict:
    """Call OpenAI and return its response parsed as a dict.

    `content` is the user message content: a prompt string, or a list of
    content parts (text + input_audio) for the audio model.
    `schema` is the pydantic response schema (Structured Outputs) for text
    models; pass None for the audio model, which does not support it.
    `reasoning_effort` applies only to text requests.
    Retries up to _MAX_ATTEMPTS on empty/unparseable/truncated responses,
    rate limits and server errors; after the last attempt raises RuntimeError
    with a diagnostic message (main.py skips the call and logs that reason).
    """
    client = _get_client()
    messages = [{"role": "user", "content": content}]
    if schema is None and isinstance(content, list):
        # Keep instructions separate from the audio attachment. In live checks,
        # mixed text/audio user messages intermittently elicited "no audio supplied".
        instructions = "\n".join(p["text"] for p in content if p.get("type") == "text")
        attachments = [p for p in content if p.get("type") != "text"]
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": attachments},
        ]
    last_error = ""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            if schema is not None:
                completion = client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema,
                    max_completion_tokens=_TEXT_MAX_OUTPUT_TOKENS,
                    reasoning_effort=reasoning_effort,
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    modalities=["text"],
                    max_completion_tokens=_AUDIO_MAX_OUTPUT_TOKENS,
                )
        except (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,  # includes APITimeoutError
        ) as exc:
            # Retry 429 rate limits (likely when several users run at once),
            # transient 5xx and timeouts; other client errors (bad request,
            # auth) won't fix themselves, so they propagate immediately.
            last_error = f"OpenAI API error: {exc!r}"
        except (
            openai.LengthFinishReasonError,
            openai.ContentFilterFinishReasonError,
        ) as exc:
            # Raised by .parse() when the structured response was cut off.
            last_error = f"OpenAI response incomplete ({exc})"
        else:
            choice = completion.choices[0]
            message = choice.message
            if message.refusal:
                last_error = f"OpenAI refused the request ({message.refusal})"
            elif choice.finish_reason != "stop":
                last_error = f"OpenAI response incomplete (finish_reason={choice.finish_reason})"
            elif schema is not None:
                if message.parsed is not None:
                    return message.parsed.model_dump()
                last_error = "OpenAI returned no parsed output"
            elif not message.content:
                last_error = "OpenAI returned no text"
            else:
                try:
                    return _parse_lenient(message.content)
                except json.JSONDecodeError as exc:
                    last_error = f"OpenAI returned invalid JSON ({exc})"
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"{last_error} [after {_MAX_ATTEMPTS} attempts]")
