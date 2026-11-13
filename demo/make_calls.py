"""
demo/make_calls.py
==================

Generate the demo call recordings from demo/scripts.json with OpenAI TTS.

Each turn is spoken by its speaker's voice (with an optional per-turn tone,
e.g. "screaming in terror"), the turns are joined with a short pause, and the
result is written as a 24 kHz 16-bit mono WAV into calls/ - ready for
`python main.py` or for drag-and-drop in the web console.

Run:
    python demo/make_calls.py            # generate the missing recordings
    python demo/make_calls.py --force    # regenerate all of them
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

# Make the project-root modules importable when run as `python demo/make_calls.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI

import config

_TTS_MODEL = "gpt-4o-mini-tts"
# response_format="pcm" is raw 24 kHz, 16-bit little-endian, mono.
_SAMPLE_RATE = 24000
_PAUSE = b"\x00\x00" * int(_SAMPLE_RATE * 0.5)  # 0.5 s of silence between turns

SCRIPTS_FILE = Path(__file__).resolve().parent / "scripts.json"


def _speak(client: OpenAI, voice: str, text: str, tone: str) -> bytes:
    """One turn -> raw PCM."""
    response = client.audio.speech.create(
        model=_TTS_MODEL,
        voice=voice,
        input=text,
        instructions=tone,
        response_format="pcm",
    )
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing selected WAVs")
    parser.add_argument("--only", nargs="+", help="Generate only these exact WAV filenames")
    args = parser.parse_args()
    calls = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))["calls"]
    if args.only:
        unknown = set(args.only) - {call["file"] for call in calls}
        if unknown:
            parser.error(f"Unknown recordings: {', '.join(sorted(unknown))}")
        calls = [call for call in calls if call["file"] in args.only]
    config.CALLS_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    for call in calls:
        out_path = config.CALLS_DIR / call["file"]
        if out_path.exists() and not args.force:
            print(f"skip  {out_path.name} (exists; --force to regenerate)")
            continue
        speakers = call["speakers"]
        pcm = b""
        for turn in call["turns"]:
            speaker = speakers[turn["speaker"]]
            tone = turn.get("tone", speaker["tone"])
            pcm += _speak(client, speaker["voice"], turn["text"], tone) + _PAUSE
        # Publish only a complete WAV, preserving the previous file on failure.
        temporary = out_path.with_suffix(".wav.tmp")
        with wave.open(str(temporary), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(_SAMPLE_RATE)
            out.writeframes(pcm)
        temporary.replace(out_path)
        seconds = len(pcm) / 2 / _SAMPLE_RATE
        print(f"wrote {out_path.name} ({seconds:.0f}s, expected: {call['expected']})")


if __name__ == "__main__":
    main()
