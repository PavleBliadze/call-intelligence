"""Live regression evaluation (uses the configured OpenAI API key).

python demo/evaluate.py --repeat 3  # scripts + boundary cases, text analysis
python demo/evaluate.py --audio     # actual WAVs through the web pipeline
Reports are written under debug/ without clearing existing recordings/results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analyzer
import classifier
import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    config.validate()
    config.DEBUG_DIR.mkdir(exist_ok=True)
    calls = json.loads((ROOT / "demo/scripts.json").read_text(encoding="utf-8"))["calls"]
    if args.audio:
        from web.app import _analyze_one

        cases = calls + [
            {"file": "call_08.wav", "expected_route": "skipped", "expected_rule_ids": []}
        ]
    else:
        cases = [
            dict(
                call,
                transcript="\n".join(
                    f"{turn['speaker']}: {turn['text']}" for turn in call["turns"]
                ),
            )
            for call in calls
        ]
        cases += json.loads((ROOT / "demo/boundary_cases.json").read_text(encoding="utf-8"))
    results = []
    for run in range(1, args.repeat + 1):
        for case in cases:
            name = case.get("file", case.get("name"))
            try:
                if args.audio:
                    result = _analyze_one(name, (config.CALLS_DIR / name).read_bytes(), force=True)
                else:
                    verdict = analyzer.analyze(case["transcript"])
                    result = {
                        "status": classifier.classify(verdict).value,
                        "triggered_rules": [asdict(r) for r in verdict.triggered_rules],
                        "reason": verdict.reason,
                        "transcript": case["transcript"],
                    }
                actual = {r["id"] for r in result["triggered_rules"]}
                expected = set(case["expected_rule_ids"])
                optional = set(case.get("optional_audio_rule_ids", [])) if args.audio else set()
                passed = (
                    not result.get("analysis_error")
                    and result["status"] == case["expected_route"]
                    and expected <= actual
                    and actual <= expected | optional
                )
                detail = f"{result['status']} flags={sorted(actual)}"
            except Exception as exc:
                result, passed, detail = {"error": str(exc)}, False, f"ERROR: {type(exc).__name__}"
            results.append(
                {
                    "case": name,
                    "run": run,
                    "passed": passed,
                    "expected_route": case["expected_route"],
                    "expected_rule_ids": case["expected_rule_ids"],
                    "result": result,
                }
            )
            print(f"{'PASS' if passed else 'FAIL'} run={run} {name}: {detail}", flush=True)
            mode = "audio" if args.audio else "text"
            (config.DEBUG_DIR / f"evaluation_{mode}.json").write_text(
                json.dumps(
                    {
                        "models": {"audio": config.AUDIO_MODEL, "text": config.TEXT_MODEL},
                        "analysis_reasoning_effort": analyzer._ANALYSIS_REASONING_EFFORT,
                        "results": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    passed = sum(r["passed"] for r in results)
    print(f"{passed}/{len(results)} passed; report: debug/evaluation_{mode}.json")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
