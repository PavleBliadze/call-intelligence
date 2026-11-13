"""Live retrieval ablation: python demo/evaluate_search.py --repeat 3.

Uses the pre-indexed library; never regenerates or retranscribes audio.
Writes individual ranks, failures, Recall@3, MRR and end-to-end latency.
Follow-up cases use fixed labeled prior contexts, not another run's predictions.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
import library
import search


def aggregate(rows):
    """Per-mode Recall@3, MRR, latency, no-match accuracy and error counts."""
    output = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        subset = [r for r in rows if r["mode"] == mode]
        positive = [r for r in subset if r["expected"]]
        negative = [r for r in subset if not r["expected"]]
        output[mode] = {
            "queries": len(subset),
            "recall_at_3": statistics.mean(r["recall_at_3"] for r in positive),
            "mrr": statistics.mean(r["reciprocal_rank"] for r in positive),
            "mean_latency_ms": round(statistics.mean(r["elapsed_ms"] for r in subset)),
            "median_latency_ms": round(statistics.median(r["elapsed_ms"] for r in subset)),
            "no_match_accuracy": statistics.mean(
                not r["returned"] and not r.get("error") for r in negative
            )
            if negative
            else None,
            "errors": sum(bool(r.get("error")) for r in subset),
            "degraded_requests": sum(bool(r.get("degraded")) for r in subset),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    config.validate()
    library.initialize()
    cases = json.loads((ROOT / "demo/search_cases.json").read_text(encoding="utf-8"))
    records = {r["file"]: r for r in library.searchable_records()}
    corpus = [
        {
            "id": r["id"],
            "file": r["file"],
            "version": r["analysis_version"],
            "summary_hash": library.summary_hash(r["summary"]),
            "transcript_hash": hashlib.sha256(r["transcript"].encode()).hexdigest(),
        }
        for r in sorted(records.values(), key=lambda r: r["id"])
    ]
    required = {name for c in cases for name in c["relevant"] + c.get("references", [])}
    if required - records.keys() or any(records[n]["index_state"] != "ready" for n in required):
        parser.error("Run python main.py first; evaluation needs the indexed demo recordings.")
    rows = []
    config.DEBUG_DIR.mkdir(exist_ok=True)
    report_path = config.DEBUG_DIR / "evaluation_search.json"
    modes = [("keyword", False), ("vector", False), ("hybrid", False), ("hybrid", True)]
    for run in range(1, args.repeat + 1):
        for case in cases:
            # Rotate mode order to reduce a fixed warmup/order advantage.
            for mode, rerank in modes[run % 4 :] + modes[: run % 4]:
                started = time.perf_counter()
                label = mode + ("_reranked" if rerank else "")
                row = {
                    "run": run,
                    "case": case["name"],
                    "mode": label,
                    "query": case["query"],
                    "expected": case["relevant"],
                    "returned": [],
                }
                try:
                    response = search.search(
                        case["query"],
                        history=case.get("history", []),
                        reference_ids=[records[n]["id"] for n in case.get("references", [])],
                        mode=mode,
                        use_reranker=rerank,
                    )
                    row.update(
                        returned=[r["file"] for r in response["results"]],
                        diagnostics=response["diagnostics"],
                        notices=response["notices"],
                    )
                    row["degraded"] = any("unavailable" in n.lower() for n in response["notices"])
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
                expected = set(case["relevant"])
                row["recall_at_3"] = (
                    len(expected & set(row["returned"][:3])) / len(expected) if expected else None
                )
                row["reciprocal_rank"] = (
                    next(
                        (
                            1 / rank
                            for rank, name in enumerate(row["returned"], 1)
                            if name in expected
                        ),
                        0,
                    )
                    if expected
                    else None
                )
                rows.append(row)
                print(
                    f"run={run} {case['name']} {label}: {row['returned']} ({row['elapsed_ms']} ms)",
                    flush=True,
                )
                report_path.write_text(
                    json.dumps(
                        {
                            "created_at": datetime.now(UTC).isoformat(),
                            "corpus": corpus,
                            "models": {
                                "text": config.TEXT_MODEL,
                                "embedding": config.EMBEDDING_MODEL,
                            },
                            "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
                            "notes": "MRR is measured over at most five returned recordings; positive queries only for Recall/MRR. No-match accuracy is separate. All modes include query planning.",
                            "metrics": aggregate(rows),
                            "results": rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    print(json.dumps(aggregate(rows), indent=2))
    print(f"Report: {report_path}")
    raise SystemExit(1 if any(r.get("error") for r in rows) else 0)


if __name__ == "__main__":
    main()
