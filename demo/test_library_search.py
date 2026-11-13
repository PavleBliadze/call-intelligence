"""Offline integration checks using an isolated library and mocked model calls."""

import hashlib
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

import analyzer
import config
import debug_dump
import file_organizer
import ingestion
import library
import rules_loader
import search
from transcriber import Transcription
from web.app import app


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 800)
    return buffer.getvalue()


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for key, value in {
            "DATA_DIR": root,
            "LIBRARY_DB": root / "library.sqlite3",
            "LIBRARY_AUDIO_DIR": root / "audio",
            "EMBEDDING_DIMENSIONS": 3,
        }.items():
            p = patch.object(config, key, value)
            p.start()
            self.addCleanup(p.stop)
        library.initialize()
        self.client = app.test_client()

    def save(
        self,
        name="call.wav",
        status="suspicious",
        categories=(6,),
        text="The midnight mailbox campaign.",
        vector=None,
    ):
        rid = hashlib.sha256(name.encode()).hexdigest()
        library.save(
            {
                "id": rid,
                "file": name,
                "duration": 0.1,
                "status": status,
                "summary": text,
                "transcript": "Person 1: " + text,
                "triggered_rules": [{"id": i, "name": str(i), "reason": text} for i in categories],
                "analysis_version": "test",
            }
        )
        library.audio_path(rid).write_bytes(wav_bytes())
        if vector is not None:
            library.save_embedding(rid, text, vector)
        return library.get(rid)

    def plan(self, query="mailbox", filters=None, refs=None, inferred=None, status=None):
        return {
            "query": query,
            "category_filters": filters or [],
            "inferred_categories": inferred or [],
            "status_filter": status,
            "recording_ids": refs or [],
        }

    @staticmethod
    def supported(query, candidates):
        return [
            dict(r, answer_sentence=r["summary"], evidence_quote=r["summary"])
            for r in candidates[:5]
        ]

    def test_persistence_fts_punctuation_and_skipped_exclusion(self):
        record = self.save(vector=[1, 0, 0])
        self.save("skip.wav", status="skipped", categories=())
        library.initialize()  # new connections/restart leave the catalog intact
        self.assertEqual(library.list_recordings()["total"], 2)
        self.assertEqual(library.keywords('"midnight mailbox"?! OR :*')[0][0], record["id"])
        self.assertEqual(len(library.searchable_records()), 1)
        self.assertEqual(library.category_stats()["searchable"], 1)
        self.assertEqual(len(library.vectors()), 1)
        with patch.object(config, "EMBEDDING_MODEL", "different-model"):
            self.assertEqual(library.vectors(), [])
            self.assertEqual(library.category_stats()["searchable"], 0)

    def test_batch_cleanup_never_removes_persistent_library(self):
        record = self.save(vector=[1, 0, 0])
        root = Path(self.temp.name)
        output = root / "batch-output"
        debug = root / "debug"
        output.mkdir()
        debug.mkdir()
        (output / "old.txt").write_text("old result")
        (debug / "old.txt").write_text("old diagnostic")
        with (
            patch.object(file_organizer, "_ROUTE_DIRS", {"safe": output}),
            patch.object(config, "DEBUG_DIR", debug),
            patch.object(config, "DEBUG_TRANSCRIPTS_DIR", debug / "transcripts"),
        ):
            file_organizer.reset_output_folders()
            debug_dump.reset_debug()
        self.assertEqual(library.list_recordings()["total"], 1)
        self.assertTrue(library.audio_path(record["id"]).exists())
        self.assertEqual(len(library.vectors()), 1)

    def test_reindex_configuration_does_not_require_original_calls(self):
        with (
            patch.object(config, "CALLS_DIR", Path(self.temp.name) / "missing-originals"),
            patch.object(config, "OPENAI_API_KEY", "test-only"),
        ):
            config.validate(require_calls=False)
            with self.assertRaises(RuntimeError):
                config.validate()

    def test_deduplication_pending_retry_and_version_reprocessing(self):
        with (
            patch.object(config, "SPEAKER_REPAIR_ENABLED", False),
            patch.object(config, "RELEVANCE_GATE_ENABLED", False),
            patch(
                "ingestion.transcriber.transcribe",
                return_value=Transcription("Person 1: Hello.", "en"),
            ) as transcribe,
            patch(
                "ingestion.analyzer.analyze",
                return_value=analyzer.Verdict(
                    classification="safe", reason="", summary="A greeting."
                ),
            ),
            patch("ingestion.embeddings.embed", side_effect=RuntimeError("offline")),
            patch("ingestion.analysis_version", return_value="v1"),
        ):
            first = ingestion.ingest("first.wav", wav_bytes())
            self.assertEqual(first["index_state"], "pending")
            self.assertEqual(first["status"], "safe")
            with patch("ingestion.embeddings.embed", return_value=np.array([1, 0, 0])):
                second = ingestion.ingest("renamed.wav", wav_bytes())
                self.assertTrue(second["cached"])
                self.assertEqual(second["index_state"], "ready")
                self.assertEqual(first["id"], second["id"])
                transcribe.assert_called_once()
                with patch("ingestion.analysis_version", return_value="v2"):
                    third = ingestion.ingest("first.wav", wav_bytes())
                self.assertFalse(third["cached"])
                self.assertEqual(transcribe.call_count, 2)
            self.assertEqual(library.list_recordings()["total"], 1)

    def test_web_upload_reanalyzes_duplicate_and_updates_same_library_entry(self):
        with (
            patch.object(config, "SPEAKER_REPAIR_ENABLED", False),
            patch.object(config, "RELEVANCE_GATE_ENABLED", False),
            patch(
                "ingestion.transcriber.transcribe",
                return_value=Transcription("Person 1: Hello.", "en"),
            ) as transcribe,
            patch(
                "ingestion.analyzer.analyze",
                side_effect=[
                    analyzer.Verdict(classification="safe", reason="", summary="First analysis."),
                    analyzer.Verdict(classification="safe", reason="", summary="Fresh analysis."),
                ],
            ) as analyze,
            patch("ingestion.embeddings.embed", return_value=np.array([1, 0, 0])),
        ):
            responses = [
                self.client.post(
                    "/api/analyze", data={"file": (io.BytesIO(wav_bytes()), "call.wav")}
                )
                for _ in range(2)
            ]
        self.assertTrue(all(r.status_code == 200 for r in responses))
        first, second = [r.json for r in responses]
        self.assertFalse(first["cached"])
        self.assertFalse(second["cached"])
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(transcribe.call_count, 2)
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(library.list_recordings()["total"], 1)
        self.assertEqual(library.get(second["id"])["summary"], "Fresh analysis.")

    def test_failed_retry_keeps_last_successful_analysis(self):
        verdict = analyzer.Verdict(
            classification="suspicious",
            reason="Rule 1",
            summary="A threat.",
            triggered_rules=[analyzer.TriggeredRule(1, "Threat of violence", "Explicit.")],
        )
        with (
            patch.object(config, "SPEAKER_REPAIR_ENABLED", False),
            patch.object(config, "RELEVANCE_GATE_ENABLED", False),
            patch(
                "ingestion.transcriber.transcribe",
                side_effect=[Transcription("Person 1: A threat.", "en"), RuntimeError("API down")],
            ),
            patch("ingestion.analyzer.analyze", return_value=verdict),
            patch("ingestion.embeddings.embed", return_value=np.array([1, 0, 0])),
        ):
            responses = [
                self.client.post(
                    "/api/analyze", data={"file": (io.BytesIO(wav_bytes()), "call.wav")}
                )
                for _ in range(2)
            ]
        first, second = [r.json for r in responses]
        self.assertNotIn("retry_error", first)
        self.assertEqual(second["status"], "suspicious")
        self.assertIn("RuntimeError", second["retry_error"])
        stored = library.get(first["id"])
        self.assertEqual(stored["status"], "suspicious")
        self.assertFalse(stored["analysis_error"])
        self.assertEqual(stored["index_state"], "ready")
        self.assertEqual(len(library.searchable_records()), 1)

    def test_audio_range_download_and_invalid_ids(self):
        record = self.save()
        response = self.client.get(record["audio_url"], headers={"Range": "bytes=0-43"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.data), 44)
        self.assertTrue(response.headers["Content-Range"].startswith("bytes 0-43/"))
        response.close()
        download = self.client.get(record["audio_url"] + "?download=1")
        self.assertIn("attachment;", download.headers["Content-Disposition"])
        download.close()
        self.assertEqual(self.client.get("/api/recordings/not-an-id/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/recordings/" + "0" * 64).status_code, 404)

    def test_api_validation_and_category_memberships(self):
        self.save(categories=(1, 4))
        self.assertEqual(self.client.get("/api/recordings?category=1").json["total"], 1)
        self.assertEqual(self.client.get("/api/recordings?category=4").json["total"], 1)
        self.assertEqual(self.client.get("/api/recordings?category=6").json["total"], 0)
        self.assertEqual(self.client.get("/api/recordings?page=-1").status_code, 400)
        self.assertEqual(self.client.post("/api/search", json={"query": ""}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/search", json={"query": "x", "categories": [True]}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                "/api/analyze", data={"file": (io.BytesIO(b"bad"), "bad.wav")}
            ).status_code,
            400,
        )

    def test_global_retrieval_and_explicit_filters(self):
        target = self.save(vector=[1, 0, 0])
        other = self.save("other.wav", categories=(1,), text="A threat.", vector=[0, 1, 0])
        with (
            patch("search.plan_query", return_value=self.plan()),
            patch("search.embeddings.embed", return_value=np.array([1, 0, 0])),
        ):
            result = search.search("mailbox", use_reranker=False)
            self.assertEqual(result["results"][0]["id"], target["id"])
            self.assertEqual(
                set(result["diagnostics"]["branch_counts"]), {"global_keyword", "global_vector"}
            )
            filtered = search.search("mailbox", categories=[1], use_reranker=False)
            self.assertEqual([r["id"] for r in filtered["results"]], [other["id"]])
            self.assertEqual(filtered["diagnostics"]["filters"]["categories"], [1])
            with patch("search.plan_query", return_value=self.plan(filters=[6])):
                planned = search.search("only Other", use_reranker=False)
                self.assertEqual([r["id"] for r in planned["results"]], [target["id"]])

    def test_embedding_failure_preserves_keywords_and_reranker_failure_omits_answer(self):
        target = self.save()
        with (
            patch("search.plan_query", side_effect=RuntimeError("offline")),
            patch("search.embeddings.embed", side_effect=RuntimeError("offline")),
            patch("search.rerank", side_effect=ValueError("invented citation")),
        ):
            result = search.search("mailbox")
        self.assertEqual(result["results"][0]["id"], target["id"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["diagnostics"]["reranking"], "unavailable")
        self.assertEqual(len(result["notices"]), 3)

    def test_category_union_success_does_not_search_global_pool(self):
        first = self.save(categories=(1,), vector=[1, 0, 0])
        second = self.save("second.wav", categories=(4,), vector=[1, 0, 0])
        self.save("outside.wav", categories=(6,), vector=[1, 0, 0])
        with (
            patch("search.plan_query", return_value=self.plan(inferred=[1, 4])),
            patch("search.embeddings.embed", return_value=np.array([1, 0, 0])),
            patch("search.library.keywords", wraps=library.keywords) as keywords,
            patch("search.vector_search", wraps=search.vector_search) as vectors,
            patch("search.rerank", side_effect=self.supported),
        ):
            result = search.search("mailbox")
        self.assertEqual({r["id"] for r in result["results"]}, {first["id"], second["id"]})
        keywords.assert_called_once_with("mailbox", None, [1, 4])
        self.assertEqual(vectors.call_args.args[2], [1, 4])
        self.assertEqual(vectors.call_count, 1)
        self.assertFalse(result["diagnostics"]["broadened"])

    def test_wrong_route_broadens_after_rejection_and_reuses_embedding(self):
        target = self.save(categories=(6,), vector=[1, 0, 0])
        wrong = self.save("wrong.wav", categories=(1,), text="Unrelated threat.", vector=[0, 1, 0])

        def judge(query, candidates):
            return self.supported(query, [r for r in candidates if r["id"] == target["id"]])

        with (
            patch("search.plan_query", return_value=self.plan(inferred=[1])) as planner,
            patch("search.embeddings.embed", return_value=np.array([1, 0, 0])) as embed,
            patch("search.rerank", side_effect=judge) as reranker,
        ):
            result = search.search("mailbox")
        self.assertEqual([r["id"] for r in result["results"]], [target["id"]])
        self.assertEqual([r["id"] for r in reranker.call_args_list[0].args[1]], [wrong["id"]])
        self.assertEqual(reranker.call_count, 2)
        planner.assert_called_once()
        embed.assert_called_once()
        d = result["diagnostics"]
        self.assertTrue(d["broadened"])
        self.assertEqual(d["fallback_reason"], "no_supported_match")
        self.assertEqual(
            [s["scope"] for s in d["stages"]], ["inferred_categories", "global_fallback"]
        )

    def test_empty_category_broadens_but_retains_explicit_status(self):
        target = self.save()
        self.save("safe.wav", status="safe", categories=())
        with (
            patch("search.plan_query", return_value=self.plan(inferred=[1], status="suspicious")),
            patch("search.rerank", side_effect=self.supported),
        ):
            result = search.search("mailbox", mode="keyword")
        self.assertEqual([r["id"] for r in result["results"]], [target["id"]])
        self.assertEqual(result["diagnostics"]["fallback_reason"], "no_candidates")
        self.assertEqual(result["diagnostics"]["filters"]["status"], "suspicious")

    def test_explicit_categories_never_broaden_after_rejection(self):
        self.save(categories=(1,))
        self.save("outside.wav", categories=(6,))
        for api_filters, planned_filters in [([1], []), (None, [1])]:
            with (
                self.subTest(api_filters=api_filters),
                patch(
                    "search.plan_query",
                    return_value=self.plan(filters=planned_filters, inferred=[6]),
                ),
                patch("search.rerank", return_value=[]) as reranker,
            ):
                result = search.search("mailbox", categories=api_filters, mode="keyword")
            self.assertEqual(result["results"], [])
            self.assertFalse(result["diagnostics"]["broadened"])
            self.assertEqual(result["diagnostics"]["categories"], [1])
            reranker.assert_called_once()

    def test_reranker_failure_or_disabled_does_not_trigger_broadening(self):
        target = self.save(categories=(1,))
        for enabled in (True, False):
            with (
                self.subTest(enabled=enabled),
                patch("search.plan_query", return_value=self.plan(inferred=[1])),
                patch("search.rerank", side_effect=RuntimeError("offline")),
            ):
                result = search.search("mailbox", mode="keyword", use_reranker=enabled)
            self.assertEqual([r["id"] for r in result["results"]], [target["id"]])
            self.assertFalse(result["diagnostics"]["broadened"])
            self.assertEqual(result["answer"], "")
            self.assertEqual(result["citations"], [])

    def test_safe_and_referenced_searches_bypass_inferred_categories(self):
        target = self.save(status="safe", categories=())
        self.save("threat.wav", categories=(1,))
        plans = [
            self.plan(inferred=[1], status="safe"),
            self.plan(inferred=[1], refs=[target["id"]]),
        ]
        for plan in plans:
            with (
                self.subTest(plan=plan),
                patch("search.plan_query", return_value=plan),
                patch("search.rerank", side_effect=self.supported),
            ):
                result = search.search("mailbox", mode="keyword")
            self.assertEqual([r["id"] for r in result["results"]], [target["id"]])
            self.assertFalse(result["diagnostics"]["broadened"])
            self.assertEqual(result["diagnostics"]["categories"], [])

    def test_no_match_after_both_stages_and_conflicting_filters(self):
        with patch("search.plan_query", return_value=self.plan(inferred=[1])):
            result = search.search("mailbox", mode="keyword")
        self.assertEqual(result["answer"], "No matching recordings found.")
        self.assertEqual(len(result["diagnostics"]["stages"]), 2)
        with (
            patch("search.plan_query", return_value=self.plan(filters=[6])),
            patch("search.embeddings.embed") as embed,
        ):
            result = search.search("only Other", categories=[1])
        self.assertEqual(result["results"], [])
        self.assertFalse(result["diagnostics"]["broadened"])
        embed.assert_not_called()

    def test_rerank_rejects_invented_ids_and_quotes(self):
        record = self.save()
        valid = {
            "recording_id": record["id"],
            "evidence_quote": "midnight mailbox campaign.",
            "match_reason": "Explicit phrase.",
            "answer": "The call mentions a campaign.",
        }
        for bad in [
            dict(valid, recording_id="0" * 64),
            dict(valid, evidence_quote="I never said this."),
        ]:
            with (
                patch("search.openai_json.generate_json", return_value={"matches": [bad]}),
                self.assertRaises(ValueError),
            ):
                search.rerank("mailbox", [record])
        with patch("search.openai_json.generate_json", return_value={"matches": [valid]}):
            self.assertEqual(search.rerank("mailbox", [record])[0]["id"], record["id"])

    def test_followup_only_uses_validated_references_and_empty_results(self):
        record = self.save()
        self.save("unrelated.wav", categories=(1,))
        with (
            patch(
                "search.plan_query", return_value=self.plan("what did he say", refs=[record["id"]])
            ),
            patch("search.rerank", return_value=[]),
        ):
            result = search.search("what did he say", mode="keyword")
        self.assertEqual(result["diagnostics"]["candidate_count"], 1)
        self.assertEqual(result["answer"], "No matching recordings found.")
        self.assertEqual(result["citations"], [])
        with (
            patch("search.openai_json.generate_json", return_value=self.plan(refs=["0" * 64])),
            self.assertRaises(ValueError),
        ):
            search.plan_query("what did he say", [], [record["id"]])

    def test_rrf_uses_ranks_not_incomparable_scores(self):
        result = search.fuse(
            [("keyword", 1, [("a", -400), ("b", -1)]), ("vector", 0.5, [("b", 0.99), ("a", 0.01)])]
        )
        self.assertEqual(result[0]["id"], "a")
        self.assertAlmostEqual(result[0]["rrf_score"], 1 / 61 + 0.5 / 62)


class OtherTests(unittest.TestCase):
    def verdict(self, detections, uncertain=False, flags=None):
        return analyzer._compute_verdict(
            rules_loader.get_active(),
            {"uncertain": uncertain, "summary": "A call.", "detections": detections},
            flags or [],
            "Person 1: I keep sending insults after you told me to stop. I will hurt you.",
        )

    def finding(self, rid, quote="I keep sending insults after you told me to stop."):
        return {
            "id": rid,
            "triggered": True,
            "reason": "Explicit admission",
            "evidence_quote": quote,
        }

    def test_other_requires_affirmative_evidence(self):
        self.assertEqual([r.id for r in self.verdict([self.finding(6)]).triggered_rules], [6])
        self.assertEqual(self.verdict([self.finding(6, "made up")]).triggered_rules, [])
        self.assertEqual(self.verdict([self.finding(6)], uncertain=True).triggered_rules, [])
        self.assertEqual(self.verdict([]).classification, "safe")

    def test_existing_transcript_or_audio_categories_suppress_other(self):
        verdict = self.verdict([self.finding(6), self.finding(1, "I will hurt you.")])
        self.assertEqual([r.id for r in verdict.triggered_rules], [1])
        flags = analyzer.detect_audio_flags(
            rules_loader.get_active(), "[AUDIO: sounds of distress or violence]"
        )
        self.assertEqual(
            [r.id for r in self.verdict([self.finding(6)], flags=flags).triggered_rules], [5]
        )


if __name__ == "__main__":
    unittest.main()
