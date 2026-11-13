"""Offline regressions for evidence handling; no API requests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyzer
import classifier
import rules_loader
import speaker_repair
import transcriber


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = rules_loader.load("threat_detection")

    def test_spoken_marker_is_not_acoustic_evidence(self):
        speech = "Person 1: The label says AUDIO: sounds of distress or violence."
        self.assertEqual(analyzer.detect_audio_flags(self.rules, speech), [])
        self.assertEqual(analyzer.strip_audio_markers(self.rules, speech), speech)

    def test_standalone_marker_survives_normalization_and_flags_once(self):
        marker = "[audio:  sounds of distress or violence]"
        transcript = transcriber._normalize_turns(f"Person 1: Hello?\n{marker}\n{marker}")
        flags = analyzer.detect_audio_flags(self.rules, transcript)
        self.assertEqual([f.id for f in flags], [5])
        self.assertEqual(analyzer.strip_audio_markers(self.rules, transcript), "Person 1: Hello?")

    def test_labels_normalize_to_canonical_form(self):
        text = transcriber._normalize_turns("PERSON 1:hi  person 2 :  hello")
        self.assertEqual(text, "Person 1: hi\nPerson 2: hello")
        self.assertEqual(
            speaker_repair._parse_turns(text), [("Person 1", "hi"), ("Person 2", "hello")]
        )
        self.assertEqual(transcriber._normalize_turns("Person 1: hi\\nPerson 2: hello"), text)

    def test_unresolved_concern_is_skipped(self):
        verdict = analyzer._compute_verdict(self.rules, {"uncertain": True, "detections": []}, [])
        self.assertEqual(classifier.classify(verdict), classifier.Route.SKIPPED)
        self.assertTrue(verdict.reason)

    def test_supported_threat_survives_other_uncertainty(self):
        verdict = analyzer._compute_verdict(
            self.rules,
            {
                "uncertain": True,
                "detections": [{"id": 1, "triggered": True, "reason": "Explicit threat"}],
            },
            [],
        )
        self.assertEqual(classifier.classify(verdict), classifier.Route.SUSPICIOUS)
        self.assertEqual([r.id for r in verdict.triggered_rules], [1])

    def test_audio_only_evidence_survives_unreadable_speech(self):
        flags = analyzer.detect_audio_flags(self.rules, "[AUDIO: sounds of distress or violence]")
        verdict = analyzer._compute_verdict(
            self.rules, {"uncertain": True, "detections": []}, flags
        )
        self.assertEqual(classifier.classify(verdict), classifier.Route.SUSPICIOUS)
        self.assertEqual([r.id for r in verdict.triggered_rules], [5])

    def test_text_model_cannot_invent_audio_flag(self):
        verdict = analyzer._compute_verdict(
            self.rules,
            {
                "uncertain": False,
                "detections": [{"id": 5, "triggered": True, "reason": "Guessed screaming"}],
            },
            [],
        )
        self.assertEqual(classifier.classify(verdict), classifier.Route.SAFE)


if __name__ == "__main__":
    unittest.main()
