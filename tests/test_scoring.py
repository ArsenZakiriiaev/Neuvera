import unittest

from neuvera_ai.contracts import AnalysisResult, SignalDetail
from neuvera_ai.scoring import aggregate_screening_results


class ScoringTests(unittest.TestCase):
    def test_aggregate_screening_results_includes_scores_and_signals(self) -> None:
        voice = AnalysisResult.ok(
            module_name="voice",
            score=0.7,
            confidence=0.8,
            signals=["voice_instability_detected"],
        )
        tremor = AnalysisResult.ok(
            module_name="tremor",
            score=0.6,
            confidence=0.7,
            signals=["hand_instability_detected"],
        )

        result = aggregate_screening_results(voice=voice, tremor=tremor)

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.overall_score)
        self.assertGreater(result.overall_score, 0.5)
        self.assertLess(result.overall_score, 1.0)
        self.assertEqual(result.voice.score, 0.7)
        self.assertEqual(result.tremor.score, 0.6)
        self.assertIn("voice_instability_detected", result.signals)
        self.assertIn("hand_instability_detected", result.signals)

    def test_aggregate_handles_only_insufficient_modules(self) -> None:
        voice = AnalysisResult.insufficient_data(
            module_name="voice",
            summary="Audio quality was too weak.",
            error_code="poor_audio_quality",
        )

        result = aggregate_screening_results(voice=voice)

        self.assertEqual(result.status, "insufficient_data")
        self.assertIsNone(result.overall_score)

    def test_module_to_dict_serializes_signal_details(self) -> None:
        voice = AnalysisResult.ok(
            module_name="voice",
            score=0.6,
            confidence=0.7,
            signals=["reduced_voice_variation"],
        )
        voice.signal_details = [
            SignalDetail(
                signal="reduced_voice_variation",
                label="Reduced voice variation",
                description="The voice sample showed flatter pitch or intensity changes than expected.",
            )
        ]

        payload = voice.to_dict()

        self.assertEqual(payload["signal_details"][0]["signal"], "reduced_voice_variation")


if __name__ == "__main__":
    unittest.main()
