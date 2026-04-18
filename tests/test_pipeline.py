import unittest
from unittest.mock import patch

from neuvera_ai.contracts import AnalysisResult
from neuvera_ai.pipeline import analyze_session


class PipelineTests(unittest.TestCase):
    def test_analyze_session_returns_partial_result(self) -> None:
        voice = AnalysisResult.ok("voice", 0.62, 0.8, signals=["reduced_voice_variation"])
        tremor = AnalysisResult.insufficient_data(
            "tremor",
            "Hand landmarks were not detected reliably enough for a stable tremor estimate.",
            "poor_hand_tracking",
        )
        movement = AnalysisResult.ok("movement", 0.41, 0.72, signals=["low_arm_mobility"])

        with patch("neuvera_ai.pipeline.analyze_voice_file", return_value=voice), patch(
            "neuvera_ai.pipeline.analyze_tremor_video", return_value=tremor
        ), patch("neuvera_ai.pipeline.analyze_movement_video", return_value=movement):
            result = analyze_session("voice.wav", "hand.mp4", "move.mp4")

        self.assertEqual(result.status, "partial")
        self.assertIsNotNone(result.overall_score)
        self.assertIn("reduced_voice_variation", result.signals)
        self.assertIn("low_arm_mobility", result.signals)


if __name__ == "__main__":
    unittest.main()
