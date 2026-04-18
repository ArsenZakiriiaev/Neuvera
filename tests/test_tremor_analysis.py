import unittest

from neuvera_ai.demo_fixtures import make_hand_landmarks
from neuvera_ai.tremor_analysis import analyze_tremor_landmarks


class TremorAnalysisTests(unittest.TestCase):
    def test_tremor_analysis_detects_oscillation_signal(self) -> None:
        landmarks, fps = make_hand_landmarks("mild_risk")
        result = analyze_tremor_landmarks(landmarks, fps=fps)

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.score)
        self.assertGreater(result.score, 0.35)
        self.assertTrue(
            "micro_oscillation_detected" in result.signals
            or "tremor_band_activity" in result.signals
        )

    def test_tremor_analysis_returns_insufficient_data_for_short_sequence(self) -> None:
        landmarks, fps = make_hand_landmarks("bad_quality")
        result = analyze_tremor_landmarks(landmarks, fps=fps, detection_rate=0.1, visible_point_ratio=0.2)

        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(result.error_code, "poor_hand_tracking")


if __name__ == "__main__":
    unittest.main()
