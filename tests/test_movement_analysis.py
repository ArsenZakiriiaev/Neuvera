import unittest

from neuvera_ai.demo_fixtures import make_pose_landmarks
from neuvera_ai.movement_analysis import analyze_pose_landmarks


class MovementAnalysisTests(unittest.TestCase):
    def test_movement_analysis_flags_low_motion(self) -> None:
        low_motion_pose, fps = make_pose_landmarks("mild_risk")
        result = analyze_pose_landmarks(low_motion_pose, fps=fps)

        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.score)
        self.assertGreater(result.score, 0.3)
        self.assertTrue(
            "low_motion_amplitude" in result.signals
            or "low_arm_mobility" in result.signals
        )


if __name__ == "__main__":
    unittest.main()
