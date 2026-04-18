import unittest
from unittest.mock import patch

from neuvera_ai.service import run_tremor_analysis_from_video


class LlmServiceTests(unittest.TestCase):
    def test_tremor_service_can_attach_llm_interpretation(self) -> None:
        fake_result = {
            "module_name": "tremor",
            "status": "ok",
            "score": 0.2,
            "confidence": 0.7,
            "signals": [],
            "metrics": {},
            "summary": "No strong warning signs detected.",
        }
        fake_llm = {
            "overview": "This screening result looks low-risk.",
            "evidence": ["Low score", "No strong tremor-band pattern"],
            "caution": "This is not a diagnosis.",
        }

        with patch("neuvera_ai.service.analyze_tremor_video") as analyze, patch(
            "neuvera_ai.service.ollama_explain_tremor",
            return_value=fake_llm,
        ):
            analyze.return_value.to_dict.return_value = fake_result
            payload = run_tremor_analysis_from_video("demo.mp4", use_llm=True)

        self.assertEqual(payload["llm_interpretation"]["overview"], fake_llm["overview"])
        self.assertEqual(payload["llm_model"], "llama3:8b")


if __name__ == "__main__":
    unittest.main()
