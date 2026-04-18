from __future__ import annotations

import json
import re
import subprocess
from typing import Any


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def build_tremor_explainer_prompt(result: dict[str, Any]) -> str:
    compact = {
        "module_name": result.get("module_name"),
        "status": result.get("status"),
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "signals": result.get("signals", []),
        "metrics": {
            key: result.get("metrics", {}).get(key)
            for key in (
                "duration_seconds",
                "dominant_frequency_hz",
                "tremor_band_ratio",
                "normalized_amplitude",
                "peak_amplitude",
                "oscillation_density",
                "reliability_factor",
                "video_quality_score",
            )
            if key in result.get("metrics", {})
        },
        "summary": result.get("summary"),
    }
    return (
        "You are helping explain a Parkinson's-related screening result for a hackathon demo.\n"
        "Rules:\n"
        "- Do not diagnose Parkinson's disease.\n"
        "- Describe this only as a screening interpretation.\n"
        "- Mention if the signal may still reflect camera motion, compression, tracking noise, or limited reliability.\n"
        "- Be concise.\n"
        "- Return valid JSON with exactly these keys: overview, evidence, caution.\n"
        "- overview: one short paragraph.\n"
        "- evidence: array of 2 to 4 short bullet-style strings.\n"
        "- caution: one short sentence.\n\n"
        f"Analysis result:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )


def ollama_explain_tremor(
    result: dict[str, Any],
    model: str = "llama3:8b",
    timeout_seconds: int = 90,
) -> dict[str, Any] | None:
    prompt = build_tremor_explainer_prompt(result)
    try:
        completed = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    output = (completed.stdout or "").strip()
    if not output:
        return None
    output = ANSI_ESCAPE_RE.sub("", output)
    output = output.replace("\r", "").strip()

    json_start = output.find("{")
    json_end = output.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        candidate = output[json_start : json_end + 1]
    else:
        candidate = output

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "overview": candidate,
            "evidence": [],
            "caution": "LLM output was returned as plain text and should be treated as supplementary explanation only.",
        }

    if not isinstance(parsed, dict):
        return None
    return parsed
