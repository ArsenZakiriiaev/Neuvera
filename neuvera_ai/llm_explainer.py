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


def _compact_module(result: dict[str, Any] | None, keep_metrics: tuple[str, ...]) -> dict[str, Any] | None:
    if not result:
        return None
    metrics = result.get("metrics") or {}
    return {
        "status": result.get("status"),
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "signals": result.get("signals", []),
        "summary": result.get("summary"),
        "metrics": {k: metrics.get(k) for k in keep_metrics if k in metrics},
    }


def build_session_explainer_prompt(session: dict[str, Any]) -> str:
    compact = {
        "overall_score": session.get("overall_score"),
        "overall_confidence": session.get("overall_confidence"),
        "summary": session.get("summary"),
        "signals": session.get("signals", []),
        "voice": _compact_module(session.get("voice"), (
            "duration_seconds", "jitter", "shimmer", "pitch_variation",
            "pitch_instability", "pause_ratio", "spectral_flatness",
            "audio_quality_score",
        )),
        "tremor": _compact_module(session.get("tremor"), (
            "duration_seconds", "dominant_frequency_hz", "tremor_band_ratio",
            "normalized_amplitude", "peak_amplitude", "oscillation_density",
            "reliability_factor", "video_quality_score",
        )),
        "movement": _compact_module(session.get("movement"), (
            "duration_seconds", "upper_motion", "total_motion", "asymmetry",
            "posture_offset", "neck_drop_ratio", "head_tilt_degrees",
            "head_tremor", "video_quality_score",
        )),
    }
    return (
        "You are explaining a Parkinson's-related screening session for a hackathon demo.\n"
        "Rules:\n"
        "- Do not diagnose Parkinson's disease — frame everything as a screening signal.\n"
        "- Write in plain, friendly language a non-clinician can understand.\n"
        "- Reference the specific numbers and signals you see (voice jitter/shimmer, tremor-band ratio, posture, head tilt, etc.).\n"
        "- Acknowledge uncertainty: mention camera / microphone noise, tracking reliability, and short sample length as possible confounds when relevant.\n"
        "- Return valid JSON with exactly these keys: overview, voice, tremor, movement, next_steps, caution.\n"
        "- overview: 2-3 sentences summarizing overall risk signal and confidence.\n"
        "- voice, tremor, movement: each a 1-2 sentence interpretation of that modality (or 'Not recorded.' if missing).\n"
        "- next_steps: array of 2-3 short practical suggestions (e.g. retry with better lighting, consult clinician if worried).\n"
        "- caution: one sentence reminding the user this is not a diagnosis.\n\n"
        f"Session result:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )


def _run_ollama(prompt: str, model: str, timeout_seconds: int) -> dict[str, Any] | None:
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
    output = ANSI_ESCAPE_RE.sub("", output).replace("\r", "").strip()

    json_start = output.find("{")
    json_end = output.rfind("}")
    candidate = output[json_start : json_end + 1] if (json_start != -1 and json_end > json_start) else output

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"overview": candidate, "caution": "LLM output returned as plain text."}

    return parsed if isinstance(parsed, dict) else None


def ollama_explain_session(
    session: dict[str, Any],
    model: str = "llama3:8b",
    timeout_seconds: int = 120,
) -> dict[str, Any] | None:
    return _run_ollama(build_session_explainer_prompt(session), model, timeout_seconds)


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
