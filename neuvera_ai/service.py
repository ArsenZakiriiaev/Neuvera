from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_explainer import ollama_explain_tremor
from .pipeline import (
    analyze_movement_video,
    analyze_session,
    analyze_tremor_video,
    analyze_voice_file,
)


def run_voice_analysis_from_file(path: str | Path) -> dict[str, Any]:
    return analyze_voice_file(path).to_dict()


def run_tremor_analysis_from_video(
    path: str | Path,
    use_llm: bool = False,
    llm_model: str = "llama3:8b",
) -> dict[str, Any]:
    payload = analyze_tremor_video(path).to_dict()
    if use_llm:
        payload["llm_model"] = llm_model
        payload["llm_interpretation"] = ollama_explain_tremor(payload, model=llm_model)
    return payload


def run_movement_analysis_from_video(path: str | Path) -> dict[str, Any]:
    return analyze_movement_video(path).to_dict()


def run_full_analysis(
    audio_path: str | Path | None = None,
    hand_video_path: str | Path | None = None,
    movement_video_path: str | Path | None = None,
) -> dict[str, Any]:
    return analyze_session(
        audio_path=audio_path,
        hand_video_path=hand_video_path,
        movement_video_path=movement_video_path,
    ).to_dict()
