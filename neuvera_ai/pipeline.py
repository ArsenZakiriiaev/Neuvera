from __future__ import annotations

from pathlib import Path

from .audio_adapter import load_audio_sample
from .contracts import AnalysisResult, SessionAnalysisResult
from .hand_video_adapter import extract_hand_landmarks_from_video
from .movement_analysis import analyze_pose_landmarks
from .pose_video_adapter import extract_pose_landmarks_from_video
from .scoring import aggregate_screening_results
from .tremor_analysis import analyze_tremor_landmarks
from .voice_analysis import analyze_voice_waveform


def analyze_voice_file(path: str | Path) -> AnalysisResult:
    try:
        sample = load_audio_sample(path)
    except FileNotFoundError:
        return AnalysisResult.failed("voice", "Audio file was not found.", "file_not_found")
    except Exception as exc:
        return AnalysisResult.insufficient_data(
            "voice",
            f"Audio input could not be processed: {exc}",
            "audio_processing_error",
        )
    return analyze_voice_waveform(sample.waveform, sample.sample_rate)


def analyze_tremor_video(path: str | Path) -> AnalysisResult:
    try:
        sample = extract_hand_landmarks_from_video(path)
    except FileNotFoundError:
        return AnalysisResult.failed("tremor", "Hand video file was not found.", "file_not_found")
    except Exception as exc:
        return AnalysisResult.insufficient_data(
            "tremor",
            f"Hand video input could not be processed: {exc}",
            "hand_video_processing_error",
        )
    return analyze_tremor_landmarks(
        sample.landmarks,
        fps=sample.fps,
        detection_rate=sample.detection_rate,
        visible_point_ratio=sample.visible_point_ratio,
        reliability_factor=sample.reliability_factor,
    )


def analyze_movement_video(path: str | Path) -> AnalysisResult:
    try:
        sample = extract_pose_landmarks_from_video(path)
    except FileNotFoundError:
        return AnalysisResult.failed("movement", "Movement video file was not found.", "file_not_found")
    except Exception as exc:
        return AnalysisResult.insufficient_data(
            "movement",
            f"Movement video input could not be processed: {exc}",
            "movement_video_processing_error",
        )
    return analyze_pose_landmarks(
        sample.landmarks,
        fps=sample.fps,
        detection_rate=sample.detection_rate,
        visible_point_ratio=sample.visible_point_ratio,
    )


def analyze_session(
    audio_path: str | Path | None = None,
    hand_video_path: str | Path | None = None,
    movement_video_path: str | Path | None = None,
) -> SessionAnalysisResult:
    voice = analyze_voice_file(audio_path) if audio_path is not None else None
    tremor = analyze_tremor_video(hand_video_path) if hand_video_path is not None else None
    movement = analyze_movement_video(movement_video_path) if movement_video_path is not None else None
    return aggregate_screening_results(voice=voice, tremor=tremor, movement=movement)
