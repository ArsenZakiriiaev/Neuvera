from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import clamp01, safe_ratio


@dataclass(slots=True)
class AudioQualityResult:
    is_usable: bool
    metrics: dict[str, float]
    reasons: list[str]
    quality_score: float


@dataclass(slots=True)
class VideoQualityResult:
    is_usable: bool
    metrics: dict[str, float]
    reasons: list[str]
    quality_score: float


def assess_audio_quality(
    waveform: np.ndarray,
    sample_rate: int,
    rms: np.ndarray | None = None,
    voiced_mask: np.ndarray | None = None,
) -> AudioQualityResult:
    waveform = np.asarray(waveform, dtype=float)
    duration_seconds = len(waveform) / max(sample_rate, 1)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms_value = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    dynamic_range = safe_ratio(peak, rms_value + 1e-6)
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask is not None and len(voiced_mask) else 0.0
    rms_std = float(np.std(rms)) if rms is not None and len(rms) else 0.0

    reasons: list[str] = []
    if duration_seconds < 1.5:
        reasons.append("audio_too_short")
    if peak < 0.03:
        reasons.append("audio_too_quiet")
    if rms_value < 0.01:
        reasons.append("low_audio_energy")
    if voiced_mask is not None and voiced_ratio < 0.2:
        reasons.append("low_voiced_coverage")

    quality_score = clamp01(
        0.35 * clamp01(duration_seconds / 6.0)
        + 0.25 * clamp01((peak - 0.02) / 0.18)
        + 0.20 * clamp01((rms_value - 0.005) / 0.08)
        + 0.20 * clamp01((voiced_ratio - 0.1) / 0.7 if voiced_mask is not None else 0.5)
    )

    return AudioQualityResult(
        is_usable=len(reasons) == 0,
        metrics={
            "duration_seconds": duration_seconds,
            "peak_amplitude": peak,
            "rms_energy": rms_value,
            "dynamic_range": dynamic_range,
            "voiced_ratio": voiced_ratio,
            "rms_std": rms_std,
        },
        reasons=reasons,
        quality_score=quality_score,
    )


def assess_video_quality(
    frame_count: int,
    fps: float,
    detection_rate: float,
    visible_point_ratio: float,
) -> VideoQualityResult:
    duration_seconds = frame_count / max(fps, 1e-6)
    reasons: list[str] = []
    if frame_count < 12:
        reasons.append("too_few_frames")
    if duration_seconds < 2.0:
        reasons.append("video_too_short")
    if detection_rate < 0.35:
        reasons.append("low_detection_rate")
    if visible_point_ratio < 0.35:
        reasons.append("low_landmark_visibility")

    quality_score = clamp01(
        0.30 * clamp01(duration_seconds / 8.0)
        + 0.40 * clamp01((detection_rate - 0.2) / 0.8)
        + 0.30 * clamp01((visible_point_ratio - 0.2) / 0.8)
    )

    return VideoQualityResult(
        is_usable=len(reasons) == 0,
        metrics={
            "frame_count": float(frame_count),
            "fps": float(fps),
            "duration_seconds": duration_seconds,
            "detection_rate": detection_rate,
            "visible_point_ratio": visible_point_ratio,
        },
        reasons=reasons,
        quality_score=quality_score,
    )
