from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import AnalysisResult
from .quality import assess_video_quality
from .signals import signal_details
from .utils import (
    average_pairwise_distance,
    band_power_ratio,
    clamp01,
    confidence_from_coverage,
    detrend_series,
    safe_ratio,
    vector_norm,
)


@dataclass(slots=True)
class TremorAnalysisConfig:
    tremor_band_hz: tuple[float, float] = (3.0, 8.0)
    target_duration_seconds: float = 8.0
    detrend_window_frames: int = 15


def analyze_tremor_landmarks(
    hand_landmarks: np.ndarray,
    fps: float,
    config: TremorAnalysisConfig | None = None,
    detection_rate: float = 1.0,
    visible_point_ratio: float = 1.0,
    reliability_factor: float = 1.0,
) -> AnalysisResult:
    config = config or TremorAnalysisConfig()
    hand_landmarks = np.asarray(hand_landmarks, dtype=float)

    if hand_landmarks.ndim != 3 or hand_landmarks.shape[1] < 5:
        raise ValueError("hand_landmarks must have shape (frames, points, dims).")

    frame_count = hand_landmarks.shape[0]
    quality = assess_video_quality(
        frame_count=frame_count,
        fps=fps,
        detection_rate=detection_rate,
        visible_point_ratio=visible_point_ratio,
    )
    if not quality.is_usable:
        return AnalysisResult.insufficient_data(
            module_name="tremor",
            summary="Hand landmarks were not detected reliably enough for a stable tremor estimate.",
            error_code="poor_hand_tracking",
            metrics=quality.metrics,
            confidence=max(0.05, quality.quality_score * 0.35),
            signals=["poor_video_quality"],
            signal_details=signal_details(["poor_video_quality"]),
        )

    points_2d = hand_landmarks[..., :2]
    centroids = np.mean(points_2d, axis=1)
    hand_scale = float(np.median([average_pairwise_distance(frame) for frame in points_2d]))
    hand_scale = max(hand_scale, 1e-4)

    centered = centroids - np.mean(centroids, axis=0, keepdims=True)
    residual_x = detrend_series(centered[:, 0], window=config.detrend_window_frames)
    residual_y = detrend_series(centered[:, 1], window=config.detrend_window_frames)
    residual = np.column_stack([residual_x, residual_y])
    displacement = vector_norm(residual)

    normalized_amplitude = safe_ratio(float(np.std(displacement)), hand_scale)
    peak_amplitude = safe_ratio(float(np.percentile(displacement, 95)), hand_scale)
    band_ratio_x, dominant_frequency_x = band_power_ratio(
        residual_x,
        fps=fps,
        band=config.tremor_band_hz,
    )
    band_ratio_y, dominant_frequency_y = band_power_ratio(
        residual_y,
        fps=fps,
        band=config.tremor_band_hz,
    )
    if band_ratio_x >= band_ratio_y:
        band_ratio = band_ratio_x
        dominant_frequency = dominant_frequency_x
    else:
        band_ratio = band_ratio_y
        dominant_frequency = dominant_frequency_y

    zero_crossings = np.sum(np.diff(np.signbit(displacement - np.median(displacement))) != 0)
    oscillation_density = safe_ratio(float(zero_crossings), quality.metrics["duration_seconds"], default=0.0)

    amplitude_risk = clamp01((normalized_amplitude - 0.012) / 0.025)
    tremor_band_risk = clamp01((band_ratio - 0.18) / 0.45)
    consistency_risk = clamp01((oscillation_density - 3.0) / 7.0)
    peak_risk = clamp01((peak_amplitude - 0.025) / 0.07)
    in_tremor_band = config.tremor_band_hz[0] <= dominant_frequency <= config.tremor_band_hz[1]
    frequency_gate = 1.0 if in_tremor_band else 0.3
    rhythmicity_gate = clamp01((band_ratio - 0.24) / 0.28) if in_tremor_band else 0.0

    raw_score = float(
        0.38 * tremor_band_risk
        + 0.27 * amplitude_risk
        + 0.20 * consistency_risk
        + 0.15 * peak_risk
    )
    score = raw_score * (0.35 + 0.65 * max(rhythmicity_gate, frequency_gate)) * reliability_factor

    signals: list[str] = []
    if tremor_band_risk > 0.55 and in_tremor_band and reliability_factor >= 0.5:
        signals.append("micro_oscillation_detected")
    if amplitude_risk > 0.6 and peak_risk > 0.35 and in_tremor_band:
        signals.append("hand_instability_detected")
    if consistency_risk > 0.65 and in_tremor_band:
        signals.append("consistent_high_frequency_motion")
    if in_tremor_band and band_ratio > 0.28:
        signals.append("tremor_band_activity")

    confidence = confidence_from_coverage(
        coverage=((0.7 * quality.quality_score) + (0.3 * detection_rate)) * reliability_factor,
        duration_seconds=quality.metrics["duration_seconds"],
        target_duration=config.target_duration_seconds,
    )

    summary = (
        "Possible Parkinson's-related hand movement irregularities were detected in this screening sample. This is not a diagnosis."
        if signals
        else "No strong Parkinson's-related hand movement warning signs were detected in this screening sample."
    )

    return AnalysisResult.ok(
        module_name="tremor",
        score=score,
        confidence=confidence,
        signals=signals,
        signal_details=signal_details(signals),
        metrics={
            **quality.metrics,
            "hand_scale": hand_scale,
            "normalized_amplitude": normalized_amplitude,
            "peak_amplitude": peak_amplitude,
            "tremor_band_ratio": band_ratio,
            "dominant_frequency_hz": dominant_frequency,
            "oscillation_density": oscillation_density,
            "video_quality_score": quality.quality_score,
            "reliability_factor": reliability_factor,
            "raw_score": raw_score,
        },
        summary=summary,
    )
