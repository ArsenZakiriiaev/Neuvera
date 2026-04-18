from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import AnalysisResult
from .quality import assess_video_quality
from .signals import signal_details
from .utils import clamp01, confidence_from_coverage, safe_ratio, vector_norm

POSE = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass(slots=True)
class MovementAnalysisConfig:
    target_duration_seconds: float = 8.0


def _trajectory_range(points: np.ndarray) -> float:
    displacement = points - np.mean(points, axis=0, keepdims=True)
    return float(np.percentile(vector_norm(displacement), 95))


def analyze_pose_landmarks(
    pose_landmarks: np.ndarray,
    fps: float,
    config: MovementAnalysisConfig | None = None,
    detection_rate: float = 1.0,
    visible_point_ratio: float = 1.0,
) -> AnalysisResult:
    config = config or MovementAnalysisConfig()
    pose_landmarks = np.asarray(pose_landmarks, dtype=float)

    if pose_landmarks.ndim != 3 or pose_landmarks.shape[1] <= POSE["right_ankle"]:
        raise ValueError("pose_landmarks must have shape (frames, points, dims) with MediaPipe Pose indexes.")

    frame_count = pose_landmarks.shape[0]
    quality = assess_video_quality(
        frame_count=frame_count,
        fps=fps,
        detection_rate=detection_rate,
        visible_point_ratio=visible_point_ratio,
    )
    if not quality.is_usable:
        return AnalysisResult.insufficient_data(
            module_name="movement",
            summary="Pose landmarks were not detected reliably enough for a movement estimate.",
            error_code="poor_pose_tracking",
            metrics=quality.metrics,
            confidence=max(0.05, quality.quality_score * 0.35),
            signals=["poor_video_quality"],
            signal_details=signal_details(["poor_video_quality"]),
        )

    xy = pose_landmarks[..., :2]
    left_shoulder = xy[:, POSE["left_shoulder"]]
    right_shoulder = xy[:, POSE["right_shoulder"]]
    left_wrist = xy[:, POSE["left_wrist"]]
    right_wrist = xy[:, POSE["right_wrist"]]
    left_ankle = xy[:, POSE["left_ankle"]]
    right_ankle = xy[:, POSE["right_ankle"]]
    left_hip = xy[:, POSE["left_hip"]]
    right_hip = xy[:, POSE["right_hip"]]

    torso_center = (left_hip + right_hip + left_shoulder + right_shoulder) / 4.0
    shoulder_width = float(np.median(vector_norm(left_shoulder - right_shoulder)))
    hip_width = float(np.median(vector_norm(left_hip - right_hip)))
    body_scale = max((shoulder_width + hip_width) / 2.0, 1e-4)

    left_wrist_range = safe_ratio(_trajectory_range(left_wrist - torso_center), body_scale)
    right_wrist_range = safe_ratio(_trajectory_range(right_wrist - torso_center), body_scale)
    left_ankle_range = safe_ratio(_trajectory_range(left_ankle - torso_center), body_scale)
    right_ankle_range = safe_ratio(_trajectory_range(right_ankle - torso_center), body_scale)

    upper_motion = (left_wrist_range + right_wrist_range) / 2.0
    lower_motion = (left_ankle_range + right_ankle_range) / 2.0
    total_motion = (upper_motion * 0.6) + (lower_motion * 0.4)

    arm_asymmetry = safe_ratio(abs(left_wrist_range - right_wrist_range), upper_motion + 1e-6)
    leg_asymmetry = safe_ratio(abs(left_ankle_range - right_ankle_range), lower_motion + 1e-6)
    asymmetry = (arm_asymmetry * 0.65) + (leg_asymmetry * 0.35)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    posture_offset = safe_ratio(float(np.std((shoulder_center - hip_center)[:, 0])), body_scale)

    low_motion_risk = clamp01((0.42 - total_motion) / 0.22)
    upper_rigidity_risk = clamp01((0.34 - upper_motion) / 0.18)
    asymmetry_risk = clamp01((asymmetry - 0.18) / 0.35)
    posture_risk = clamp01((posture_offset - 0.03) / 0.08)

    score = float(
        0.35 * low_motion_risk
        + 0.30 * upper_rigidity_risk
        + 0.20 * asymmetry_risk
        + 0.15 * posture_risk
    )

    signals: list[str] = []
    if low_motion_risk > 0.55:
        signals.append("low_motion_amplitude")
    if upper_rigidity_risk > 0.55:
        signals.append("low_arm_mobility")
    if asymmetry_risk > 0.55:
        signals.append("movement_asymmetry")
    if posture_risk > 0.6:
        signals.append("posture_irregularity")

    confidence = confidence_from_coverage(
        coverage=(0.7 * quality.quality_score) + (0.3 * detection_rate),
        duration_seconds=quality.metrics["duration_seconds"],
        target_duration=config.target_duration_seconds,
    )

    summary = (
        "Possible Parkinson's-related movement or posture irregularities were detected in this screening sample. This is not a diagnosis."
        if signals
        else "No strong Parkinson's-related movement or posture warning signs were detected in this screening sample."
    )

    return AnalysisResult.ok(
        module_name="movement",
        score=score,
        confidence=confidence,
        signals=signals,
        signal_details=signal_details(signals),
        metrics={
            **quality.metrics,
            "body_scale": body_scale,
            "upper_motion": upper_motion,
            "lower_motion": lower_motion,
            "total_motion": total_motion,
            "asymmetry": asymmetry,
            "posture_offset": posture_offset,
            "video_quality_score": quality.quality_score,
        },
        summary=summary,
    )
