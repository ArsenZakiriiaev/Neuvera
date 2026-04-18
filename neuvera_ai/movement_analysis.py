from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import AnalysisResult
from .quality import assess_video_quality
from .signals import signal_details
from .utils import clamp01, confidence_from_coverage, detrend_series, safe_ratio, vector_norm

POSE = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
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
    nose = xy[:, POSE["nose"]]
    left_ear = xy[:, POSE["left_ear"]]
    right_ear = xy[:, POSE["right_ear"]]

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

    # Head / neck analysis: stooped posture, lateral head tilt, and head tremor.
    # Neck drop: in an upright frontal view, the nose sits well above the shoulder line.
    # If the head is carried forward/down (stooped), the vertical gap shrinks and the
    # head moves toward the shoulder line (y increases toward bottom in image coords).
    nose_shoulder_gap = (shoulder_center[:, 1] - nose[:, 1]) / body_scale  # positive when head above shoulders
    neck_drop_ratio = float(np.mean(nose_shoulder_gap))  # smaller = more stooped
    # Lateral head tilt: angle of the ear-to-ear line relative to horizontal.
    ear_vec = left_ear - right_ear
    head_tilt_angles = np.degrees(np.arctan2(ear_vec[:, 1], ear_vec[:, 0] + 1e-6))
    head_tilt_angles = np.where(head_tilt_angles > 90, head_tilt_angles - 180, head_tilt_angles)
    head_tilt_angles = np.where(head_tilt_angles < -90, head_tilt_angles + 180, head_tilt_angles)
    head_tilt_abs_deg = float(np.mean(np.abs(head_tilt_angles)))
    # Head tremor: detrended nose jitter relative to shoulder center, normalized by body scale.
    nose_rel = nose - shoulder_center
    nose_x_res = detrend_series(nose_rel[:, 0], window=15)
    nose_y_res = detrend_series(nose_rel[:, 1], window=15)
    head_tremor = safe_ratio(
        float(np.sqrt(np.mean(nose_x_res ** 2) + np.mean(nose_y_res ** 2))),
        body_scale,
    )

    low_motion_risk = clamp01((0.42 - total_motion) / 0.22)
    upper_rigidity_risk = clamp01((0.34 - upper_motion) / 0.18)
    asymmetry_risk = clamp01((asymmetry - 0.18) / 0.35)
    posture_risk = clamp01((posture_offset - 0.03) / 0.08)
    # Typical upright neck gap is ~0.55-0.85 of body_scale; stoop brings it under ~0.45.
    stooped_posture_risk = clamp01((0.5 - neck_drop_ratio) / 0.3)
    head_tilt_risk = clamp01((head_tilt_abs_deg - 6.0) / 14.0)
    head_tremor_risk = clamp01((head_tremor - 0.008) / 0.02)

    score = float(
        0.28 * low_motion_risk
        + 0.24 * upper_rigidity_risk
        + 0.16 * asymmetry_risk
        + 0.10 * posture_risk
        + 0.12 * stooped_posture_risk
        + 0.05 * head_tilt_risk
        + 0.05 * head_tremor_risk
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
    if stooped_posture_risk > 0.6:
        signals.append("stooped_posture")
    if head_tilt_risk > 0.6:
        signals.append("head_tilt")
    if head_tremor_risk > 0.6:
        signals.append("head_tremor")

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
            "neck_drop_ratio": neck_drop_ratio,
            "head_tilt_degrees": head_tilt_abs_deg,
            "head_tremor": head_tremor,
            "video_quality_score": quality.quality_score,
        },
        summary=summary,
    )
