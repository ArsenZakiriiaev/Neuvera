from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PoseVideoSample:
    landmarks: np.ndarray
    fps: float
    frame_count: int
    detection_rate: float
    visible_point_ratio: float


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "pose_video_adapter requires opencv-python. Install with `pip install neuvera-ai[video]`."
        ) from exc
    return cv2


def _require_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "pose_video_adapter requires mediapipe. Install with `pip install neuvera-ai[video]`."
        ) from exc
    return mp


def extract_pose_landmarks_from_video(path: str | Path) -> PoseVideoSample:
    cv2 = _require_cv2()
    mp = _require_mediapipe()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open pose video file.")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = raw_fps if 5.0 <= raw_fps <= 120.0 else 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detected_frames = 0
    all_landmarks: list[np.ndarray] = []

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                detected_frames += 1
                all_landmarks.append(
                    np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
                        dtype=float,
                    )
                )

    capture.release()

    if not all_landmarks:
        raise ValueError("No pose landmarks detected in video.")

    landmarks = np.stack(all_landmarks, axis=0)
    detection_rate = detected_frames / max(frame_count, detected_frames, 1)
    # Score visibility only on the upper-body landmarks we actually analyze —
    # lower body (ankles, knees) often sits out of frame on a laptop webcam and would
    # otherwise drag the coverage metric down even on a perfectly usable recording.
    upper_body_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
    visibility_channel = landmarks[..., 3] if landmarks.shape[-1] > 3 else np.ones(landmarks.shape[:2])
    visible_point_ratio = float(np.mean(visibility_channel[:, upper_body_indices] > 0.5))
    return PoseVideoSample(
        landmarks=landmarks,
        fps=fps,
        frame_count=landmarks.shape[0],
        detection_rate=detection_rate,
        visible_point_ratio=visible_point_ratio,
    )
