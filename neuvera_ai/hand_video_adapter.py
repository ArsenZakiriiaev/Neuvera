from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class HandVideoSample:
    landmarks: np.ndarray
    fps: float
    frame_count: int
    detection_rate: float
    visible_point_ratio: float
    reliability_factor: float = 1.0


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "hand_video_adapter requires opencv-python. Install with `pip install neuvera-ai[video]`."
        ) from exc
    return cv2


def _require_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "hand_video_adapter requires mediapipe. Install with `pip install neuvera-ai[video]`."
        ) from exc
    return mp


def _extract_hand_landmarks_with_optical_flow(path: str | Path) -> HandVideoSample:
    cv2 = _require_cv2()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open hand video file.")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = raw_fps if 5.0 <= raw_fps <= 120.0 else 30.0
    tracked_frames: list[np.ndarray] = []
    detection_frames = 0

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise ValueError("Could not read frames from hand video file.")

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    height, width = first_gray.shape[:2]
    x0, x1 = int(width * 0.2), int(width * 0.8)
    y0, y1 = int(height * 0.2), int(height * 0.8)
    center_mask = np.zeros_like(first_gray)
    center_mask[y0:y1, x0:x1] = 255
    outer_mask = np.full_like(first_gray, 255)
    outer_mask[y0:y1, x0:x1] = 0

    features = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=21,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=center_mask,
    )
    if features is None or len(features) < 5:
        capture.release()
        raise ValueError("Could not find enough trackable points in hand video.")

    background_features = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=64,
        qualityLevel=0.01,
        minDistance=9,
        blockSize=7,
        mask=outer_mask,
    )

    prev_gray = first_gray
    prev_points = features.reshape(-1, 2)
    prev_background_points = (
        background_features.reshape(-1, 2)
        if background_features is not None and len(background_features) >= 5
        else None
    )
    frame_step = 2

    def _normalize(points: np.ndarray) -> np.ndarray:
        normalized = np.zeros((points.shape[0], 3), dtype=float)
        normalized[:, 0] = points[:, 0] / max(width, 1)
        normalized[:, 1] = points[:, 1] / max(height, 1)
        return normalized

    tracked_frames.append(_normalize(prev_points))
    detection_frames += 1
    raw_frame_index = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        raw_frame_index += 1
        if raw_frame_index % frame_step != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points.astype(np.float32),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            prev_gray = gray
            continue

        background_shift = np.zeros(2, dtype=np.float32)
        if prev_background_points is not None and len(prev_background_points) >= 5:
            next_bg_points, bg_status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_background_points.astype(np.float32),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_bg_points is not None and bg_status is not None:
                bg_status = bg_status.reshape(-1).astype(bool)
                if np.any(bg_status):
                    bg_shift_vectors = next_bg_points[bg_status] - prev_background_points[bg_status]
                    background_shift = np.median(bg_shift_vectors, axis=0).astype(np.float32)
                    prev_background_points = next_bg_points[bg_status].reshape(-1, 2)

        status = status.reshape(-1).astype(bool)
        good_points = next_points[status] - background_shift
        if good_points.shape[0] < 5:
            # Re-seed feature tracking on the current frame.
            reseed = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=21,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
                mask=center_mask,
            )
            if reseed is None or len(reseed) < 5:
                prev_gray = gray
                continue
            prev_points = reseed.reshape(-1, 2)
            tracked_frames.append(_normalize(prev_points))
            detection_frames += 1
            prev_gray = gray
            continue

        prev_points = good_points.reshape(-1, 2)
        tracked_frames.append(_normalize(prev_points))
        detection_frames += 1
        prev_gray = gray

    capture.release()

    if not tracked_frames:
        raise ValueError("No trackable hand motion points were extracted from video.")

    point_count = max(frame.shape[0] for frame in tracked_frames)
    padded_frames = []
    for frame in tracked_frames:
        if frame.shape[0] == point_count:
            padded_frames.append(frame)
            continue
        padded = np.repeat(frame[-1:, :], point_count - frame.shape[0], axis=0)
        padded_frames.append(np.vstack([frame, padded]))

    landmarks = np.stack(padded_frames, axis=0)
    effective_fps = fps / frame_step
    tracked_ratio = detection_frames / max(raw_frame_index // frame_step + 1, 1)
    detection_rate = tracked_ratio * 0.9
    visible_point_ratio = float(np.mean(np.isfinite(landmarks[..., :2])))
    return HandVideoSample(
        landmarks=landmarks,
        fps=effective_fps,
        frame_count=landmarks.shape[0],
        detection_rate=detection_rate,
        visible_point_ratio=visible_point_ratio,
        reliability_factor=0.6,
    )


def extract_hand_landmarks_from_video(path: str | Path) -> HandVideoSample:
    return _extract_hand_landmarks_with_optical_flow(path)
