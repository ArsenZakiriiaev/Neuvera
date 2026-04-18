from __future__ import annotations

import numpy as np


def make_voice_waveform(case: str, sample_rate: int = 16000, duration_seconds: float = 4.0) -> np.ndarray:
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    if case == "normal":
        return 0.35 * np.sin(2 * np.pi * (180 + 15 * np.sin(2 * np.pi * 1.2 * time)) * time)
    if case == "mild_risk":
        signal = 0.18 * np.sin(2 * np.pi * (155 + 2 * np.sin(2 * np.pi * 6.0 * time)) * time)
        signal *= 1.0 + 0.05 * np.sin(2 * np.pi * 0.8 * time)
        return signal
    if case == "bad_quality":
        return 0.004 * np.random.default_rng(7).normal(size=time.shape[0])
    raise ValueError(f"Unknown voice fixture case: {case}")


def make_hand_landmarks(case: str, frames: int = 180, fps: float = 30.0) -> tuple[np.ndarray, float]:
    points = 21
    time = np.arange(frames) / fps
    landmarks = np.zeros((frames, points, 3), dtype=float)
    for point_index in range(points):
        landmarks[:, point_index, 0] = 0.5 + (point_index * 0.001)
        landmarks[:, point_index, 1] = 0.5 + (point_index * 0.0015)

    if case == "normal":
        landmarks[:, :, 0] += 0.001 * np.sin(2 * np.pi * 1.2 * time)[:, None]
    elif case == "mild_risk":
        landmarks[:, :, 0] += 0.01 * np.sin(2 * np.pi * 5.0 * time)[:, None]
    elif case == "bad_quality":
        landmarks = landmarks[:8]
    else:
        raise ValueError(f"Unknown hand fixture case: {case}")
    return landmarks, fps


def make_pose_landmarks(case: str, frames: int = 120, fps: float = 30.0) -> tuple[np.ndarray, float]:
    pose = np.zeros((frames, 33, 4), dtype=float)
    time = np.linspace(0, 2 * np.pi, frames)

    pose[:, 11] = [0.4, 0.3, 0.0, 0.99]
    pose[:, 12] = [0.6, 0.3, 0.0, 0.99]
    pose[:, 23] = [0.45, 0.6, 0.0, 0.99]
    pose[:, 24] = [0.55, 0.6, 0.0, 0.99]
    pose[:, 27] = [0.45, 0.95, 0.0, 0.99]
    pose[:, 28] = [0.55, 0.95, 0.0, 0.99]

    if case == "normal":
        amplitude = 0.04
    elif case == "mild_risk":
        amplitude = 0.005
    elif case == "bad_quality":
        return pose[:8], fps
    else:
        raise ValueError(f"Unknown pose fixture case: {case}")

    pose[:, 15, 0] = 0.35 + amplitude * np.sin(time)
    pose[:, 15, 1] = 0.45
    pose[:, 16, 0] = 0.65 + amplitude * np.sin(time)
    pose[:, 16, 1] = 0.45
    return pose, fps
