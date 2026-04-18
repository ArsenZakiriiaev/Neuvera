from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import signal


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return float(numerator) / float(denominator)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def normalize_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    peak = np.max(np.abs(values)) if values.size else 0.0
    if peak < 1e-9:
        return values.copy()
    return values / peak


def robust_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(mad * 1.4826)


def detrend_series(values: np.ndarray, window: int) -> np.ndarray:
    baseline = moving_average(values, max(3, window))
    return values - baseline


def band_power_ratio(
    values: np.ndarray,
    fps: float,
    band: tuple[float, float],
    min_frequency: float = 0.2,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 8 or fps <= 0:
        return 0.0, 0.0
    frequencies, power = signal.welch(values, fs=fps, nperseg=min(len(values), 256))
    valid = frequencies >= min_frequency
    if not np.any(valid):
        return 0.0, 0.0
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    total_power = integrate(power[valid], frequencies[valid])
    band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
    band_power = integrate(power[band_mask], frequencies[band_mask]) if np.any(band_mask) else 0.0
    dominant_idx = int(np.argmax(power[valid]))
    dominant_frequency = float(frequencies[valid][dominant_idx])
    return safe_ratio(band_power, total_power), dominant_frequency


def average_pairwise_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 1.0
    diffs = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    return float(np.mean(distances[np.triu_indices(len(points), k=1)]))


def flatten_signals(*signal_lists: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for signal_list in signal_lists:
        for signal_name in signal_list:
            if signal_name not in seen:
                seen.add(signal_name)
                output.append(signal_name)
    return output


def confidence_from_coverage(coverage: float, duration_seconds: float, target_duration: float) -> float:
    duration_factor = clamp01(duration_seconds / max(target_duration, 1e-6))
    return clamp01(0.45 * coverage + 0.55 * duration_factor)


def neutral_blend(score: float, confidence: float, baseline: float = 0.5) -> float:
    return float((score * confidence) + (baseline * (1.0 - confidence)))


def describe_signal_count(count: int) -> str:
    if count <= 0:
        return "No strong irregularity signals were detected."
    if count == 1:
        return "One screening irregularity signal was detected."
    return f"{count} screening irregularity signals were detected."


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(values), axis=-1))


def ensure_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    return values


def mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def std_or_zero(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0
