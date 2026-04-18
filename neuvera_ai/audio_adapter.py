from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import normalize_signal


@dataclass(slots=True)
class AudioSample:
    waveform: np.ndarray
    sample_rate: int
    features: dict[str, np.ndarray | float]


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "audio_adapter requires librosa. Install with `pip install neuvera-ai[audio]`."
        ) from exc
    return librosa


def _load_audio(path: str | Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    librosa = _require_librosa()
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    if sf is not None:
        try:
            waveform, sample_rate = sf.read(str(path))
            waveform = np.asarray(waveform, dtype=float)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if sample_rate != target_sample_rate:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate,
                )
                sample_rate = target_sample_rate
            return waveform, sample_rate
        except Exception:
            pass

    waveform, sample_rate = librosa.load(str(path), sr=target_sample_rate, mono=True)
    return np.asarray(waveform, dtype=float), sample_rate


def load_audio_sample(
    path: str | Path,
    target_sample_rate: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> AudioSample:
    librosa = _require_librosa()
    waveform, sample_rate = _load_audio(path, target_sample_rate=target_sample_rate)
    waveform = normalize_signal(np.asarray(waveform, dtype=float))
    if waveform.size == 0:
        raise ValueError("Audio file is empty.")

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    silence_threshold = np.quantile(rms, 0.25) * 0.8 if rms.size else 0.0
    voiced_mask = rms > silence_threshold if rms.size else np.array([], dtype=bool)
    pitch = librosa.yin(
        waveform,
        fmin=70,
        fmax=350,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return AudioSample(
        waveform=waveform,
        sample_rate=sample_rate,
        features={
            "rms": rms,
            "pitch": np.asarray(pitch, dtype=float),
            "voiced_mask": voiced_mask.astype(float),
            "duration_seconds": float(len(waveform) / max(sample_rate, 1)),
        },
    )
