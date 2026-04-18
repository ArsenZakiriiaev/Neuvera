from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import AnalysisResult
from .quality import assess_audio_quality
from .signals import signal_details
from .utils import clamp01, confidence_from_coverage, normalize_signal, safe_ratio


@dataclass(slots=True)
class VoiceAnalysisConfig:
    frame_length: int = 2048
    hop_length: int = 512
    silence_quantile: float = 0.25
    target_duration_seconds: float = 6.0


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "voice_analysis requires librosa. Install with `pip install neuvera-ai[audio]`."
        ) from exc
    return librosa


def analyze_voice_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    config: VoiceAnalysisConfig | None = None,
) -> AnalysisResult:
    librosa = _require_librosa()
    config = config or VoiceAnalysisConfig()

    waveform = normalize_signal(np.asarray(waveform, dtype=float))
    duration_seconds = len(waveform) / max(sample_rate, 1)
    if waveform.size == 0 or duration_seconds < 1.0:
        return AnalysisResult.insufficient_data(
            module_name="voice",
            summary="Voice sample was too short for a stable screening estimate.",
            error_code="insufficient_voice_sample",
            metrics={"duration_seconds": duration_seconds},
        )

    rms = librosa.feature.rms(
        y=waveform, frame_length=config.frame_length, hop_length=config.hop_length
    )[0]
    if rms.size == 0:
        return AnalysisResult.insufficient_data(
            module_name="voice",
            summary="Voice sample did not contain enough usable energy for analysis.",
            error_code="insufficient_voice_energy",
            metrics={"duration_seconds": duration_seconds},
        )

    silence_threshold = np.quantile(rms, config.silence_quantile) * 0.8
    voiced_mask = rms > silence_threshold
    quality = assess_audio_quality(
        waveform=waveform,
        sample_rate=sample_rate,
        rms=rms,
        voiced_mask=voiced_mask,
    )
    if not quality.is_usable:
        return AnalysisResult.insufficient_data(
            module_name="voice",
            summary="Audio quality was too weak for a reliable voice screening estimate.",
            error_code="poor_audio_quality",
            metrics=quality.metrics,
            confidence=max(0.05, quality.quality_score * 0.35),
            signals=["poor_audio_quality", "low_voice_energy"] if "low_audio_energy" in quality.reasons else ["poor_audio_quality"],
            signal_details=signal_details(
                ["poor_audio_quality", "low_voice_energy"]
                if "low_audio_energy" in quality.reasons
                else ["poor_audio_quality"]
            ),
        )

    pause_ratio = 1.0 - float(np.mean(voiced_mask))

    f0 = librosa.yin(
        waveform,
        fmin=70,
        fmax=350,
        sr=sample_rate,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )
    f0 = np.asarray(f0, dtype=float)
    valid_pitch_mask = np.isfinite(f0) & (f0 > 0)
    valid_pitch = f0[valid_pitch_mask]

    pitch_median = float(np.median(valid_pitch)) if valid_pitch.size else 0.0
    pitch_variation = safe_ratio(float(np.std(valid_pitch)), pitch_median)
    # Jitter: cycle-to-cycle relative change in f0 (voiced frames only).
    if valid_pitch.size > 2:
        jitter = float(np.mean(np.abs(np.diff(valid_pitch))) / max(np.mean(valid_pitch), 1e-6))
    else:
        jitter = 0.0
    pitch_instability = safe_ratio(
        float(np.std(np.diff(valid_pitch))) if valid_pitch.size > 1 else 0.0,
        pitch_median,
    )
    # Shimmer: cycle-to-cycle relative change in amplitude on voiced frames.
    voiced_rms = rms[voiced_mask] if np.any(voiced_mask) else rms
    if voiced_rms.size > 2:
        shimmer = float(np.mean(np.abs(np.diff(voiced_rms))) / max(np.mean(voiced_rms), 1e-6))
    else:
        shimmer = 0.0
    energy_variation = safe_ratio(float(np.std(rms)), float(np.mean(rms)), default=0.0)
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform, frame_length=config.frame_length, hop_length=config.hop_length
    )[0]
    zcr_mean = float(np.mean(zcr))
    # Harmonics-to-noise ratio proxy via spectral flatness (higher flatness = noisier voice).
    try:
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(
            y=waveform, n_fft=config.frame_length, hop_length=config.hop_length,
        )[0]))
    except Exception:
        spectral_flatness = 0.0

    reduced_variation_risk = clamp01((0.15 - pitch_variation) / 0.10)
    instability_risk = clamp01((pitch_instability - 0.03) / 0.045)
    jitter_risk = clamp01((jitter - 0.012) / 0.022)
    shimmer_risk = clamp01((shimmer - 0.06) / 0.10)
    pause_risk = clamp01((pause_ratio - 0.30) / 0.30)
    monotone_energy_risk = clamp01((0.30 - energy_variation) / 0.18)
    harshness_risk = clamp01((zcr_mean - 0.14) / 0.12)
    breathiness_risk = clamp01((spectral_flatness - 0.25) / 0.25)

    score = float(
        0.22 * jitter_risk
        + 0.20 * shimmer_risk
        + 0.16 * instability_risk
        + 0.14 * reduced_variation_risk
        + 0.12 * monotone_energy_risk
        + 0.08 * pause_risk
        + 0.05 * breathiness_risk
        + 0.03 * harshness_risk
    )

    signals: list[str] = []
    if jitter_risk > 0.55 or instability_risk > 0.55:
        signals.append("voice_instability_detected")
    if shimmer_risk > 0.55:
        signals.append("amplitude_instability_detected")
    if reduced_variation_risk > 0.55:
        signals.append("reduced_voice_variation")
    if pause_risk > 0.6:
        signals.append("unusual_pause_pattern")
    if breathiness_risk > 0.6:
        signals.append("breathy_voice_quality")
    if harshness_risk > 0.65:
        signals.append("noisy_voice_capture")

    pitch_coverage = float(np.mean(valid_pitch_mask)) if f0.size else 0.0
    confidence = confidence_from_coverage(
        coverage=(0.55 * pitch_coverage + 0.25 * quality.metrics["voiced_ratio"] + 0.20 * quality.quality_score),
        duration_seconds=duration_seconds,
        target_duration=config.target_duration_seconds,
    )

    summary = (
        "Possible Parkinson's-related voice changes were detected in this screening sample. This is not a diagnosis."
        if signals
        else "No strong Parkinson's-related voice warning signs were detected in this screening sample."
    )

    metrics = {
        "duration_seconds": duration_seconds,
        "voiced_ratio": quality.metrics["voiced_ratio"],
        "pause_ratio": pause_ratio,
        "pitch_variation": pitch_variation,
        "pitch_instability": pitch_instability,
        "jitter": jitter,
        "shimmer": shimmer,
        "energy_variation": energy_variation,
        "zcr_mean": zcr_mean,
        "spectral_flatness": spectral_flatness,
        "audio_quality_score": quality.quality_score,
        "peak_amplitude": quality.metrics["peak_amplitude"],
        "rms_energy": quality.metrics["rms_energy"],
    }
    return AnalysisResult.ok(
        module_name="voice",
        score=score,
        confidence=confidence,
        signals=signals,
        signal_details=signal_details(signals),
        metrics=metrics,
        summary=summary,
    )
