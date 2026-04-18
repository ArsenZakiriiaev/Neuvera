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
    pitch_instability = safe_ratio(
        float(np.std(np.diff(valid_pitch))) if valid_pitch.size > 1 else 0.0,
        pitch_median,
    )
    energy_variation = safe_ratio(float(np.std(rms)), float(np.mean(rms)), default=0.0)
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform, frame_length=config.frame_length, hop_length=config.hop_length
    )[0]
    zcr_mean = float(np.mean(zcr))

    reduced_variation_risk = clamp01((0.18 - pitch_variation) / 0.12)
    instability_risk = clamp01((pitch_instability - 0.035) / 0.05)
    pause_risk = clamp01((pause_ratio - 0.25) / 0.35)
    monotone_energy_risk = clamp01((0.32 - energy_variation) / 0.18)
    harshness_risk = clamp01((zcr_mean - 0.14) / 0.12)

    score = float(
        0.32 * instability_risk
        + 0.25 * reduced_variation_risk
        + 0.18 * pause_risk
        + 0.15 * monotone_energy_risk
        + 0.10 * harshness_risk
    )

    signals: list[str] = []
    if instability_risk > 0.55:
        signals.append("voice_instability_detected")
    if reduced_variation_risk > 0.55:
        signals.append("reduced_voice_variation")
    if pause_risk > 0.6:
        signals.append("unusual_pause_pattern")
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
        "energy_variation": energy_variation,
        "zcr_mean": zcr_mean,
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
