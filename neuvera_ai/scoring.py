from __future__ import annotations

from dataclasses import dataclass

from .contracts import AnalysisResult, SessionAnalysisResult
from .signals import signal_details
from .utils import clamp01, flatten_signals, neutral_blend


@dataclass(slots=True)
class AggregationConfig:
    voice_weight: float = 0.34
    tremor_weight: float = 0.36
    movement_weight: float = 0.30
    positive_signal_bonus: float = 0.015


def _module_weight(module: AnalysisResult, base_weight: float) -> float:
    if module.status != "ok" or module.score is None:
        return 0.0
    return base_weight * max(0.15, module.confidence)


def aggregate_screening_results(
    voice: AnalysisResult | None = None,
    tremor: AnalysisResult | None = None,
    movement: AnalysisResult | None = None,
    config: AggregationConfig | None = None,
) -> SessionAnalysisResult:
    config = config or AggregationConfig()
    modules = [
        (voice, config.voice_weight),
        (tremor, config.tremor_weight),
        (movement, config.movement_weight),
    ]

    weighted_scores: list[tuple[float, float]] = []
    valid_confidences: list[float] = []
    statuses: list[str] = []
    available_modules = 0

    for module, base_weight in modules:
        if module is None:
            continue
        available_modules += 1
        statuses.append(module.status)
        weight = _module_weight(module, base_weight)
        if weight > 0.0:
            weighted_scores.append((neutral_blend(module.score or 0.5, module.confidence), weight))
            valid_confidences.append(module.confidence)

    all_signals = flatten_signals(
        voice.signals if voice else [],
        tremor.signals if tremor else [],
        movement.signals if movement else [],
    )

    if not weighted_scores:
        status = "failed" if available_modules == 0 else "insufficient_data"
        summary = (
            "No screening inputs were provided."
            if available_modules == 0
            else "Available inputs did not contain enough reliable data for an overall estimate."
        )
        return SessionAnalysisResult(
            status=status,
            overall_score=None,
            overall_confidence=0.0 if available_modules == 0 else 0.15,
            voice=voice,
            tremor=tremor,
            movement=movement,
            signals=all_signals,
            signal_details=signal_details(all_signals),
            summary=summary,
            metrics={
                "module_count": float(available_modules),
                "usable_module_count": 0.0,
                "signal_count": float(len(all_signals)),
            },
        )

    total_weight = sum(weight for _, weight in weighted_scores)
    base_score = sum(score * weight for score, weight in weighted_scores) / total_weight
    bonus = min(len([signal for signal in all_signals if signal != "poor_video_quality" and signal != "poor_audio_quality"]), 6) * config.positive_signal_bonus
    overall_score = clamp01(base_score + bonus)
    overall_confidence = sum(valid_confidences) / len(valid_confidences)

    if all(status == "ok" for status in statuses if status):
        status = "ok"
    elif any(status == "ok" for status in statuses):
        status = "partial"
    else:
        status = "insufficient_data"

    summary = (
        "Possible Parkinson's-related voice or movement warning signs were detected in this screening session. This is not a diagnosis."
        if any(signal not in {"poor_audio_quality", "poor_video_quality"} for signal in all_signals)
        else "No strong Parkinson's-related multimodal warning signs were detected in this screening session."
    )

    return SessionAnalysisResult(
        status=status,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        voice=voice,
        tremor=tremor,
        movement=movement,
        signals=all_signals,
        signal_details=signal_details(all_signals),
        summary=summary,
        metrics={
            "module_count": float(available_modules),
            "usable_module_count": float(len(weighted_scores)),
            "signal_count": float(len(all_signals)),
            "weighted_module_sum": total_weight,
        },
    )
