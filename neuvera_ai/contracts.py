from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "Neuvera is an early screening and monitoring support tool. "
    "It is designed to flag possible Parkinson's-related warning signs, "
    "not to diagnose Parkinson's disease or replace clinical evaluation."
)


@dataclass(slots=True)
class SignalDetail:
    signal: str
    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    """Stable backend-facing contract for one analysis module."""

    module_name: str
    status: str
    score: float | None
    confidence: float
    signals: list[str] = field(default_factory=list)
    signal_details: list[SignalDetail] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    disclaimer: str = DISCLAIMER
    error_code: str | None = None

    @classmethod
    def ok(
        cls,
        module_name: str,
        score: float,
        confidence: float,
        signals: list[str] | None = None,
        signal_details: list[SignalDetail] | None = None,
        metrics: dict[str, float] | None = None,
        summary: str = "",
    ) -> "AnalysisResult":
        return cls(
            module_name=module_name,
            status="ok",
            score=score,
            confidence=confidence,
            signals=signals or [],
            signal_details=signal_details or [],
            metrics=metrics or {},
            summary=summary,
        )

    @classmethod
    def insufficient_data(
        cls,
        module_name: str,
        summary: str,
        error_code: str,
        metrics: dict[str, float] | None = None,
        confidence: float = 0.15,
        signals: list[str] | None = None,
        signal_details: list[SignalDetail] | None = None,
    ) -> "AnalysisResult":
        return cls(
            module_name=module_name,
            status="insufficient_data",
            score=None,
            confidence=confidence,
            signals=signals or [],
            signal_details=signal_details or [],
            metrics=metrics or {},
            summary=summary,
            error_code=error_code,
        )

    @classmethod
    def failed(
        cls,
        module_name: str,
        summary: str,
        error_code: str,
        metrics: dict[str, float] | None = None,
    ) -> "AnalysisResult":
        return cls(
            module_name=module_name,
            status="failed",
            score=None,
            confidence=0.0,
            metrics=metrics or {},
            summary=summary,
            error_code=error_code,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["score"] is not None:
            payload["score"] = round(float(payload["score"]), 4)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["metrics"] = {
            key: round(float(value), 4) for key, value in self.metrics.items()
        }
        payload["signal_details"] = [
            detail.to_dict() if hasattr(detail, "to_dict") else dict(detail)
            for detail in self.signal_details
        ]
        return payload


@dataclass(slots=True)
class SessionAnalysisResult:
    status: str
    overall_score: float | None
    overall_confidence: float
    voice: AnalysisResult | None = None
    tremor: AnalysisResult | None = None
    movement: AnalysisResult | None = None
    signals: list[str] = field(default_factory=list)
    signal_details: list[SignalDetail] = field(default_factory=list)
    summary: str = ""
    disclaimer: str = DISCLAIMER
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": None if self.overall_score is None else round(float(self.overall_score), 4),
            "overall_confidence": round(float(self.overall_confidence), 4),
            "voice": None if self.voice is None else self.voice.to_dict(),
            "tremor": None if self.tremor is None else self.tremor.to_dict(),
            "movement": None if self.movement is None else self.movement.to_dict(),
            "signals": list(self.signals),
            "signal_details": [detail.to_dict() for detail in self.signal_details],
            "summary": self.summary,
            "disclaimer": self.disclaimer,
            "metrics": {
                key: round(float(value), 4) for key, value in self.metrics.items()
            },
        }
