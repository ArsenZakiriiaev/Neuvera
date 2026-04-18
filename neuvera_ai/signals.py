from __future__ import annotations

from .contracts import SignalDetail

SIGNAL_CATALOG: dict[str, tuple[str, str]] = {
    "reduced_voice_variation": (
        "Reduced voice variation",
        "The voice sample showed flatter pitch or loudness changes than expected for the task.",
    ),
    "voice_instability_detected": (
        "Voice instability detected",
        "The voice sample showed short-term instability that can be useful as a screening signal.",
    ),
    "unusual_pause_pattern": (
        "Unusual pause pattern",
        "The recording contained more or longer pauses than expected for the task.",
    ),
    "noisy_voice_capture": (
        "Noisy voice capture",
        "The voice sample had a noisy or harsh profile that can reduce analysis quality.",
    ),
    "low_voice_energy": (
        "Low voice energy",
        "The recording had limited usable voice energy for confident screening.",
    ),
    "hand_instability_detected": (
        "Hand instability detected",
        "Small repeated hand movements were observed across the tracked frames.",
    ),
    "micro_oscillation_detected": (
        "Micro-oscillation detected",
        "The hand trajectory showed fine oscillatory movement over time.",
    ),
    "tremor_band_activity": (
        "Tremor-band activity",
        "A strong portion of movement energy fell inside the configured tremor-like frequency band.",
    ),
    "consistent_high_frequency_motion": (
        "Consistent high-frequency motion",
        "Repeated quick oscillations were present across the hand trajectory.",
    ),
    "low_motion_amplitude": (
        "Low motion amplitude",
        "Overall body movement was lower than expected for the guided task.",
    ),
    "low_arm_mobility": (
        "Low arm mobility",
        "Arm movement range appeared reduced during the pose sequence.",
    ),
    "movement_asymmetry": (
        "Movement asymmetry",
        "Left and right side movement ranges were noticeably uneven.",
    ),
    "posture_irregularity": (
        "Posture irregularity",
        "Body alignment changed in a way that suggests posture-related motor irregularity.",
    ),
    "stooped_posture": (
        "Stooped posture",
        "The head sat lower relative to the shoulders than expected for an upright stance.",
    ),
    "head_tilt": (
        "Lateral head tilt",
        "The head was tilted sideways by a larger angle than expected for an upright stance.",
    ),
    "head_tremor": (
        "Head tremor",
        "Small rhythmic head movements were observed across the tracked frames.",
    ),
    "amplitude_instability_detected": (
        "Amplitude instability (shimmer)",
        "Cycle-to-cycle loudness fluctuations were higher than expected for a steady reading voice.",
    ),
    "breathy_voice_quality": (
        "Breathy / noisy voice quality",
        "The voice spectrum was flatter / noisier than a clear phonation, a Parkinson's-related voice sign.",
    ),
    "poor_audio_quality": (
        "Poor audio quality",
        "The recording quality was too weak for a stable voice estimate.",
    ),
    "poor_video_quality": (
        "Poor video quality",
        "The video quality or detection coverage was too weak for a stable estimate.",
    ),
}


def signal_details(signals: list[str]) -> list[SignalDetail]:
    details: list[SignalDetail] = []
    for signal_name in signals:
        label, description = SIGNAL_CATALOG.get(
            signal_name,
            (signal_name.replace("_", " ").title(), "A screening irregularity signal was detected."),
        )
        details.append(
            SignalDetail(signal=signal_name, label=label, description=description)
        )
    return details
