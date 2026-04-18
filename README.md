# Neuvera

Neuvera is a browser-based early screening MVP for possible Parkinson's-related warning signs.

This repository contains the Dev 1 analysis layer:

- raw input adapters for audio and video
- heuristic voice, tremor, and movement analysis
- quality-aware score aggregation
- backend-facing service wrappers

## Structure

```text
neuvera_ai/
  audio_adapter.py
  hand_video_adapter.py
  pose_video_adapter.py
  voice_analysis.py
  tremor_analysis.py
  movement_analysis.py
  scoring.py
  pipeline.py
  service.py
  quality.py
  signals.py
  contracts.py
  demo_fixtures.py
tests/
INTEGRATION.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[audio,video,dev]
```

## Main Usage

```python
from neuvera_ai.service import run_full_analysis

result = run_full_analysis(
    audio_path="sample.wav",
    hand_video_path="hand_test.mp4",
    movement_video_path="movement_test.mp4",
)
print(result)
```

The end-to-end pipeline:

1. loads raw audio/video files
2. extracts features or landmarks
3. runs module-level screening heuristics
4. validates signal quality
5. returns one JSON-ready backend response

Medical/product wording is intentionally grounded in official symptom references from NINDS/NIH, Parkinson's Foundation, and NHS. See [MEDICAL_GROUNDED_SIGNAL_MAP.md](/home/sonny/MEDICAL_GROUNDED_SIGNAL_MAP.md).

## Response Shape

Top level:

```json
{
  "status": "ok",
  "overall_score": 0.57,
  "overall_confidence": 0.72,
  "voice": {},
  "tremor": {},
  "movement": {},
  "signals": ["reduced_voice_variation", "hand_instability_detected"],
  "summary": "Possible Parkinson's-related voice or movement warning signs were detected in this screening session. This is not a diagnosis.",
  "disclaimer": "Neuvera is an early screening and monitoring support tool. It is designed to flag possible Parkinson's-related warning signs, not to diagnose Parkinson's disease or replace clinical evaluation."
}
```

Per module:

```json
{
  "module_name": "voice",
  "status": "ok",
  "score": 0.63,
  "confidence": 0.71,
  "signals": ["voice_instability_detected"],
  "signal_details": [],
  "metrics": {},
  "summary": "Possible Parkinson's-related voice changes were detected in this screening sample. This is not a diagnosis.",
  "error_code": null
}
```

## Demo Safety

The pipeline returns structured `insufficient_data` or `failed` responses for:

- empty or missing files
- short audio/video samples
- poor audio energy
- weak hand or pose landmark detection
- processing errors from adapters

## Product Framing

Neuvera is not a diagnostic or clinical system. It is an early screening and monitoring support MVP intended for demo use and technical prototyping.
