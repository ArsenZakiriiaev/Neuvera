# Neuvera AI Integration Guide

This document is for Dev 2 backend integration.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[audio,video,dev]
```

If you only need synthetic tests and score aggregation, `numpy` and `scipy` are enough.  
For real files:

- audio needs `librosa`
- video needs `opencv-python` and `mediapipe`

## Main Entry Points

Thin service layer for backend use:

```python
from neuvera_ai.service import (
    run_voice_analysis_from_file,
    run_tremor_analysis_from_video,
    run_movement_analysis_from_video,
    run_full_analysis,
)
```

Examples:

```python
voice = run_voice_analysis_from_file("sample.wav")
tremor = run_tremor_analysis_from_video("hand_test.mp4")
movement = run_movement_analysis_from_video("movement_test.mp4")
session = run_full_analysis(
    audio_path="sample.wav",
    hand_video_path="hand_test.mp4",
    movement_video_path="movement_test.mp4",
)
```

## Top-Level Response Schema

```json
{
  "status": "ok",
  "overall_score": 0.58,
  "overall_confidence": 0.73,
  "voice": {},
  "tremor": {},
  "movement": {},
  "signals": ["reduced_voice_variation", "hand_instability_detected"],
  "signal_details": [
    {
      "signal": "reduced_voice_variation",
      "label": "Reduced voice variation",
      "description": "The voice sample showed flatter pitch or intensity changes than expected."
    }
  ],
  "summary": "Possible Parkinson's-related voice or movement warning signs were detected in this screening session. This is not a diagnosis.",
  "disclaimer": "Neuvera is an early screening and monitoring support tool. It is designed to flag possible Parkinson's-related warning signs, not to diagnose Parkinson's disease or replace clinical evaluation.",
  "metrics": {
    "module_count": 3,
    "usable_module_count": 3
  }
}
```

## Per-Module Response Schema

Every module returns:

```json
{
  "module_name": "voice",
  "status": "ok",
  "score": 0.61,
  "confidence": 0.74,
  "signals": ["voice_instability_detected"],
  "signal_details": [],
  "metrics": {},
  "summary": "Possible Parkinson's-related voice changes were detected in this screening sample. This is not a diagnosis.",
  "disclaimer": "...",
  "error_code": null
}
```

## Status Values

- `ok`: module produced a usable result
- `partial`: session score exists, but one or more modules had insufficient data
- `insufficient_data`: file loaded or parsed, but signal quality was too weak
- `failed`: hard failure, for example missing file

## Supported Inputs

- Audio: formats readable by `librosa`, including `.wav`, `.mp3`, `.ogg`, `.flac`
- Video: formats readable by OpenCV, usually `.mp4`, `.mov`, `.avi`, `.webm`

## Minimum Recommended Capture Conditions

- Audio:
  - 3-6 seconds
  - close microphone
  - no strong background noise
  - audible speech or sustained vowel
- Hand video:
  - 5-8 seconds
  - one hand clearly visible
  - hand stays inside frame
  - stable lighting
- Movement video:
  - 5-10 seconds
  - upper body or full body visible
  - enough space for simple guided movement
  - camera mostly stationary

## Common Error Cases

- `file_not_found`
- `audio_processing_error`
- `hand_video_processing_error`
- `movement_video_processing_error`
- `poor_audio_quality`
- `poor_hand_tracking`
- `poor_pose_tracking`

These errors are returned as structured JSON. Backend should pass them through instead of converting them to 500 errors unless the file transport itself failed.
