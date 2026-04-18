# Neuvera Medical-Grounded Signal Map

This document defines which product-facing signals Neuvera may use and how they are grounded in authoritative Parkinson's references.

It is intentionally conservative:

- use symptom language supported by official sources
- frame outputs as screening indicators, not diagnosis
- avoid claiming that any single computed feature identifies Parkinson's disease

## Source Base

- NINDS / NIH: Parkinson's disease overview and symptom descriptions
  https://www.ninds.nih.gov/health-information/disorders/parkinsons-disease
- Parkinson's Foundation: What is Parkinson's?
  https://www.parkinson.org/understanding-parkinsons/what-is-parkinsons
- Parkinson's Foundation: Movement symptoms
  https://www.parkinson.org/understanding-parkinsons/movement-symptoms
- Parkinson's Foundation: Trouble moving or walking
  https://www.parkinson.org/understanding-parkinsons/movement-symptoms/trouble-moving
- Parkinson's Foundation: Speech & Swallowing
  https://www.parkinson.org/library/fact-sheets/speech-swallowing
- NHS: Parkinson's disease
  https://www.nhs.uk/conditions/parkinsons-disease/

## Product Framing Rules

- Allowed:
  - "possible Parkinson's-related warning signs"
  - "screening estimate"
  - "movement irregularities"
  - "voice changes associated with Parkinson's-related symptoms"
  - "not a diagnosis"
- Not allowed:
  - "diagnosed Parkinson's"
  - "detected Parkinson's disease"
  - "medical-grade assessment"
  - "clinically validated diagnosis"

## Officially Supported Symptom Areas

The following symptom areas are directly supported by official sources:

- Tremor / shaking
- Slowness of movement / bradykinesia
- Rigidity / stiffness
- Postural instability / balance and posture changes
- Reduced arm swing
- Gait changes such as smaller steps, shuffling, slower pace, trouble turning
- Speech changes such as soft speech, monotone voice, slurred or unclear speech, fast or slowed speech, hesitation

## Allowed Neuvera Signal Mapping

### Voice

`reduced_voice_variation`
- Backend meaning:
  - lower pitch variation and/or lower energy variation than expected
- Product text:
  - "Reduced voice variation was observed in the sample."
- Grounding:
  - Parkinson's Foundation and NINDS both describe monotone or soft speech as Parkinson's-related speech changes.

`voice_instability_detected`
- Backend meaning:
  - short-term pitch instability or irregularity
- Product text:
  - "Voice instability was observed in the sample."
- Grounding:
  - Parkinson's Foundation describes breathy, hoarse, reduced-clarity and motor-related speech changes. This signal should stay framed as a screening heuristic, not a clinical label.

`unusual_pause_pattern`
- Backend meaning:
  - lower voiced coverage / more hesitation or pauses than expected
- Product text:
  - "The voice sample included more pauses or hesitation than expected."
- Grounding:
  - NINDS notes that some people may hesitate before speaking; Parkinson's Foundation also describes slowed or disrupted communication.

`noisy_voice_capture`
- Backend meaning:
  - harsh/noisy recording profile
- Product text:
  - "Recording quality may have reduced the reliability of the voice estimate."
- Grounding:
  - This is a technical quality flag, not a Parkinson's symptom claim.

### Tremor

`hand_instability_detected`
- Backend meaning:
  - repeated small-amplitude hand movement
- Product text:
  - "Hand instability was observed during the screening task."
- Grounding:
  - NINDS and Parkinson's Foundation describe tremor as a key movement symptom. We keep this label broad and screening-oriented.

`micro_oscillation_detected`
- Backend meaning:
  - repeated oscillatory hand motion in tracked landmarks
- Product text:
  - "Fine oscillatory hand movement was observed."
- Grounding:
  - This is an engineering description of tremor-like motion, not a clinical diagnosis label.

`tremor_band_activity`
- Backend meaning:
  - strong movement energy inside the configured tremor-like frequency band
- Product text:
  - "Tremor-like rhythmic hand movement was observed."
- Grounding:
  - Tremor is an official symptom area; frequency-band wording is our implementation heuristic.

### Movement / Posture

`low_motion_amplitude`
- Backend meaning:
  - lower-than-expected movement range
- Product text:
  - "Overall movement amplitude appeared reduced during the task."
- Grounding:
  - Parkinson's Foundation describes bradykinesia, hypokinesia and smaller movements.

`low_arm_mobility`
- Backend meaning:
  - reduced arm movement range
- Product text:
  - "Arm movement appeared reduced during the task."
- Grounding:
  - Parkinson's Foundation explicitly mentions reduced arm swing and stiffness-related reduced range of motion.

`movement_asymmetry`
- Backend meaning:
  - uneven left-vs-right movement range
- Product text:
  - "Movement appeared uneven between the left and right sides."
- Grounding:
  - NINDS notes symptoms often begin on one side and remain more severe on one side.

`posture_irregularity`
- Backend meaning:
  - posture or alignment changes during task performance
- Product text:
  - "Posture changes associated with motor irregularity were observed."
- Grounding:
  - NINDS and Parkinson's Foundation both describe posture changes and postural instability as relevant symptom areas.

## Implementation Notes

- A Neuvera signal is not a diagnosis and not a direct clinical finding.
- A Neuvera signal is a product-facing interpretation of measurable audio/video patterns chosen to align with official symptom descriptions.
- If a signal depends heavily on recording quality, degrade confidence and prefer a quality warning over a symptom claim.

## Recommended Top-Level Summary

Use:

`Possible Parkinson's-related voice or movement warning signs were detected in this screening session. This is not a diagnosis.`

Avoid:

`Parkinson's detected`
