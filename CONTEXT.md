# Answering-machine detection

Terms for multi-turn answering-machine detection (AMD). See
[design and usage](examples/telephony/amd.md) for runtime behavior.

## Language

**Accepted turn**: A participant turn committed by AgentSession at client-side end of turn (EOT).

**Prediction**: The category AMD assigns to an accepted turn.

**Stage**: The retained call category that constrains subsequent predictions and reply behavior.
_Avoid_: Last prediction, when referring to retained call state.

**Normal transition**: A stage change permitted without correcting an earlier classification.

**Stage correction**: An explicit prediction that replaces a mistaken machine stage for the current accepted turn and subsequent turns.

## Relationships

- An **Accepted turn** uses the transcript available at its EOT cutoff.
  AMD does not create independent turns or revise an authorized reply when a late transcript arrives.
- A **Prediction** of `wait` or `uncertain` keeps the current **Stage**.
- A classifier request contains the current **Stage**, the last accepted **Prediction**,
  and separate lists for normal predictions and explicit corrections.
- A **Stage correction** requires explicit intent and transcript evidence.
  It preserves earlier prediction events, delivered voicemail, and sent DTMF.
- A terminal **Prediction** of human or unavailable ends detection.
  Later evidence cannot reopen that run through a **Stage correction**.

## Example dialogue

> **Developer:** The previous prediction was `wait`. Can the next turn become screening?
> **Domain expert:** Check the retained stage. From voicemail, screening requires an explicit stage correction with transcript evidence.

## Flagged ambiguities

- "Previous label" can mean the last **Prediction** or the retained **Stage**.
  Requests include both because `wait` and `uncertain` do not replace the stage.
