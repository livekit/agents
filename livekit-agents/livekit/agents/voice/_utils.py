from __future__ import annotations

from opentelemetry.trace import Span

from livekit import rtc

from ..telemetry import utils as telemetry_utils


def _set_participant_attributes(span: Span, participant: rtc.Participant) -> None:
    span.set_attributes(telemetry_utils.participant_attributes(participant))
