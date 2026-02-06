from __future__ import annotations

from opentelemetry.trace import Span

from livekit import rtc

from ..telemetry import trace_types


def _set_participant_attributes(span: Span, participant: rtc.Participant) -> None:
    span.set_attributes(
        {
            trace_types.ATTR_PARTICIPANT_ID: participant.sid,
            trace_types.ATTR_PARTICIPANT_IDENTITY: participant.identity,
            trace_types.ATTR_PARTICIPANT_KIND: rtc.ParticipantKind.Name(participant.kind),
        }
    )
