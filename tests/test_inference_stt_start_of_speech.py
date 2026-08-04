"""`inference.STT` must report speech onset when the provider detects it server-side.

Cartesia Ink-2 signals onset with `turn.start`, which the gateway sends on as
`start_of_speech`. That used to arrive as an `interim_transcript` with an empty string,
which `_process_transcript` drops, so onset fell back to the first transcript carrying
words — around 800ms late against the same audio through the Cartesia plugin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents.inference.stt import SpeechStream as InferenceSpeechStream
from livekit.agents.stt import SpeechEventType
from livekit.agents.utils.aio.channel import Chan, ChanEmpty

pytestmark = [pytest.mark.unit]


def _stream() -> InferenceSpeechStream:
    stream = InferenceSpeechStream.__new__(InferenceSpeechStream)
    stream._event_ch = Chan()  # type: ignore[assignment]
    stream._speaking = False
    stream._request_id = "req-1"
    stream._speech_duration = 0.0
    stream._start_time_offset = 0.0
    stream._pending_extra = None
    stream._opts = SimpleNamespace(language="en")
    return stream


def _drain(stream: InferenceSpeechStream) -> list[SpeechEventType]:
    out = []
    while True:
        try:
            out.append(stream._event_ch.recv_nowait().type)
        except ChanEmpty:
            return out


def test_start_of_speech_message_reports_onset_immediately() -> None:
    """The onset event must not wait for a transcript."""
    stream = _stream()

    stream._process_start_of_speech()

    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]
    assert stream._speaking is True


def test_onset_is_not_reported_twice() -> None:
    """A later transcript must not re-announce onset the provider already gave us."""
    stream = _stream()

    stream._process_start_of_speech()
    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]

    stream._process_transcript({"transcript": "are you"}, is_final=False)

    # only the interim, no second onset
    assert _drain(stream) == [SpeechEventType.INTERIM_TRANSCRIPT]


def test_duplicate_start_of_speech_is_ignored() -> None:
    stream = _stream()

    stream._process_start_of_speech()
    stream._process_start_of_speech()

    assert _drain(stream) == [SpeechEventType.START_OF_SPEECH]


def test_providers_without_onset_still_fall_back_to_the_first_transcript() -> None:
    """Models that send no onset event keep today's behaviour."""
    stream = _stream()

    stream._process_transcript({"transcript": "are you"}, is_final=False)

    assert _drain(stream) == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
    ]


def test_empty_interim_alone_never_reports_onset() -> None:
    """The shape the gateway sends today: an empty interim carries no onset.

    This is the bug. The gateway must send `start_of_speech` instead; the SDK cannot
    infer onset from an empty transcript without firing for providers that emit empty
    interims for unrelated reasons.
    """
    stream = _stream()

    stream._process_transcript({"transcript": ""}, is_final=False)

    assert _drain(stream) == []
    assert stream._speaking is False


def test_onset_resets_after_the_turn_ends() -> None:
    """A second turn on the same stream must report its own onset."""
    stream = _stream()

    stream._process_start_of_speech()
    stream._process_transcript(
        {"transcript": "are you open on sunday", "start": 0.0, "duration": 1.0},
        is_final=True,
    )
    assert stream._speaking is False

    stream._process_start_of_speech()
    assert SpeechEventType.START_OF_SPEECH in _drain(stream)
