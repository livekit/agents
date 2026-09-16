from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.google.beta.gemini_stt import STT

pytestmark = pytest.mark.plugin("google")


class _Transcription:
    def __init__(
        self,
        text: str | None,
        *,
        finished: bool | None = None,
        language_code: str | None = None,
    ) -> None:
        self.text = text
        self.finished = finished
        self.language_code = language_code


class _ServerContent:
    def __init__(
        self,
        *,
        input_transcription: _Transcription | None = None,
        interim_input_transcription: _Transcription | None = None,
        turn_complete: bool | None = None,
        generation_complete: bool | None = None,
    ) -> None:
        self.input_transcription = input_transcription
        self.interim_input_transcription = interim_input_transcription
        self.turn_complete = turn_complete
        self.generation_complete = generation_complete


class _Message:
    def __init__(self, server_content: _ServerContent) -> None:
        self.server_content = server_content


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


async def _drain(messages: list[_Message]) -> list[stt.SpeechEvent]:
    """Run a stream against a scripted set of Live API messages."""
    with patch("livekit.plugins.google.beta.gemini_stt.Client") as client_cls:
        mock_client = MagicMock()
        client_cls.return_value = mock_client

        mock_session = AsyncMock()
        mock_client.aio.live.connect.return_value.__aenter__.return_value = mock_session

        async def receive():  # type: ignore[no-untyped-def]
            for message in messages:
                yield message

        mock_session.receive = MagicMock(side_effect=receive)

        google_stt = STT(api_key="test-key")
        stream = google_stt.stream()
        stream.push_frame(_frame())
        stream.end_input()

        return [event async for event in stream]


def _texts(events: list[stt.SpeechEvent], event_type: stt.SpeechEventType) -> list[str]:
    return [e.alternatives[0].text for e in events if e.type == event_type]


@pytest.mark.asyncio
async def test_finalized_transcript_is_emitted_on_arrival() -> None:
    """The message shapes captured from a live gemini-3.5-transcribe-live session.

    Both streams carry a complete hypothesis rather than deltas: the interims are
    resent in full while the user speaks, and `input_transcription` is the finalized,
    authoritative transcript for the turn (re-transcribed, so its wording can differ
    from the interims). It is the final -- no accumulation, and no waiting on a
    completion flag.
    """
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("Greetings."))),
            _Message(
                _ServerContent(interim_input_transcription=_Transcription("Greetings, welcome"))
            ),
            _Message(
                _ServerContent(
                    input_transcription=_Transcription("Greetings, welcome to the age of AI.")
                )
            ),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == [
        "Greetings, welcome to the age of AI."
    ]
    # the finalized transcript must not also surface as an interim
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "Greetings.",
        "Greetings, welcome",
    ]


@pytest.mark.asyncio
async def test_each_turn_finalizes_separately() -> None:
    """A session yields one `input_transcription` per speech turn.

    Captured live by splitting the sample audio with 3s of silence: two turns, two
    complete transcripts. Accumulating them would concatenate separate utterances.
    """
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription("First utterance."))),
            _Message(_ServerContent(generation_complete=True)),
            _Message(_ServerContent(input_transcription=_Transcription("Second utterance."))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == [
        "First utterance.",
        "Second utterance.",
    ]


@pytest.mark.asyncio
async def test_back_to_back_finals_without_completion_between_them() -> None:
    """Two finalized transcripts must stay separate even with no completion signal."""
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription("First utterance."))),
            _Message(_ServerContent(input_transcription=_Transcription("Second utterance."))),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == [
        "First utterance.",
        "Second utterance.",
    ]


@pytest.mark.asyncio
async def test_cumulative_interims_are_not_concatenated() -> None:
    """Each interim is a full snapshot, so it replaces the previous one."""
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("I am"))),
            _Message(_ServerContent(interim_input_transcription=_Transcription("I am a human"))),
        ]
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["I am", "I am a human"]


@pytest.mark.asyncio
async def test_interim_only_turn_still_commits() -> None:
    """If no finalized transcript arrives, the turn must not be dropped."""
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("hello there"))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello there"]


@pytest.mark.asyncio
async def test_completion_after_a_final_does_not_duplicate_it() -> None:
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("hello"))),
            _Message(_ServerContent(input_transcription=_Transcription("Hello."))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["Hello."]


@pytest.mark.asyncio
async def test_completion_without_any_transcript_emits_nothing() -> None:
    """A bare generation_complete must not emit an empty final."""
    events = await _drain([_Message(_ServerContent(generation_complete=True))])

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == []


@pytest.mark.asyncio
async def test_detected_language_is_reported() -> None:
    events = await _drain(
        [
            _Message(
                _ServerContent(input_transcription=_Transcription("bonjour", language_code="fr-FR"))
            ),
        ]
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "fr-FR"


@pytest.mark.asyncio
async def test_configured_language_is_used_when_none_is_detected() -> None:
    events = await _drain([_Message(_ServerContent(input_transcription=_Transcription("hi")))])

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[0].alternatives[0].language == "en-US"


# The message text Gemini closes a duration-capped session with, from a live session log.
_GOAWAY = (
    "received 1008 (policy violation) Connection aborted because the client failed to "
    "close the connection after receiving a GoAway signal once the session duration "
    "limit was reached"
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Exception(_GOAWAY), True),
        (Exception("1008 None. ... after receiving a GoAway signal once the session durat"), True),
        (
            Exception("1007 The requested combination of response modalities is not supported"),
            False,
        ),
        (Exception("429 RESOURCE_EXHAUSTED"), False),
    ],
)
def test_session_duration_close_is_classified(error: Exception, expected: bool) -> None:
    """A duration rollover is routine -- the retry layer reconnects -- so it must not be
    logged as a failure every ten minutes."""
    from livekit.plugins.google.beta.gemini_stt import _is_session_duration_close

    assert _is_session_duration_close(error) is expected
