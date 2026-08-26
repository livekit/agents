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
async def test_transcribe_live_turn_commits_on_generation_complete() -> None:
    """The gemini-3.5-transcribe-live shape, captured from a live session.

    `interim_input_transcription` resends the whole turn each update, the authoritative
    `input_transcription` lands once at the end, and neither `finished` nor
    `turn_complete` is ever populated -- `generation_complete` is the only signal that
    the turn is done. Gating the final on `finished`/`turn_complete` emitted no final
    at all, so no turn was ever committed.
    """
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("Greetings."))),
            _Message(
                _ServerContent(interim_input_transcription=_Transcription("Greetings, welcome"))
            ),
            _Message(
                _ServerContent(
                    interim_input_transcription=_Transcription("Greetings, welcome to the age")
                )
            ),
            _Message(
                _ServerContent(
                    input_transcription=_Transcription("Greetings, welcome to the age of AI.")
                )
            ),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    # the authoritative transcript wins, not the accumulated interim text
    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == [
        "Greetings, welcome to the age of AI."
    ]
    # `input_transcription` must not add an interim of its own: the interim stream
    # already covers live text, so it would just duplicate the final
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "Greetings.",
        "Greetings, welcome",
        "Greetings, welcome to the age",
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
async def test_delta_input_transcription_accumulates() -> None:
    """The older live models stream `input_transcription` in deltas; both shapes must work."""
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hello"))),
            _Message(_ServerContent(input_transcription=_Transcription(" world"))),
            _Message(_ServerContent(turn_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello world"]
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello", "hello world"]


@pytest.mark.asyncio
async def test_finished_flag_finalizes_when_a_model_sets_it() -> None:
    events = await _drain(
        [
            _Message(
                _ServerContent(input_transcription=_Transcription(" hello", finished=True)),
            ),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello"]


@pytest.mark.asyncio
async def test_interim_only_turn_still_commits() -> None:
    """If a model never sends `input_transcription`, the turn must not be dropped."""
    events = await _drain(
        [
            _Message(_ServerContent(interim_input_transcription=_Transcription("hello there"))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello there"]


@pytest.mark.asyncio
async def test_consecutive_turns_do_not_leak_into_each_other() -> None:
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" first"))),
            _Message(_ServerContent(generation_complete=True)),
            _Message(_ServerContent(input_transcription=_Transcription(" second"))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["first", "second"]


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
                _ServerContent(
                    input_transcription=_Transcription(" bonjour", language_code="fr-FR")
                )
            ),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "fr-FR"


@pytest.mark.asyncio
async def test_configured_language_is_used_when_none_is_detected() -> None:
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hi"))),
            _Message(_ServerContent(generation_complete=True)),
        ]
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[0].alternatives[0].language == "en-US"
