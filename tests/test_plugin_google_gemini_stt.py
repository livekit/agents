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
        finished: bool = False,
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
        turn_complete: bool = False,
    ) -> None:
        self.input_transcription = input_transcription
        self.interim_input_transcription = interim_input_transcription
        self.turn_complete = turn_complete


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
async def test_transcription_deltas_accumulate_into_one_final() -> None:
    """`input_transcription` streams deltas, so one utterance must yield one final.

    Emitting each delta as its own FINAL_TRANSCRIPT would split a single utterance
    into several user turns downstream.
    """
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hello"))),
            _Message(_ServerContent(input_transcription=_Transcription(" world"))),
            _Message(_ServerContent(input_transcription=_Transcription("!", finished=True))),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello world!"]
    # every delta still surfaces as a growing interim
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "hello",
        "hello world",
        "hello world!",
    ]


@pytest.mark.asyncio
async def test_turn_complete_finalizes_when_finished_is_absent() -> None:
    """`finished` isn't guaranteed on every turn; turn_complete must close it out."""
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hello world"))),
            _Message(_ServerContent(turn_complete=True)),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["hello world"]


@pytest.mark.asyncio
async def test_consecutive_turns_do_not_leak_into_each_other() -> None:
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" first", finished=True))),
            _Message(_ServerContent(input_transcription=_Transcription(" second", finished=True))),
        ]
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["first", "second"]


@pytest.mark.asyncio
async def test_interim_transcription_extends_committed_text() -> None:
    """The low-latency preview covers the tail that hasn't been committed yet."""
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hello"))),
            _Message(_ServerContent(interim_input_transcription=_Transcription(" wor"))),
        ]
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello", "hello wor"]
    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == []


@pytest.mark.asyncio
async def test_detected_language_is_reported() -> None:
    """With auto-detection the response language wins over the configured default."""
    events = await _drain(
        [
            _Message(
                _ServerContent(
                    input_transcription=_Transcription(
                        " bonjour", finished=True, language_code="fr-FR"
                    )
                )
            ),
        ]
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "fr-FR"


@pytest.mark.asyncio
async def test_configured_language_is_used_when_none_is_detected() -> None:
    events = await _drain(
        [
            _Message(_ServerContent(input_transcription=_Transcription(" hi", finished=True))),
        ]
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[0].alternatives[0].language == "en-US"
