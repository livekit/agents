from __future__ import annotations

from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIStatusError, stt
from livekit.plugins.nabrah.stt import STT, SpeechStream

pytestmark = pytest.mark.unit


@pytest.fixture
def stream() -> SpeechStream:
    def _discard_task(
        coroutine: Coroutine[Any, Any, Any], *args: object, **kwargs: object
    ) -> MagicMock:
        coroutine.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_discard_task):
        return SpeechStream(
            stt_instance=STT(api_key="test-key"),
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            language="ar-SA",
            http_session=MagicMock(spec=aiohttp.ClientSession),
        )


def test_closed_utterance_with_repeated_prefix_starts_a_new_utterance(
    stream: SpeechStream,
) -> None:
    stream._process_message(
        {"type": "transcript", "text": "مرحبا", "is_final": True, "audio_processed": 1.0}
    )
    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا بكم",
            "is_final": False,
            "audio_processed": 2.0,
        }
    )

    assert stream._current_text() == "مرحبا مرحبا بكم"


def test_flushed_prefix_preserves_provider_punctuation(stream: SpeechStream) -> None:
    stream._stt._end_of_turn_confirm_delay_seconds = None

    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا <eot>",
            "is_final": False,
            "audio_processed": 1.0,
        }
    )
    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا. كيف <eot>",
            "is_final": False,
            "audio_processed": 2.0,
        }
    )

    assert stream._utt_flushed_clean == "مرحبا. كيف"

    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا. كيف حالك",
            "is_final": False,
            "audio_processed": 3.0,
        }
    )

    assert stream._current_text() == "حالك"


def test_filtered_word_does_not_move_raw_word_cursor_backwards(stream: SpeechStream) -> None:
    stream._stt._end_of_turn_confirm_delay_seconds = None
    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا <eot>",
            "words": [{"word": "مرحبا"}, {"word": "<eot>"}],
        }
    )

    assert stream._utt_flushed_words == 2

    stream._process_message(
        {
            "type": "transcript",
            "text": "مرحبا <eot> كيف",
            "words": [{"word": "مرحبا"}, {"word": "<eot>"}, {"word": "كيف"}],
        }
    )

    assert list(stream._utt_words) == ["كيف"]


def test_provider_error_does_not_expose_provider_message(stream: SpeechStream) -> None:
    transcript = "private customer transcript"

    with pytest.raises(APIStatusError) as exc_info:
        stream._process_message({"type": "error", "message": transcript})

    assert transcript not in str(exc_info.value)


async def test_malformed_message_does_not_expose_provider_content(
    stream: SpeechStream, caplog: pytest.LogCaptureFixture
) -> None:
    transcript = "private customer transcript"
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.receive = AsyncMock(
        side_effect=[
            MagicMock(type=aiohttp.WSMsgType.TEXT, data='{"type": "transcript"}'),
            MagicMock(type=aiohttp.WSMsgType.CLOSED),
        ]
    )
    stream._input_done = True

    with (
        caplog.at_level("WARNING", logger="livekit.plugins.nabrah"),
        patch.object(stream, "_process_message", side_effect=ValueError(transcript)),
    ):
        await stream._recv_task(ws)

    assert "nabrah STT returned malformed data" in caplog.text
    assert transcript not in caplog.text


def test_stream_failure_is_not_retried_after_audio_is_consumed(stream: SpeechStream) -> None:
    assert stream._stream_failure("failed").retryable is True

    stream._audio_position = 0.1

    assert stream._stream_failure("failed").retryable is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("is_final", "false"), ("audio_processed", float("nan"))],
)
def test_invalid_transcript_fields_are_rejected(
    stream: SpeechStream, field: str, value: object
) -> None:
    message: dict[str, object] = {"type": "transcript", "text": "مرحبا", field: value}

    with pytest.raises(ValueError):
        stream._process_message(message)


def test_flush_reports_usage_without_transcript(stream: SpeechStream) -> None:
    emitted: list[stt.SpeechEvent] = []
    stream._audio_position = 1.25

    with patch.object(stream, "_emit", side_effect=emitted.append):
        stream._flush_eos()

    usage_events = [
        event for event in emitted if event.type == stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert len(usage_events) == 1
    assert usage_events[0].recognition_usage is not None
    assert usage_events[0].recognition_usage.audio_duration == pytest.approx(1.25)
