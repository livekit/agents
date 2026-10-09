from __future__ import annotations

import time
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIStatusError,
    stt,
)
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


def test_open_utterance_correction_replaces_previous_hypothesis(stream: SpeechStream) -> None:
    stream._process_message({"type": "transcript", "text": "مرحبا بكم", "is_final": False})
    stream._process_message({"type": "transcript", "text": "مرحبا بكن", "is_final": False})

    assert stream._current_text() == "مرحبا بكن"


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


async def test_mid_stream_failure_is_retryable(stream: SpeechStream) -> None:
    """A socket that drops mid-utterance must not take the session's STT with it."""
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.receive = AsyncMock(return_value=MagicMock(type=aiohttp.WSMsgType.ERROR))
    stream._audio_position = 12.5

    with pytest.raises(APIConnectionError) as exc_info:
        await stream._recv_task(ws)

    assert exc_info.value.retryable is True


async def test_unexpected_close_after_audio_is_retryable(stream: SpeechStream) -> None:
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.receive = AsyncMock(return_value=MagicMock(type=aiohttp.WSMsgType.CLOSED))
    stream._audio_position = 12.5
    stream._input_done = False

    with pytest.raises(APIConnectionError) as exc_info:
        await stream._recv_task(ws)

    assert exc_info.value.retryable is True


def test_reset_clears_the_cursors_a_reconnect_would_misread(stream: SpeechStream) -> None:
    """Flushed cursors index into one socket's cumulative stream."""
    stream._process_message({"type": "transcript", "text": "مرحبا بكم.", "is_final": False})
    stream._flush_eos()

    assert stream._utt_flushed_chars > 0

    stream._reset_connection_state()

    assert stream._utt_flushed_clean == ""
    assert stream._utt_flushed_chars == 0
    assert stream._utt_flushed_words == 0
    assert stream._utt_raw == ""
    assert stream._utt_raw_seen == 0
    assert stream._is_speaking is False


def test_transcript_after_reconnect_is_not_truncated(stream: SpeechStream) -> None:
    """The replayed utterance must not be sliced at the previous socket's cursor."""
    stream._stt._end_of_turn_confirm_delay_seconds = None
    stream._process_message({"type": "transcript", "text": "مرحبا بكم.", "is_final": False})
    stream._flush_eos()

    stream._reset_connection_state()
    stream._process_message({"type": "transcript", "text": "مرحبا بكم", "is_final": False})

    assert stream._current_text() == "مرحبا بكم"


async def _flush(stream: SpeechStream) -> None:
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.send_bytes = AsyncMock()
    ws.send_str = AsyncMock()
    stream._input_ch.send_nowait(SpeechStream._FlushSentinel())
    stream._input_ch.close()
    await stream._send_task(ws)


async def test_flush_waits_for_the_recognizer_to_acknowledge_the_audio(
    stream: SpeechStream,
) -> None:
    """Writing audio is not consuming it: committing on the spot closes an empty turn."""
    stream._audio_position = 1.0
    emitted: list[stt.SpeechEvent] = []

    with patch.object(stream, "_emit", side_effect=emitted.append):
        await _flush(stream)

        assert stream._pending_flush_position == 1.0
        assert not emitted

        stream._process_message(
            {"type": "transcript", "text": "مرحبا", "is_final": False, "audio_processed": 1.0}
        )
        stream._maybe_complete_flush()

    finals = [e for e in emitted if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["مرحبا"]
    assert stream._pending_flush_position is None


async def test_flush_does_not_commit_before_the_clock_catches_up(stream: SpeechStream) -> None:
    stream._audio_position = 5.0
    emitted: list[stt.SpeechEvent] = []

    with patch.object(stream, "_emit", side_effect=emitted.append):
        await _flush(stream)
        stream._process_message(
            {"type": "transcript", "text": "مرحبا", "is_final": False, "audio_processed": 1.0}
        )
        stream._maybe_complete_flush()

    assert not [e for e in emitted if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert stream._pending_flush_position == 5.0


async def test_flush_commits_when_the_recognizer_never_acknowledges(
    stream: SpeechStream,
) -> None:
    """A recognizer with nothing to say never moves the clock."""
    stream._audio_position = 5.0
    stream._process_message({"type": "transcript", "text": "مرحبا", "is_final": False})
    emitted: list[stt.SpeechEvent] = []

    with patch.object(stream, "_emit", side_effect=emitted.append):
        await _flush(stream)
        stream._pending_flush_deadline = time.monotonic() - 0.1
        stream._maybe_complete_flush()

    finals = [e for e in emitted if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.alternatives[0].text for e in finals] == ["مرحبا"]


def test_reset_drops_a_flush_the_dead_socket_can_no_longer_acknowledge(
    stream: SpeechStream,
) -> None:
    stream._pending_flush_position = 5.0
    stream._pending_flush_deadline = time.monotonic() + 2.0

    stream._reset_connection_state()

    assert stream._pending_flush_position is None
    assert stream._pending_flush_deadline is None


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (404, False), (429, True), (503, True)],
)
async def test_handshake_status_decides_retryability(
    stream: SpeechStream, status: int, retryable: bool
) -> None:
    """Redialling a URL the server rejected outright only delays the real error."""
    stream._session.ws_connect = MagicMock(  # type: ignore[method-assign]
        side_effect=aiohttp.WSServerHandshakeError(
            MagicMock(), (), status=status, message="rejected"
        )
    )

    with pytest.raises(APIStatusError) as exc_info:
        await stream._connect_ws()

    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable


async def test_transport_failure_stays_retryable(stream: SpeechStream) -> None:
    stream._session.ws_connect = MagicMock(  # type: ignore[method-assign]
        side_effect=aiohttp.ClientOSError("connection reset")
    )

    with pytest.raises(APIConnectionError) as exc_info:
        await stream._connect_ws()

    assert exc_info.value.retryable is True


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
