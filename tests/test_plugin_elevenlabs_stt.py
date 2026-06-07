from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.stt import SpeechEventType
from livekit.plugins.elevenlabs import STT
from livekit.plugins.elevenlabs.stt import SpeechStream, VADOptions

pytestmark = [pytest.mark.unit, pytest.mark.plugin("elevenlabs")]


def _make_stream(*, server_vad: VADOptions | None = None) -> SpeechStream:
    stt = STT(
        api_key="test-key",
        model_id="scribe_v2_realtime",
        server_vad=server_vad,
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        return SpeechStream(
            stt=stt,
            opts=stt._opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            language=None,
            http_session=MagicMock(),
        )


def _commit_message(text: str) -> dict:
    return {
        "message_type": "committed_transcript",
        "text": text,
        "language_code": "en",
        "words": [{"text": text, "start": 0.0, "end": 0.4}],
    }


async def test_server_vad_commit_ends_speech() -> None:
    stream = _make_stream(server_vad={"vad_threshold": 0.4})

    stream._process_stream_event(_commit_message("hello"))

    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    assert stream._event_ch.recv_nowait().type == SpeechEventType.FINAL_TRANSCRIPT
    assert stream._event_ch.recv_nowait().type == SpeechEventType.END_OF_SPEECH
    assert stream._speaking is False


async def test_manual_commit_still_waits_for_empty_commit_to_end_speech() -> None:
    stream = _make_stream()

    stream._process_stream_event(_commit_message("hello"))

    assert stream._event_ch.recv_nowait().type == SpeechEventType.START_OF_SPEECH
    assert stream._event_ch.recv_nowait().type == SpeechEventType.FINAL_TRANSCRIPT
    assert stream._speaking is True
