"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.plugin("gradium")


def test_stt_default_language():
    """STT defaults to English."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key")

    assert stt._opts.language == "en"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr")

    assert stt._opts.language == "fr"


def test_stream_language_overrides_stream_options():
    """stream language overrides the copied stream config."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream(language="de")

    assert stream._opts.language == "de"
    assert stt._opts.language == "fr"


@pytest.mark.asyncio
async def test_speech_stream_sends_language_in_json_config():
    """SpeechStream sends language in setup json_config."""
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    sent_messages: list[str] = []

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            sent_messages.append(message)

    class FakeSession:
        async def ws_connect(self, url, headers):
            return FakeWebSocket()

    stt = STT(api_key="test-key")
    opts = STTOptions(language=LanguageCode("pt"), temperature=0.3)

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),
        )

    await stream._connect_ws()

    setup_msg = json.loads(sent_messages[0])
    assert setup_msg["json_config"] == {"language": "pt", "temp": 0.3}


@pytest.mark.asyncio
async def test_send_side_socket_drop_raises_api_connection_error():
    """A mid-send WebSocket drop is surfaced as a retryable APIConnectionError.

    The base ``SpeechStream._main_task`` only reconnects on ``APIError``; any other
    exception kills the stream. ``ws.send_bytes``/``send_str`` on a peer-closed socket
    raises a raw ``aiohttp.ClientConnectionResetError`` (a ``ConnectionResetError``),
    which is *not* an ``APIError`` — so before this fix a send-side drop terminated the
    session with no reconnect, while an identical recv-side drop raised the retryable
    ``APIStatusError``. This test pins the send path to the same retryable behavior.
    """
    import aiohttp

    from livekit import rtc
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    class FakeWebSocket:
        def __init__(self) -> None:
            self._sends = 0
            self.close_code: int | None = None

        async def send_str(self, message: str) -> None:
            # first send is the setup payload from _connect_ws — let it succeed,
            # then drop the socket on the first audio chunk from send_task.
            self._sends += 1
            if self._sends == 1:
                return
            raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")

        async def receive(self):
            # recv_task must stay blocked so the send drop is what completes _run
            await asyncio.Event().wait()

        async def close(self) -> None:
            pass

    class FakeSession:
        closed = False

        async def ws_connect(self, url, headers):
            return FakeWebSocket()

    stt = STT(api_key="test-key")
    opts = STTOptions(language=LanguageCode("en"))

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),
        )

    # one full 1920-sample buffer so send_task emits exactly one audio chunk, which drops
    samples = 1920
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=opts.sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )
    stream.push_frame(frame)
    stream.end_input()

    with pytest.raises(APIConnectionError):
        await stream._run()
