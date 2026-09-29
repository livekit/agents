"""A lost audio connection must still tear the stream down (offline)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError
from livekit.plugins.speechmatics import stt as speechmatics_stt

pytestmark = pytest.mark.plugin("speechmatics")


async def test_audio_failure_still_disconnects_client(monkeypatch) -> None:
    client = MagicMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_audio = AsyncMock()
    client.on = MagicMock()
    client.is_ready_for_audio = False  # gate closed by send_audio
    client.session_error = None  # clean close -> "lost connection"
    monkeypatch.setattr(speechmatics_stt, "AgentSttAsyncClient", lambda **kw: client)

    instance = speechmatics_stt.STT(api_key="test-key", vad=None)
    stream = instance.stream()
    stream._task.cancel()
    stream._metrics_task.cancel()

    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )
    stream.push_frame(frame)
    stream.end_input()

    with pytest.raises(APIConnectionError):
        await stream._run()

    message_task = stream._tasks[1]
    await asyncio.sleep(0)
    assert client.disconnect.await_count == 1, "client was never disconnected"
    assert message_task.done(), "message task leaked"
    assert stream not in instance._streams, "stream left in the active streams list"
