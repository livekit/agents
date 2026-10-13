from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt, utils
from livekit.plugins.baseten.qwen3_stt import Qwen3SpeechStream

pytestmark = pytest.mark.plugin("baseten")


@pytest.mark.parametrize(
    "end_time,empty_final", [(0.4, False), (0.0, False), (None, False), (None, True)]
)
async def test_qwen3_endpoint_preserves_segment_timing(end_time, empty_final):
    stream = object.__new__(Qwen3SpeechStream)
    stream._start_time_offset = 7.0
    stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
    stream._opts = SimpleNamespace(word_timestamps=True, handshake=lambda: {})
    stream._input_ch = utils.aio.Chan()
    stream._input_ch.close()
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    segment = {"text": "hello", "start_time": 0.1}
    if end_time is not None:
        segment["end_time"] = end_time
    message = {
        "type": "transcription",
        "segments": [segment],
        "is_final": True,
        "is_end_of_audio_flush": True,
    }
    messages = [message]
    if empty_final:
        messages = [
            {**message, "is_final": False, "is_end_of_audio_flush": False},
            {**message, "segments": []},
        ]
    ws = SimpleNamespace(
        send_str=AsyncMock(),
        close=AsyncMock(),
        receive=AsyncMock(
            side_effect=[
                aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(item), "") for item in messages
            ]
        ),
    )
    stream._backend = SimpleNamespace(
        _ensure_session=lambda: SimpleNamespace(ws_connect=AsyncMock(return_value=ws)),
        _model_endpoint="wss://test.invalid",
        _api_key="test-key",
    )
    await stream._run()
    endpoint = events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []
