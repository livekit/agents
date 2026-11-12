# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neuphonic TTS: settings that are baked into the websocket URL.

``_connect_ws`` builds its url as
``/speak/{lang_code}?speed={speed}&lang_code={lang_code}&encoding=...&voice_id={voice_id}``,
so ``lang_code``, ``speed`` and ``voice_id`` are all fixed at connection time. The
socket then lives in a ``utils.ConnectionPool`` created with
``mark_refreshed_on_get=True``, which restarts ``max_session_duration`` on every
acquire, so a reused connection can outlive a settings change indefinitely. Changing
any of the three must therefore invalidate the pool.

Also tests the retry behavior for streaming synthesis (regression for #7569):
the segments channel must be created per-run so that a retry after a connection
failure gets a fresh channel and does not fail with ChanClosed.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import APIConnectOptions
from livekit.plugins.neuphonic import TTS

pytestmark = pytest.mark.plugin("neuphonic")


def _tts_with_recording_pool() -> tuple[TTS, list[bool]]:
    tts = TTS(api_key="test-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]
    return tts, calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voice_id", "some-other-voice-id"),
        ("speed", 1.25),
        ("lang_code", "es"),
    ],
)
def test_update_options_invalidates_pool_for_connection_params(field: str, value: object) -> None:
    """Each of the three fields is a url parameter, so each must invalidate."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(**{field: value})

    assert calls == [True], f"changing {field} must invalidate the pooled connection"


def test_update_options_noop_leaves_pool_alone() -> None:
    """A call that changes nothing must not tear down a healthy pooled connection."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options()

    assert calls == []


def test_update_options_invalidates_once_for_several_changes() -> None:
    """Changing all three together still only needs a single invalidation."""
    tts, calls = _tts_with_recording_pool()

    tts.update_options(lang_code="fr", voice_id="another-voice", speed=0.9)

    assert calls == [True]
    assert tts._opts.voice_id == "another-voice"
    assert tts._opts.speed == 0.9


SR = 22050


class _FakeWS:
    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self.close_code = None
        self.q: asyncio.Queue = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        msg = json.loads(data)
        self.sent.append(msg["text"])
        pcm = b"\x01\x00" * int(SR * 0.1)
        for p in (
            {"audio": base64.b64encode(pcm).decode(), "context_id": msg["context_id"]},
            {"audio": "", "context_id": msg["context_id"], "stop": True},
        ):
            self.q.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps({"data": p}))
            )

    async def receive(self, timeout=None):
        return await self.q.get()

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_stream_retry_after_connection_error_does_not_raise_chanclosed() -> None:
    """Regression test for #7569: retry after a connection failure must not fail
    with ChanClosed because the segments channel was already closed by the
    first attempt's tokenizer.

    The first connect raises ClientConnectionError. The second returns a fake
    websocket that answers each <STOP>-terminated message with one audio frame
    and one stop frame.
    """
    tts = TTS(api_key="test-key", sample_rate=SR)
    ws = _FakeWS()
    calls = 0

    async def connect(timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise aiohttp.ClientConnectionError("simulated transient failure")
        return ws

    tts._pool._connect_cb = connect

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0))
    stream.push_text("Hello there, this is a single sentence.")
    stream.end_input()

    duration = 0.0
    try:
        async for ev in stream:
            duration += ev.frame.duration
    finally:
        await stream.aclose()
        await tts.aclose()

    assert calls == 2, f"expected 2 connect attempts, got {calls}"
    assert duration > 0, "expected audio output from successful retry"
