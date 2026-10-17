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

"""Respeecher TTS: the model is part of the websocket URL.

``_connect_ws`` builds ``{base}{model}/tts/websocket?...``, so a pooled connection is
bound to the model it was opened with. Changing the model has to stop those
connections being reused — and has to close them, not just set them aside where
nothing will ever drain them.
"""

from __future__ import annotations

import pytest

from livekit.plugins.respeecher import TTS

pytestmark = pytest.mark.plugin("respeecher")

EN = "/public/tts/en-rt"
UA = "/public/tts/ua-rt"


class _FakeWS:
    def __init__(self, model: str) -> None:
        self.model = model

    def __repr__(self) -> str:
        return f"_FakeWS({self.model})"


@pytest.fixture
def closed(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWS]:
    """Replace the network with fakes that record the model each socket was opened with.

    Patched on the class so that every pool the TTS builds uses them, including one
    created after construction.
    """
    closed_sockets: list[_FakeWS] = []

    async def fake_connect(self: TTS, timeout: float) -> _FakeWS:
        return _FakeWS(self._opts.model)

    async def fake_close(self: TTS, ws: _FakeWS) -> None:
        closed_sockets.append(ws)

    monkeypatch.setattr(TTS, "_connect_ws", fake_connect)
    monkeypatch.setattr(TTS, "_close_ws", fake_close)
    return closed_sockets


@pytest.mark.asyncio
async def test_changing_model_closes_the_idle_pooled_connection(closed: list[_FakeWS]) -> None:
    """An idle socket opened for the old model is closed, not left open until aclose()."""
    tts = TTS(api_key="test-key", model=EN)

    old = await tts._pool.get(timeout=10.0)
    tts._pool.put(old)  # finished with it: idle in the pool

    tts.update_options(model=UA)
    new = await tts._pool.get(timeout=10.0)  # next acquisition drains the pool

    assert new is not old
    assert new.model == UA, "the next connection must be opened for the new model"
    assert old in closed, "the old model's idle socket was left open"


@pytest.mark.asyncio
async def test_changing_model_lets_an_in_use_connection_finish(closed: list[_FakeWS]) -> None:
    """A socket a stream is still using survives the change, and is closed once returned."""
    tts = TTS(api_key="test-key", model=EN)

    in_use = await tts._pool.get(timeout=10.0)  # checked out, never returned yet
    tts.update_options(model=UA)
    await tts._pool.get(timeout=10.0)

    assert in_use not in closed, "changing the model severed a stream in flight"

    tts._pool.put(in_use)  # the stream finished with it
    await tts._pool.get(timeout=10.0)
    assert in_use in closed, "the returned old-model socket was never closed"


@pytest.mark.asyncio
async def test_same_model_keeps_the_pooled_connection(closed: list[_FakeWS]) -> None:
    """Re-setting the current model must not throw away a warm connection."""
    tts = TTS(api_key="test-key", model=EN)

    warm = await tts._pool.get(timeout=10.0)
    tts._pool.put(warm)

    tts.update_options(model=EN)

    assert await tts._pool.get(timeout=10.0) is warm
    assert closed == []
