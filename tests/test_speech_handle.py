from __future__ import annotations

import asyncio

import pytest

from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


class _Core:
    """The core's handle as the host reads it: done as soon as the core published the speech's
    end, which is before the session's ``speech_done`` reaches the host loop."""

    host = None
    id = "speech_1"
    is_done = False

    def done(self) -> bool:
        return self.is_done


async def test_done_callbacks_run_in_registration_order_when_speech_done_lags() -> None:
    core = _Core()
    handle = SpeechHandle(core)  # type: ignore[arg-type]
    calls: list[str] = []
    handle.add_done_callback(lambda _: calls.append("first"))
    core.is_done = True
    # added after the core ended the speech but before the session reported it done
    handle.add_done_callback(lambda _: calls.append("second"))
    await asyncio.sleep(0)
    handle._mark_done()
    await asyncio.sleep(0)
    assert calls == ["first", "second"]
