from __future__ import annotations

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.nvidia import TTS

pytestmark = pytest.mark.unit


async def test_synthesis_failure_reaches_the_reader_with_its_sdk_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("Riva synthesis failed")

    class Service:
        def synthesize_online(self, *args, **kwargs):
            raise failure

    model = TTS(use_ssl=False)
    monkeypatch.setattr(model, "_ensure_session", lambda: Service())
    with pytest.raises(APIConnectionError) as caught:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("Hello there.")
            stream.end_input()
            async for _ in stream:
                pass
    assert caught.value.__cause__ is failure
