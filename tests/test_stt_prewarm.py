from __future__ import annotations

import pytest

from livekit.agents.stt import FallbackAdapter, StreamAdapter, STTCapabilities

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


class PrewarmableSTT(FakeSTT):
    """FakeSTT that records prewarm calls, optionally without streaming support."""

    def __init__(self, *, streaming: bool = True) -> None:
        super().__init__()
        self._capabilities = STTCapabilities(streaming=streaming, interim_results=False)
        self.prewarm_count = 0

    def prewarm(self) -> None:
        self.prewarm_count += 1


async def test_fallback_adapter_prewarms_primary_stt() -> None:
    primary = PrewarmableSTT()
    fallback = PrewarmableSTT()

    fallback_adapter = FallbackAdapter([primary, fallback])
    try:
        fallback_adapter.prewarm()

        assert primary.prewarm_count == 1, "expected the primary STT to be prewarmed"
        assert fallback.prewarm_count == 0, (
            "expected only the primary STT to be prewarmed, the fallbacks should stay cold"
        )
    finally:
        await fallback_adapter.aclose()


async def test_stream_adapter_forwards_prewarm() -> None:
    wrapped = PrewarmableSTT(streaming=False)

    stream_adapter = StreamAdapter(stt=wrapped, vad=FakeVAD())
    try:
        stream_adapter.prewarm()

        assert wrapped.prewarm_count == 1, (
            "expected StreamAdapter to forward prewarm to the wrapped STT"
        )
    finally:
        await stream_adapter.aclose()


async def test_prewarm_reaches_non_streaming_primary() -> None:
    # a non-streaming STT is transparently wrapped in a StreamAdapter by the fallback
    # adapter, so prewarm has to survive both hops to reach the provider
    primary = PrewarmableSTT(streaming=False)
    fallback = PrewarmableSTT()

    fallback_adapter = FallbackAdapter([primary, fallback], vad=FakeVAD())
    try:
        fallback_adapter.prewarm()

        assert primary.prewarm_count == 1, (
            "expected prewarm to reach the primary STT through the wrapping StreamAdapter"
        )
        assert fallback.prewarm_count == 0
    finally:
        await fallback_adapter.aclose()
