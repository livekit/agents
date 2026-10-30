"""TTS markup lifetime: the markup must not keep a dropped tts alive."""

from __future__ import annotations

import gc
import weakref

import pytest

from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

pytestmark = pytest.mark.unit


class _TTS(TTS):
    class Markup(TTS.Markup):
        def _provider_key(self) -> str:
            # reads the tts, like the inworld and gemini plugins do
            return "inworld" if "tts-2" in self._tts.model else ""

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

    @property
    def model(self) -> str:
        return "inworld-tts-2"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        raise NotImplementedError


def test_a_dropped_tts_is_freed_without_the_cyclic_gc() -> None:
    """The tts and its markup must not form a reference cycle.

    Caching the markup on the tts made a two-object cycle, so a dropped tts — with its
    options, HTTP session and connection pool — stayed alive until the cyclic GC ran,
    once per session for an agent that builds its tts in the job entrypoint.
    """
    gc.collect()
    gc.disable()
    try:
        speaker = _TTS()
        assert speaker.markup._provider_key() == "inworld"

        ref = weakref.ref(speaker)
        del speaker

        assert ref() is None
    finally:
        gc.enable()


def test_the_markup_is_rebuilt_on_each_access() -> None:
    """Nothing holds the markup, so it is not shared between accesses either."""
    speaker = _TTS()

    assert speaker.markup is not speaker.markup
    assert speaker.markup._provider_key() == "inworld"
