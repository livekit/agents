from __future__ import annotations

import gc
import weakref

import pytest

from livekit.agents import inference

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


def test_a_dropped_tts_is_freed_without_the_cyclic_gc() -> None:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        speaker = FakeTTS()
        speaker.markup._provider_key()
        speaker_ref = weakref.ref(speaker)
        del speaker
        assert speaker_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()


def test_markup_works_after_its_tts_is_dropped() -> None:
    markup = inference.TTS("fishaudio/s2.1-pro", api_key="fake", api_secret="fake").markup
    gc.collect()
    assert markup._provider_key() == "fishaudio"
