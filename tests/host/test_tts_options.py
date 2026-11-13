"""``tts.options`` live in the core: every TTS, plugin or framework, takes ``sentence_tokenizer``
and ``pacing`` through ``update_options``, and a change reaches the next synthesis wherever the TTS
was handed."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import pkgutil
from typing import Any

import pytest

import livekit.plugins
from livekit.agents import Agent, VoiceSession, inference, tts
from livekit.agents.types import NOT_GIVEN
from tests.fakes import FakeAudioOutput, FakeSentenceTokenizer, FakeTTS


def _tts_classes() -> list[type[tts.TTS[Any]]]:
    for module in pkgutil.walk_packages(livekit.plugins.__path__, "livekit.plugins."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # an optional dependency the environment lacks
            continue
    found: list[type[tts.TTS[Any]]] = []
    pending: list[type[tts.TTS[Any]]] = [tts.TTS]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            pending.append(sub)
            if sub.__module__.startswith(("livekit.plugins.", "livekit.agents.")):
                found.append(sub)
    return sorted(set(found), key=lambda c: f"{c.__module__}.{c.__qualname__}")


CLASSES = [c for c in _tts_classes() if "update_options" in vars(c)]


def test_every_tts_with_its_own_update_options_is_covered() -> None:
    names = {f"{c.__module__}.{c.__qualname__}" for c in CLASSES}
    assert "livekit.agents.tts.FallbackAdapter" in names
    assert "livekit.agents.inference.tts.TTS" in {
        f"{c.__module__}.{c.__qualname__}" for c in _tts_classes()
    }
    assert len([n for n in names if n.startswith("livekit.plugins.")]) >= 30


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c.__module__}.{c.__qualname__}")
def test_update_options_takes_the_text_options(cls: type[tts.TTS[Any]]) -> None:
    parameters = inspect.signature(cls.update_options).parameters
    for name in ("sentence_tokenizer", "pacing"):
        assert name in parameters, f"{cls.__qualname__}.update_options lacks {name}"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is NOT_GIVEN


@pytest.mark.parametrize(
    "cls",
    # a chain's options are its serving member's, which `FallbackAdapter`'s own tests cover
    [
        c
        for c in CLASSES
        if c.__module__.startswith("livekit.plugins.") and not issubclass(c, tts.FallbackAdapter)
    ],
    ids=lambda c: f"{c.__module__}.{c.__qualname__}",
)
def test_update_options_passes_the_text_options_on(cls: type[tts.TTS[Any]]) -> None:
    """A plugin's ``update_options`` passes the text options on first: they change even on a TTS
    its provider never configured, whose own options code may then fail."""
    provider = cls.__new__(cls)
    tts.TTS.__init__(
        provider,
        capabilities=tts.TTSCapabilities(streaming=False),
        sample_rate=24000,
        num_channels=1,
    )
    tokenizer = FakeSentenceTokenizer()

    with contextlib.suppress(AttributeError):
        provider.update_options(sentence_tokenizer=tokenizer, pacing=None)

    assert provider.options.sentence_tokenizer is tokenizer
    assert provider.options.pacing is None


async def test_a_tts_updated_after_it_was_wrapped_splits_by_its_new_tokenizer() -> None:
    """A TTS changed after a ``StreamAdapter`` and a ``FallbackAdapter`` wrap it splits their next
    synthesis by its new tokenizer."""
    provider = FakeTTS(streaming=False)
    adapter = tts.StreamAdapter(tts=provider)
    fallback = tts.FallbackAdapter([adapter])
    provider.update_options(sentence_tokenizer=FakeSentenceTokenizer(), pacing=None)

    async with fallback.stream() as stream:
        stream.push_text("one two|three")
        stream.end_input()
        _ = [audio async for audio in stream]
    await fallback.aclose()

    assert provider.texts == ["one two", "three"]
    assert adapter.options.sentence_tokenizer is provider.options.sentence_tokenizer


async def test_the_adapters_update_options_reach_the_tts_they_wrap() -> None:
    """``update_options`` on a ``StreamAdapter`` changes its TTS's options; on a
    ``FallbackAdapter``, every member's."""
    first, second = FakeTTS(streaming=False), FakeTTS()
    adapter = tts.StreamAdapter(tts=first)
    fallback = tts.FallbackAdapter([adapter, second])
    tokenizer = FakeSentenceTokenizer()

    fallback.update_options(sentence_tokenizer=tokenizer, pacing=None)

    assert first.options.sentence_tokenizer is tokenizer
    assert second.options.sentence_tokenizer is tokenizer
    assert fallback.options.sentence_tokenizer is tokenizer
    await fallback.aclose()


async def test_a_tts_updated_after_it_was_handed_to_a_session_splits_its_next_reply() -> None:
    provider = FakeTTS(streaming=False)

    async with VoiceSession(tts=provider, vad=None) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Help."))
        provider.update_options(sentence_tokenizer=FakeSentenceTokenizer(), pacing=None)
        await session.say("one two|three")

    assert provider.texts == ["one two", "three"]


async def test_inference_tts_update_options_changes_its_text_options() -> None:
    model = inference.TTS("cartesia/sonic-3", api_key="key", api_secret="secret")
    tokenizer = FakeSentenceTokenizer()

    model.update_options(voice="other", sentence_tokenizer=tokenizer, pacing=None)

    assert model.options.sentence_tokenizer is tokenizer and model.options.pacing is None
    await model.aclose()
