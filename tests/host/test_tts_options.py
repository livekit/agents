"""``tts.options`` live in the core: every TTS, plugin or framework, takes ``sentence_tokenizer``
and ``pacing`` through ``update_options``, and a change reaches the next synthesis wherever the TTS
was handed."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any, cast

import pytest

import livekit.plugins
from livekit.agents import Agent, VoiceSession, http, inference, tts
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import (
    azure,
    cartesia,
    deepgram,
    fishaudio,
    google,
    inworld,
    minimax,
    mistralai,
    rime,
    sarvam,
    soniox,
    vakyam,
)
from livekit.plugins.azure import tts as azure_tts
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


# what a plugin needs besides an API key to be constructed
_CONSTRUCTION: dict[str, dict[str, str]] = {
    "livekit.plugins.azure.tts.TTS": {"speech_key": "test", "speech_region": "westus"},
    "livekit.plugins.baseten.tts.TTS": {"model_endpoint": "https://model.invalid/predict"},
    "livekit.plugins.slng.tts._Connection": {"voice": "test", "model": "deepgram/aura:2"},
}


def _construct(cls: type[tts.TTS[Any]]) -> tts.TTS[Any]:
    name = f"{cls.__module__}.{cls.__qualname__}"
    key = {"api_key": "test"} if "api_key" in inspect.signature(cls).parameters else {}
    return cls(**key, **_CONSTRUCTION.get(name, {}))


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
async def test_update_options_passes_the_text_options_on(cls: type[tts.TTS[Any]]) -> None:
    """A plugin's ``update_options`` passes the text options on to its ``options``."""
    async with http.open():
        provider = _construct(cls)
        tokenizer = FakeSentenceTokenizer()
        try:
            provider.update_options(sentence_tokenizer=tokenizer, pacing=None)

            assert provider.options.sentence_tokenizer is tokenizer
            assert provider.options.pacing is None
        finally:
            await provider.aclose()


Refusal = Callable[[Any, FakeSentenceTokenizer, tts.SentenceStreamPacer], None]

# each plugin's update_options with an option it refuses, beside new text options
_REFUSALS: list[tuple[str, Callable[[], tts.TTS[Any]], Refusal]] = [
    (
        "azure",
        lambda: azure.TTS(speech_key="test", speech_region="westus"),
        lambda t, tok, pacer: t.update_options(
            prosody=azure_tts.ProsodyConfig(rate=3.0), sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "cartesia",
        lambda: cartesia.TTS(api_key="test", model="sonic-3"),
        lambda t, tok, pacer: t.update_options(speed="fast", sentence_tokenizer=tok, pacing=pacer),
    ),
    (
        "deepgram",
        lambda: deepgram.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(speed=5.0, sentence_tokenizer=tok, pacing=pacer),
    ),
    (
        "fishaudio",
        lambda: fishaudio.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(
            temperature=2.0, sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "google-gemini",
        lambda: google.GeminiTTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(
            speaker="nobody", sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "inworld",
        lambda: inworld.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(
            encoding=cast(Any, "NOT_AN_ENCODING"), sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "minimax",
        lambda: minimax.TTS(api_key="test", model="speech-02-hd"),
        lambda t, tok, pacer: t.update_options(speed=2.5, sentence_tokenizer=tok, pacing=pacer),
    ),
    (
        "mistralai",
        lambda: mistralai.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(
            voice="one", ref_audio="two", sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "rime",
        lambda: rime.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(
            audio_format=cast(Any, "not-a-format"), sentence_tokenizer=tok, pacing=pacer
        ),
    ),
    (
        "sarvam",
        lambda: sarvam.TTS(api_key="test", model="bulbul:v3"),
        lambda t, tok, pacer: t.update_options(pace=2.5, sentence_tokenizer=tok, pacing=pacer),
    ),
    (
        "soniox",
        lambda: soniox.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(speed=5.0, sentence_tokenizer=tok, pacing=pacer),
    ),
    (
        "vakyam",
        lambda: vakyam.TTS(api_key="test"),
        lambda t, tok, pacer: t.update_options(speed=99.0, sentence_tokenizer=tok, pacing=pacer),
    ),
]


@pytest.mark.parametrize(
    ("factory", "refuse"),
    [pytest.param(factory, refuse, id=name) for name, factory, refuse in _REFUSALS],
)
async def test_a_refused_update_leaves_the_text_options(
    factory: Callable[[], tts.TTS[Any]], refuse: Refusal
) -> None:
    """``update_options`` validates every option before applying any: one it refuses leaves
    the sentence tokenizer and the pacing it was given unapplied."""
    async with http.open():
        provider = factory()
        before = provider.options
        try:
            with pytest.raises(ValueError):
                refuse(
                    provider,
                    FakeSentenceTokenizer(),
                    tts.SentenceStreamPacer(min_remaining_audio=1.25),
                )

            assert provider.options.sentence_tokenizer is before.sentence_tokenizer
            assert provider.options.pacing == before.pacing
        finally:
            await provider.aclose()


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
