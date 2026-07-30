"""Transcription context for the OpenAI realtime STT.

`gpt-live-transcribe` and `gpt-transcribe` accept `prompt`, `keywords` and `languages` as
recognition hints, and the transcription config can be re-sent mid-session. These tests cover
the payload that ends up in `session.update` and the in-band update path.
See https://developers.openai.com/api/docs/guides/realtime-transcription
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from livekit.agents.utils import is_given
from livekit.plugins.openai import stt

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    """A websocket that records what the plugin sends and never sends anything back."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._close_ev = asyncio.Event()

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)

    async def receive(self) -> None:
        await self._close_ev.wait()

    async def close(self) -> None:
        self.closed = True
        self._close_ev.set()

    @property
    def session_updates(self) -> list[dict[str, Any]]:
        return [msg for msg in self.sent if msg.get("type") == "session.update"]


class _FakeSession:
    def __init__(self) -> None:
        self.ws = _FakeWebSocket()

    async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
        return self.ws


async def _connected_stream(instance: stt.STT, session: _FakeSession) -> stt.SpeechStream:
    instance._session = session  # type: ignore[assignment]
    stream = instance.stream()

    async def wait_connected() -> None:
        while stream._ws is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_connected(), timeout=5)
    return stream


def _make_stt(**kwargs: Any) -> stt.STT:
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "model": "gpt-live-transcribe",
        "use_realtime": True,
    }
    defaults.update(kwargs)
    return stt.STT(**defaults)


def _transcription(instance: stt.STT) -> dict[str, Any]:
    session_update = instance._build_session_update()
    transcription: dict[str, Any] = session_update["session"]["audio"]["input"]["transcription"]
    return transcription


# -- transcription config --


def test_context_fields_are_sent() -> None:
    instance = _make_stt(
        prompt="A customer support call about a premium plan.",
        keywords=["premium plan", "AC-42"],
        languages=["en", "fr"],
        delay="low",
    )

    transcription = _transcription(instance)

    assert transcription["model"] == "gpt-live-transcribe"
    assert transcription["prompt"] == "A customer support call about a premium plan."
    assert transcription["keywords"] == ["premium plan", "AC-42"]
    assert transcription["languages"] == ["en", "fr"]
    assert transcription["delay"] == "low"


def test_singular_language_is_never_sent_alongside_languages() -> None:
    """The Realtime API rejects a session update carrying both fields."""
    instance = _make_stt(language="en", languages=["en", "fr"])

    transcription = _transcription(instance)

    assert transcription["languages"] == ["en", "fr"]
    assert "language" not in transcription


def test_language_is_carried_over_as_a_list_for_gpt_live_transcribe() -> None:
    instance = _make_stt(language="ja")

    transcription = _transcription(instance)

    assert transcription["languages"] == ["ja"]
    assert "language" not in transcription


def test_other_models_keep_the_singular_language() -> None:
    instance = _make_stt(model="gpt-4o-mini-transcribe", language="ja")

    transcription = _transcription(instance)

    assert transcription["language"] == "ja"
    assert "languages" not in transcription


def test_context_is_ignored_for_unsupported_models(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        instance = _make_stt(model="gpt-4o-mini-transcribe", keywords=["AC-42"])

    transcription = _transcription(instance)

    assert "keywords" not in transcription
    assert "'keywords' is not supported" in caplog.text


def test_delay_is_ignored_outside_realtime(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        instance = _make_stt(use_realtime=False, delay="low")

    assert not is_given(instance._opts.delay)
    assert "'delay' is only supported for realtime transcription" in caplog.text


def test_keywords_reject_forbidden_characters() -> None:
    for keyword in ("<premium>", "two\nlines"):
        with pytest.raises(ValueError, match="forbidden character"):
            _make_stt(keywords=[keyword])


# -- keyterm capability --


def test_keyterms_capability_follows_the_model() -> None:
    assert _make_stt(model="gpt-live-transcribe").capabilities.keyterms is True
    assert _make_stt(model="gpt-transcribe").capabilities.keyterms is True
    assert _make_stt(model="gpt-4o-mini-transcribe").capabilities.keyterms is False
    # keywords are a realtime-only field
    assert _make_stt(use_realtime=False).capabilities.keyterms is False


def test_session_keyterms_merge_with_user_keywords() -> None:
    instance = _make_stt(keywords=["AC-42"])

    instance._update_session_keyterms(["AC-42", "premium plan"])

    assert _transcription(instance)["keywords"] == ["AC-42", "premium plan"]


def test_session_keyterms_are_dropped_when_cleared() -> None:
    instance = _make_stt(keywords=["AC-42"])

    instance._update_session_keyterms(["premium plan"])
    instance._update_session_keyterms([])

    assert _transcription(instance)["keywords"] == ["AC-42"]


# -- in-band updates --


async def test_context_update_is_sent_on_the_live_connection() -> None:
    instance = _make_stt(keywords=["AC-42"])
    session = _FakeSession()
    stream = await _connected_stream(instance, session)

    instance.update_options(keywords=["AC-42", "billing"], delay="high")
    assert stream._session_update_atask is not None
    await asyncio.wait_for(stream._session_update_atask, timeout=5)

    # the connection is kept: the update rides on the existing session
    assert len(session.ws.session_updates) == 2
    transcription = session.ws.session_updates[-1]["session"]["audio"]["input"]["transcription"]
    assert transcription["keywords"] == ["AC-42", "billing"]
    assert transcription["delay"] == "high"

    await stream.aclose()


async def test_session_keyterms_are_pushed_to_the_live_connection() -> None:
    instance = _make_stt()
    session = _FakeSession()
    stream = await _connected_stream(instance, session)

    instance._update_session_keyterms(["premium plan"])
    assert stream._session_update_atask is not None
    await asyncio.wait_for(stream._session_update_atask, timeout=5)

    transcription = session.ws.session_updates[-1]["session"]["audio"]["input"]["transcription"]
    assert transcription["keywords"] == ["premium plan"]

    await stream.aclose()


async def test_context_update_without_a_connection_is_a_no_op() -> None:
    instance = _make_stt()
    stream = instance.stream()

    instance.update_options(prompt="a support call")
    assert stream._session_update_atask is not None
    await asyncio.wait_for(stream._session_update_atask, timeout=5)

    await stream.aclose()


async def test_context_updates_are_sent_in_order() -> None:
    instance = _make_stt()
    session = _FakeSession()
    stream = await _connected_stream(instance, session)

    instance.update_options(prompt="first")
    instance.update_options(prompt="second")
    assert stream._session_update_atask is not None
    await asyncio.wait_for(stream._session_update_atask, timeout=5)

    prompts = [
        msg["session"]["audio"]["input"]["transcription"].get("prompt")
        for msg in session.ws.session_updates[1:]
    ]
    assert prompts == ["first", "second"]

    await stream.aclose()
