"""Tests for the Soniox STT plugin: target-language segments on `SpeechData`.

Covers:
- `SpeechData.__post_init__` coerces `target_languages` strings to `LanguageCode`.
- `_TokenAccumulator._lang_segments` coalesces consecutive same-language tokens.
- `_lang_segments_to_fields` helper edge cases.
- `_recv_messages_task` end-to-end via a fake WebSocket, asserting the resulting
  `SpeechData` fields on `FINAL_TRANSCRIPT` and `INTERIM_TRANSCRIPT`.

Mirrors the unit-test style of `tests/test_plugin_assemblyai_stt.py`: patches
`asyncio.create_task` during `SpeechStream.__init__` so no real connection is
opened, then drives `_recv_messages_task` directly with canned token messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from livekit.agents import APIStatusError
from livekit.agents.language import LanguageCode
from livekit.agents.stt import SpeechData, SpeechEventType

pytestmark = pytest.mark.plugin("soniox")

# ---------------------------------------------------------------------------
# SpeechData.__post_init__: target_languages coercion
# ---------------------------------------------------------------------------


def test_speech_data_target_languages_coerces_strings_to_language_code():
    sd = SpeechData(
        language="en",
        text="hello",
        target_languages=["en", "es"],
        target_texts=["hello, ", "hola."],
    )

    assert sd.target_languages is not None
    assert all(isinstance(lang, LanguageCode) for lang in sd.target_languages)
    assert sd.target_languages == [LanguageCode("en"), LanguageCode("es")]
    assert sd.target_texts == ["hello, ", "hola."]


def test_speech_data_target_languages_none_stays_none():
    sd = SpeechData(language="en", text="hello")
    assert sd.target_languages is None
    assert sd.target_texts is None


def test_speech_data_target_languages_preserves_existing_language_code_instances():
    code = LanguageCode("en")
    sd = SpeechData(language="en", text="hello", target_languages=[code])

    assert sd.target_languages is not None
    assert sd.target_languages[0] is code


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_endpoint_latency_adjustment_level_accepts_supported_values(level: int):
    from livekit.plugins.soniox import STTOptions

    assert (
        STTOptions(endpoint_latency_adjustment_level=level).endpoint_latency_adjustment_level
        == level
    )


@pytest.mark.parametrize("level", [-1, 4])
def test_endpoint_latency_adjustment_level_rejects_unsupported_values(level: int):
    from livekit.plugins.soniox import STTOptions

    with pytest.raises(
        ValueError, match="endpoint_latency_adjustment_level must be between 0 and 3"
    ):
        STTOptions(endpoint_latency_adjustment_level=level)


# ---------------------------------------------------------------------------
# _TokenAccumulator._lang_segments coalescing
# ---------------------------------------------------------------------------


def test_token_accumulator_coalesces_consecutive_same_language_runs():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    for lang, text in [
        ("en", "Hello"),
        ("en", " world"),
        ("es", " hola"),
        ("es", " mundo"),
        ("en", " again"),
    ]:
        accumulator.update({"text": text, "language": lang, "is_final": True})

    assert accumulator._lang_segments == [
        ("en", "Hello world"),
        ("es", " hola mundo"),
        ("en", " again"),
    ]
    assert accumulator.text == "Hello world hola mundo again"
    assert "".join(text for _, text in accumulator._lang_segments) == accumulator.text


def test_token_accumulator_lang_segments_empty_initially():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    assert accumulator._lang_segments == []


# ---------------------------------------------------------------------------
# _lang_segments_to_fields helper
# ---------------------------------------------------------------------------


def test_lang_segments_to_fields_empty_returns_none_pair():
    from livekit.plugins.soniox.stt import _lang_segments_to_fields

    langs, texts = _lang_segments_to_fields([])
    assert langs is None
    assert texts is None


def test_lang_segments_to_fields_coerces_to_language_code():
    from livekit.plugins.soniox.stt import _lang_segments_to_fields

    langs, texts = _lang_segments_to_fields([("en", "Hello, "), ("es", "hola.")])
    assert langs == [LanguageCode("en"), LanguageCode("es")]
    assert texts == ["Hello, ", "hola."]
    assert langs is not None
    assert all(isinstance(lang, LanguageCode) for lang in langs)


# ---------------------------------------------------------------------------
# _recv_messages_task end-to-end via a fake WebSocket
# ---------------------------------------------------------------------------


class _FakeWSMessage:
    """Minimal stand-in for `aiohttp.WSMessage` used by `_recv_messages_task`."""

    def __init__(self, data: Any, msg_type: aiohttp.WSMsgType = aiohttp.WSMsgType.TEXT) -> None:
        self.type = msg_type
        self.data = data if isinstance(data, str) else json.dumps(data)


class _FakeWebSocket:
    """One-shot async-iterable that yields canned messages once, then stays empty.

    `__anext__` awaits `asyncio.sleep(0)` so each iteration yields control back
    to the event loop — without that yield, the plugin's tight outer
    `while self._ws:` loop after the iterator exhausts would starve the test's
    `_event_ch.recv()` consumer and the test would hang.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [_FakeWSMessage(m) for m in messages]

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> _FakeWSMessage:
        await asyncio.sleep(0)
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _make_stream(translation: Any = None):
    """Construct a Soniox `SpeechStream` without spawning its real `_main_task`.

    Patches `asyncio.create_task` (the call site in `RecognizeStream.__init__`) so
    `_main_task` isn't scheduled — the stream's `_event_ch` is still a live
    channel that the test drives directly via `_recv_messages_task`.
    """
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.soniox import STT
    from livekit.plugins.soniox.stt import SpeechStream

    stt_instance = STT(api_key="test-key")
    if translation is not None:
        stt_instance._params.translation = translation

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(stt=stt_instance, conn_options=DEFAULT_API_CONNECT_OPTIONS)
    return stream


@pytest.mark.parametrize("level", [None, 0, 2])
async def test_endpoint_latency_adjustment_level_websocket_config(level: int | None):
    from livekit.plugins.soniox import STTOptions

    stream = _make_stream()
    stream._stt._params = STTOptions(endpoint_latency_adjustment_level=level)

    class FakeWebSocket:
        config: dict[str, Any] | None = None

        async def send_str(self, message: str) -> None:
            self.config = json.loads(message)

    class FakeSession:
        def __init__(self, ws: FakeWebSocket) -> None:
            self.ws = ws

        async def ws_connect(self, url: str) -> FakeWebSocket:
            return self.ws

    ws = FakeWebSocket()
    stream._stt._http_session = FakeSession(ws)

    await stream._connect_ws()

    assert ws.config is not None
    if level is None:
        assert "endpoint_latency_adjustment_level" not in ws.config
    else:
        assert ws.config["endpoint_latency_adjustment_level"] == level


async def _drive_recv(
    stream, messages: list[dict[str, Any]], *, expect_events: int, timeout: float = 2.0
):
    """Drive `_recv_messages_task` with canned messages, return collected events.

    Sets `stream._ws = None` after collecting events so the plugin's outer
    `while self._ws:` loop exits cleanly. Cancellation is the fallback for
    timeout / event-count mismatches.
    """
    stream._ws = _FakeWebSocket(messages)
    task = asyncio.create_task(stream._recv_messages_task())
    events = []
    try:
        for _ in range(expect_events):
            ev = await asyncio.wait_for(stream._event_ch.recv(), timeout=timeout)
            events.append(ev)
        return events
    finally:
        stream._ws = None
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        except BaseException:
            pass


def _final_token(text: str, language: str, translation_status: str | None = None) -> dict:
    token: dict[str, Any] = {"text": text, "language": language, "is_final": True}
    if translation_status is not None:
        token["translation_status"] = translation_status
    return token


def _nonfinal_token(text: str, language: str, translation_status: str | None = None) -> dict:
    token: dict[str, Any] = {"text": text, "language": language, "is_final": False}
    if translation_status is not None:
        token["translation_status"] = translation_status
    return token


END_TOKEN_FINAL: dict[str, Any] = {"text": "<end>", "is_final": True}


# --- Two-way translation, code-switched input -------------------------------


async def test_final_transcript_two_way_code_switched():
    """The issue's canonical example: 'No hablo español but I speak English' →
    'I don't speak Spanish, pero hablo inglés.' produces per-run source and
    target lists with different language orderings."""
    from livekit.plugins.soniox.stt import TranslationConfig

    stream = _make_stream(
        translation=TranslationConfig(type="two_way", language_a="en", language_b="es")
    )

    messages = [
        {
            "tokens": [
                _final_token("No hablo español, ", "es", translation_status="original"),
                _final_token("but I speak English.", "en", translation_status="original"),
                _final_token("I don't speak Spanish, ", "en", translation_status="translation"),
                _final_token("pero hablo inglés.", "es", translation_status="translation"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 1000,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)

    types = [e.type for e in events]
    assert SpeechEventType.FINAL_TRANSCRIPT in types
    assert SpeechEventType.END_OF_SPEECH in types

    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    assert len(final.alternatives) == 1
    sd = final.alternatives[0]

    assert sd.text == "I don't speak Spanish, pero hablo inglés."
    assert sd.language == LanguageCode("en")

    assert sd.source_languages == [LanguageCode("es"), LanguageCode("en")]
    assert sd.source_texts == ["No hablo español, ", "but I speak English."]

    assert sd.target_languages == [LanguageCode("en"), LanguageCode("es")]
    assert sd.target_texts == ["I don't speak Spanish, ", "pero hablo inglés."]
    assert sd.target_texts is not None
    assert "".join(sd.target_texts) == sd.text


# --- One-way translation ----------------------------------------------------


async def test_final_transcript_one_way_translation_single_target_run():
    """One-way translation produces a single-entry `target_languages` list."""
    from livekit.plugins.soniox.stt import TranslationConfig

    stream = _make_stream(translation=TranslationConfig(type="one_way", target_language="es"))

    messages = [
        {
            "tokens": [
                _final_token("Hello world.", "en", translation_status="original"),
                _final_token("Hola mundo.", "es", translation_status="translation"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 500,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    sd = final.alternatives[0]

    assert sd.text == "Hola mundo."
    assert sd.language == LanguageCode("es")
    assert sd.source_languages == [LanguageCode("en")]
    assert sd.source_texts == ["Hello world."]
    assert sd.target_languages == [LanguageCode("es")]
    assert sd.target_texts == ["Hola mundo."]


# --- "none" untranslated chunk ---------------------------------------------


async def test_final_transcript_untranslated_chunk_yields_asymmetric_lists():
    """A `translation_status: "none"` token sits in source-only; the resulting
    source and target lists legitimately have different lengths."""
    from livekit.plugins.soniox.stt import TranslationConfig

    stream = _make_stream(
        translation=TranslationConfig(type="two_way", language_a="en", language_b="de")
    )

    messages = [
        {
            "tokens": [
                _final_token("Good morning. ", "en", translation_status="original"),
                _final_token("Bonjour à tous. ", "fr", translation_status="none"),
                _final_token("How are you?", "en", translation_status="original"),
                _final_token("Guten Morgen. ", "de", translation_status="translation"),
                _final_token("Wie geht's?", "de", translation_status="translation"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 1200,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    sd = final.alternatives[0]

    # The "Bonjour à tous." fr chunk sits between two en chunks, so the source
    # per-run breakdown is three entries (en, fr, en) — the same coalescing
    # behavior as the target side, just driven by different `language` runs.
    assert sd.source_languages == [
        LanguageCode("en"),
        LanguageCode("fr"),
        LanguageCode("en"),
    ]
    assert sd.source_texts == [
        "Good morning. ",
        "Bonjour à tous. ",
        "How are you?",
    ]

    # Both translation tokens are German → single coalesced target run.
    assert sd.target_languages == [LanguageCode("de")]
    assert sd.target_texts == ["Guten Morgen. Wie geht's?"]

    # The two per-run lists are independent — different lengths are legitimate.
    assert sd.source_languages is not None
    assert sd.target_languages is not None
    assert len(sd.source_languages) != len(sd.target_languages)


# --- Interim transcript shape ----------------------------------------------


async def test_interim_transcript_merges_final_and_non_final_per_run():
    """Interim path merges final-so-far with current non-final tokens for
    both source and target sides via `_merge_lang_segments`."""
    from livekit.plugins.soniox.stt import TranslationConfig

    stream = _make_stream(
        translation=TranslationConfig(type="two_way", language_a="en", language_b="es")
    )

    messages = [
        {
            "tokens": [
                _final_token("Hola, ", "es", translation_status="original"),
                _final_token("Hello, ", "en", translation_status="translation"),
                _nonfinal_token("¿cómo estás?", "es", translation_status="original"),
                _nonfinal_token("how are you?", "en", translation_status="translation"),
            ],
            "total_audio_proc_ms": 800,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=2)

    interim = next(
        e
        for e in events
        if e.type in (SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.PREFLIGHT_TRANSCRIPT)
    )
    sd = interim.alternatives[0]

    assert sd.source_languages == [LanguageCode("es")]
    assert sd.source_texts == ["Hola, ¿cómo estás?"]
    assert sd.target_languages == [LanguageCode("en")]
    assert sd.target_texts == ["Hello, how are you?"]


async def test_interim_transcript_no_translation_populates_source_runs():
    """Interim path also surfaces per-run `source_*` in non-translation mode,
    same symmetric fix as the final-transcript case."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _final_token("こんにちは、", "ja"),
                _nonfinal_token("My name is Sam.", "en"),
            ],
            "total_audio_proc_ms": 600,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=2)
    interim = next(
        e
        for e in events
        if e.type in (SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.PREFLIGHT_TRANSCRIPT)
    )
    sd = interim.alternatives[0]

    assert sd.source_languages == [LanguageCode("ja"), LanguageCode("en")]
    assert sd.source_texts == ["こんにちは、", "My name is Sam."]
    assert sd.target_languages is None
    assert sd.target_texts is None


async def test_recv_messages_raises_on_server_error_frame():
    stream = _make_stream(translation=None)
    stream._ws = _FakeWebSocket(
        [
            {
                "error_code": 401,
                "error_message": "Incorrect API key provided",
                "total_audio_proc_ms": 0,
            }
        ]
    )

    with pytest.raises(APIStatusError) as exc_info:
        await asyncio.wait_for(stream._recv_messages_task(), timeout=1.0)

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    assert exc_info.value.body is not None


# --- Non-translation mode -------------------------------------------------


async def test_final_transcript_no_translation_single_language_populates_source():
    """Non-translation, single-language input: `source_*` is populated from the
    per-run breakdown (single entry), `target_*` is None. This matches the
    `SpeechData` docstring's "multi-language detection services" semantics and
    is the same fix-symptom as translation mode — `final._lang_segments` was
    being discarded when `final_original` was empty."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _final_token("Hello world.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 500,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    sd = final.alternatives[0]

    assert sd.text == "Hello world."
    assert sd.language == LanguageCode("en")
    assert sd.source_languages == [LanguageCode("en")]
    assert sd.source_texts == ["Hello world."]
    assert sd.target_languages is None
    assert sd.target_texts is None


async def test_final_transcript_no_translation_code_switched_populates_source_runs():
    """Non-translation, code-switched input: `source_*` carries the per-run
    breakdown across detected languages. Mirrors the follow-up issue comment's
    JA+EN example — same plugin gap as the original target-side symptom, just
    surfaced through the source side when translation is off."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _final_token("こんにちは、君の名前は何だ。", "ja"),
                _final_token(" My name is Sam.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 1500,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    sd = final.alternatives[0]

    assert sd.text == "こんにちは、君の名前は何だ。 My name is Sam."
    # `sd.language` is the plugin's opinionated lossy summary (currently
    # most-chars-wins) and isn't what this test is exercising -- the per-run
    # `source_languages` / `source_texts` are. Don't assert on `language`
    # here so the test stays decoupled from upstream's summary heuristic.
    assert sd.source_languages == [LanguageCode("ja"), LanguageCode("en")]
    assert sd.source_texts == [
        "こんにちは、君の名前は何だ。",
        " My name is Sam.",
    ]
    assert sd.target_languages is None
    assert sd.target_texts is None
    assert sd.source_texts is not None
    assert "".join(sd.source_texts) == sd.text


# ---------------------------------------------------------------------------
# Per-word confidence / timing on `SpeechData.words`
# ---------------------------------------------------------------------------


def test_token_accumulator_groups_subword_tokens_into_words():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update(
        {"text": "My", "is_final": True, "confidence": 0.9, "start_ms": 0, "end_ms": 100}
    )
    accumulator.update(
        {"text": " walk", "is_final": True, "confidence": 0.4, "start_ms": 100, "end_ms": 250}
    )
    accumulator.update(
        {"text": "ing", "is_final": True, "confidence": 0.6, "start_ms": 250, "end_ms": 400}
    )

    words = accumulator.timed_words()
    assert words is not None
    assert [str(word) for word in words] == ["My", "walking"]
    assert words[0].confidence == pytest.approx(0.9)
    assert words[1].confidence == pytest.approx(0.5)  # mean of "walk" + "ing"
    assert words[1].start_time == pytest.approx(0.1)
    assert words[1].end_time == pytest.approx(0.4)


def test_token_accumulator_words_apply_start_time_offset():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update(
        {"text": "hi", "is_final": True, "confidence": 0.8, "start_ms": 500, "end_ms": 700}
    )

    words = accumulator.timed_words(start_time_offset=10.0)
    assert words is not None
    assert words[0].start_time == pytest.approx(10.5)
    assert words[0].end_time == pytest.approx(10.7)
    assert words[0].start_time_offset == pytest.approx(10.0)


def test_token_accumulator_unspaced_script_tokens_become_separate_words():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update({"text": "你好", "is_final": True, "confidence": 0.9})
    accumulator.update({"text": "世界", "is_final": True, "confidence": 0.7})

    words = accumulator.timed_words()
    assert words is not None
    assert [str(word) for word in words] == ["你好", "世界"]
    assert words[0].confidence == pytest.approx(0.9)
    assert words[1].confidence == pytest.approx(0.7)


def test_token_accumulator_word_without_confidence_is_not_given():
    from livekit.agents.types import NOT_GIVEN
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update({"text": "hello", "is_final": True})

    words = accumulator.timed_words()
    assert words is not None
    assert words[0].confidence is NOT_GIVEN
    assert words[0].start_time is NOT_GIVEN
    assert words[0].end_time is NOT_GIVEN


def test_token_accumulator_reset_clears_words():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update({"text": "hello world", "is_final": True, "confidence": 0.9})
    accumulator.reset()

    assert accumulator.timed_words() is None


def test_merged_speech_data_concatenates_final_and_nonfinal_words():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    final = _TokenAccumulator()
    final.update({"text": "Hello", "is_final": True, "confidence": 0.9})
    non_final = _TokenAccumulator()
    non_final.update({"text": " wor", "is_final": False, "confidence": 0.5})

    sd = final.merged_speech_data(non_final)
    assert sd.words is not None
    assert [str(word) for word in sd.words] == ["Hello", "wor"]


async def test_final_transcript_carries_word_confidences():
    """End-to-end: sub-word tokens with confidence produce `SpeechData.words`
    grouped at whitespace boundaries, alongside the existing scalar mean."""
    stream = _make_stream()

    messages = [
        {
            "tokens": [
                {
                    "text": "Hel",
                    "language": "en",
                    "is_final": True,
                    "confidence": 0.8,
                    "start_ms": 0,
                    "end_ms": 80,
                },
                {
                    "text": "lo",
                    "language": "en",
                    "is_final": True,
                    "confidence": 0.6,
                    "start_ms": 80,
                    "end_ms": 150,
                },
                {
                    "text": " world.",
                    "language": "en",
                    "is_final": True,
                    "confidence": 0.9,
                    "start_ms": 150,
                    "end_ms": 400,
                },
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 500,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    final = next(e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT)
    sd = final.alternatives[0]

    assert sd.text == "Hello world."
    assert sd.confidence == pytest.approx((0.8 + 0.6 + 0.9) / 3)

    assert sd.words is not None
    assert [str(word) for word in sd.words] == ["Hello", "world."]
    assert sd.words[0].confidence == pytest.approx(0.7)  # mean of "Hel" + "lo"
    assert sd.words[1].confidence == pytest.approx(0.9)
    assert sd.words[0].start_time == pytest.approx(0.0)
    assert sd.words[0].end_time == pytest.approx(0.15)
    assert sd.words[1].start_time == pytest.approx(0.15)
    assert sd.words[1].end_time == pytest.approx(0.4)


def test_token_accumulator_thai_tokens_become_separate_words():
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update({"text": "สวัสดี", "is_final": True, "confidence": 0.8})
    accumulator.update({"text": "ครับ", "is_final": True, "confidence": 0.6})

    words = accumulator.timed_words()
    assert words is not None
    assert [str(word) for word in words] == ["สวัสดี", "ครับ"]
