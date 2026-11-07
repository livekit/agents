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
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.language import LanguageCode
from livekit.agents.stt import SpeechData, SpeechEventType
from livekit.agents.types import TimedString

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


def test_token_accumulator_span_covers_all_timed_tokens():
    """A token that carries timing late must not collapse the span onto itself.

    Regression for #6885: when the leading tokens of an utterance arrive
    without ``start_ms``/``end_ms`` and only a trailing token has them, the
    emitted span used to collapse onto that trailing token (e.g. two words in
    180ms).
    """
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update({"text": " He's", "language": "en", "is_final": True})
    accumulator.update({"text": " trying", "language": "en", "is_final": True})
    accumulator.update(
        {"text": ".", "language": "en", "is_final": True, "start_ms": 43663, "end_ms": 43843}
    )

    assert accumulator.text == " He's trying."
    assert accumulator.start_time == 43663
    assert accumulator.end_time == 43843


def test_token_accumulator_end_time_never_regresses():
    """A trailing token whose ``end_ms`` regresses must not pull the end back.

    Regression for #6885: ``end_time`` was overwritten by every token that
    carried ``end_ms``, so a non-monotonic trailing token shortened the span.
    """
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update(
        {"text": " He's", "language": "en", "is_final": True, "start_ms": 41000, "end_ms": 41400}
    )
    accumulator.update(
        {"text": " trying", "language": "en", "is_final": True, "start_ms": 41400, "end_ms": 43843}
    )
    accumulator.update(
        {"text": ".", "language": "en", "is_final": True, "start_ms": 43663, "end_ms": 43700}
    )

    assert accumulator.start_time == 41000
    assert accumulator.end_time == 43843


def test_token_accumulator_start_time_takes_earliest():
    """The earliest ``start_ms`` wins even if a later token carries an earlier one."""
    from livekit.plugins.soniox.stt import _TokenAccumulator

    accumulator = _TokenAccumulator()
    accumulator.update(
        {"text": " He's", "language": "en", "is_final": True, "start_ms": 41000, "end_ms": 41400}
    )
    accumulator.update(
        {"text": " trying", "language": "en", "is_final": True, "start_ms": 40000, "end_ms": 43843}
    )

    assert accumulator.start_time == 40000
    assert accumulator.end_time == 43843


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

    async def receive(self) -> _FakeWSMessage:
        return await self.__anext__()

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


async def test_websocket_authenticates_with_header():
    stream = _make_stream()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.send_str = AsyncMock()
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    stream._stt._http_session = session

    await stream._connect_ws()

    assert session.ws_connect.call_args.kwargs["headers"] == {"Authorization": "Bearer test-key"}
    config = json.loads(ws.send_str.call_args.args[0])
    assert "api_key" not in config
    assert config["model"] == stream._stt._params.model


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

        async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
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


@pytest.mark.parametrize("offset", [0.0, 8.5])
async def test_eos_uses_final_token_end_time_and_resets_between_turns(offset):
    stream = _make_stream()
    stream.start_time_offset = offset
    events = await _drive_recv(
        stream,
        [
            {
                "tokens": [
                    {**_final_token("hello", "en"), "start_ms": 1000, "end_ms": 2400},
                    END_TOKEN_FINAL,
                    _final_token("again", "en"),
                    END_TOKEN_FINAL,
                ],
            }
        ],
        expect_events=4,
    )

    endpoints = [event for event in events if event.type == SpeechEventType.END_OF_SPEECH]
    assert len(endpoints) == 2
    assert endpoints[0].alternatives[0].end_time == pytest.approx(offset + 2.4)
    assert endpoints[1].alternatives == []


@pytest.mark.parametrize("source_end, translation_end", [(2400, None), (2400, 3000), (None, 2400)])
async def test_eos_prefers_source_token_timing_for_translation(source_end, translation_end):
    from livekit.plugins.soniox.stt import TranslationConfig

    stream = _make_stream(translation=TranslationConfig(type="one_way", target_language="en"))
    stream.start_time_offset = 8.5
    events = await _drive_recv(
        stream,
        [
            {
                "tokens": [
                    {
                        **_final_token("hola", "es", "original"),
                        **({"start_ms": 1000, "end_ms": source_end} if source_end else {}),
                    },
                    {
                        **_final_token("hello", "en", "translation"),
                        **(
                            {"start_ms": 1000, "end_ms": translation_end} if translation_end else {}
                        ),
                    },
                    END_TOKEN_FINAL,
                ],
            }
        ],
        expect_events=2,
    )

    assert events[-1].type == SpeechEventType.END_OF_SPEECH
    assert events[-1].alternatives[0].end_time == pytest.approx(10.9)


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


# --- RECOGNITION_USAGE: output tokens ---------------------------------------


async def test_usage_event_reports_final_tokens_as_output_tokens():
    """Each final transcription token counts once toward `output_tokens`;
    the `<end>` control token does not."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _final_token("Hello", "en"),
                _final_token(" world.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 500,
        }
    ]

    events = await _drive_recv(stream, messages, expect_events=3)
    usage = next(e for e in events if e.type == SpeechEventType.RECOGNITION_USAGE)

    assert usage.recognition_usage is not None
    assert usage.recognition_usage.audio_duration == 0.5
    assert usage.recognition_usage.output_tokens == 2


async def test_usage_event_excludes_non_final_tokens():
    """Non-final tokens are re-sent as final later — counting them would
    double-bill. Only the final occurrences count."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _nonfinal_token("Hel", "en"),
                _nonfinal_token("lo", "en"),
            ],
            "total_audio_proc_ms": 300,
        },
        {
            "tokens": [
                _final_token("Hello", "en"),
                _final_token(" world.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 800,
        },
    ]

    # Both messages report usage; only the final tokens contribute output tokens.
    events = await _drive_recv(stream, messages, expect_events=6)
    usages = [e.recognition_usage for e in events if e.type == SpeechEventType.RECOGNITION_USAGE]
    assert [usage.output_tokens for usage in usages if usage is not None] == [0, 2]


async def test_usage_events_report_incremental_output_tokens():
    """A second endpoint reports only the tokens accumulated since the
    previous usage event, mirroring the incremental audio_duration."""
    stream = _make_stream(translation=None)

    messages = [
        {
            "tokens": [
                _final_token("First.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 1000,
        },
        {
            "tokens": [
                _final_token("Second", "en"),
                _final_token(" utterance.", "en"),
                END_TOKEN_FINAL,
            ],
            "total_audio_proc_ms": 2500,
        },
    ]

    # Per endpoint: START_OF_SPEECH is not emitted before endpoint flush
    # (tokens are final-only), so each message yields FINAL + END_OF_SPEECH + USAGE.
    events = await _drive_recv(stream, messages, expect_events=6)
    usages = [e for e in events if e.type == SpeechEventType.RECOGNITION_USAGE]

    assert len(usages) == 2
    assert usages[0].recognition_usage is not None
    assert usages[0].recognition_usage.audio_duration == 1.0
    assert usages[0].recognition_usage.output_tokens == 1
    assert usages[1].recognition_usage is not None
    assert usages[1].recognition_usage.audio_duration == 1.5
    assert usages[1].recognition_usage.output_tokens == 2


async def test_usage_event_counts_translation_tokens():
    """In translation mode both original and translated final tokens are
    provider output and count toward `output_tokens`."""
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
    usage = next(e for e in events if e.type == SpeechEventType.RECOGNITION_USAGE)

    assert usage.recognition_usage is not None
    assert usage.recognition_usage.output_tokens == 2


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


async def test_endpoint_after_full_retraction_emits_end_of_speech():
    """Regression for https://github.com/livekit/agents/issues/7728.

    A segment whose tokens were all retracted (non-final tokens are not
    retained across websocket messages) must still be closed by the endpoint:
    the `<end>` hits the `else` branch of `send_endpoint_transcript`, which
    previously only reset buffers. With `turn_detection="stt"` the session was
    left stuck in `speaking` — no `user_away_timeout`, no silence re-prompts,
    and the next utterance never got its own `START_OF_SPEECH`.
    """
    stream = _make_stream(translation=None)

    messages = [
        {
            # control: a normal utterance finalises and emits END_OF_SPEECH.
            "tokens": [_final_token(" 42", "en"), END_TOKEN_FINAL],
            "total_audio_proc_ms": 400,
        },
        {
            # the bug: a non-final token is sent, then retracted — the next
            # message carries only the endpoint with no final text.
            "tokens": [_nonfinal_token(" 42", "en")],
            "total_audio_proc_ms": 500,
        },
        {
            "tokens": [END_TOKEN_FINAL],
            "total_audio_proc_ms": 600,
        },
    ]

    # Event cadence is deterministic here: 5 speech events + one
    # RECOGNITION_USAGE per increase of total_audio_proc_ms (3 messages = 3).
    events = await _drive_recv(stream, messages, expect_events=8)
    # RECOGNITION_USAGE rides the same channel per audio cadence — filter it;
    # we only assert the speech-event sequence.
    events = [e for e in events if e.type != SpeechEventType.RECOGNITION_USAGE]

    assert [e.type for e in events] == [
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    # The retraction endpoint must NOT carry a transcript (nothing was said).
    assert events[-1].alternatives == []

    # A fresh utterance after the retraction gets its own START_OF_SPEECH.
    stream2 = _make_stream(translation=None)
    messages2 = [
        {"tokens": [_nonfinal_token(" 42", "en")], "total_audio_proc_ms": 100},
        {"tokens": [END_TOKEN_FINAL], "total_audio_proc_ms": 200},
        {"tokens": [_final_token(" Halo.", "en"), END_TOKEN_FINAL], "total_audio_proc_ms": 900},
    ]
    events2 = await _drive_recv(stream2, messages2, expect_events=8)
    events2 = [e for e in events2 if e.type != SpeechEventType.RECOGNITION_USAGE]
    assert [e.type for e in events2] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events2[3].alternatives[0].text == " Halo."


# --- RECOGNITION_USAGE reporting cadence -----------------------------------


async def test_recognition_usage_reported_for_silent_frames():
    """Frames that carry no tokens still report the audio Soniox processed.

    The regression case: a silent speaker produces no endpoint token, and a
    cancelled stream never reads the `finished` frame, so gating on those two
    reported nothing at all.
    """
    stream = _make_stream(translation=None)

    events = await _drive_recv(
        stream,
        [{"tokens": [], "total_audio_proc_ms": 3000}],
        expect_events=1,
    )

    assert [e.type for e in events] == [SpeechEventType.RECOGNITION_USAGE]
    assert events[0].recognition_usage is not None
    assert events[0].recognition_usage.audio_duration == pytest.approx(3.0)


async def test_recognition_usage_reported_without_endpoint_token():
    """Interim-only frames report usage too, ordered after the transcript events."""
    stream = _make_stream(translation=None)

    events = await _drive_recv(
        stream,
        [{"tokens": [_nonfinal_token("hola", "es")], "total_audio_proc_ms": 2500}],
        expect_events=3,
    )

    assert [e.type for e in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[-1].recognition_usage is not None
    assert events[-1].recognition_usage.audio_duration == pytest.approx(2.5)


async def test_recognition_usage_reports_delta_and_skips_unchanged_totals():
    """Usage is the delta against what was already reported, so the running
    total stays monotonic and an unchanged total emits nothing."""
    stream = _make_stream(translation=None)

    events = await _drive_recv(
        stream,
        [
            {"tokens": [], "total_audio_proc_ms": 3000},
            # Same total: already accounted for, so no event.
            {"tokens": [], "total_audio_proc_ms": 3000},
            {"tokens": [], "total_audio_proc_ms": 5000},
        ],
        expect_events=2,
    )

    assert [e.type for e in events] == [SpeechEventType.RECOGNITION_USAGE] * 2
    durations = [e.recognition_usage.audio_duration for e in events if e.recognition_usage]
    assert durations == pytest.approx([3.0, 2.0])
    assert sum(durations) == pytest.approx(5.0)


async def test_recognition_usage_still_reported_on_endpoint_frame():
    """The endpoint path keeps reporting usage, after the transcript events."""
    stream = _make_stream(translation=None)

    events = await _drive_recv(
        stream,
        [
            {
                "tokens": [_final_token("Hello world.", "en"), END_TOKEN_FINAL],
                "total_audio_proc_ms": 1500,
            }
        ],
        expect_events=3,
    )

    assert [e.type for e in events] == [
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[-1].recognition_usage is not None
    assert events[-1].recognition_usage.audio_duration == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Half-open socket detection (same fix as #7206 for Deepgram v1)
# ---------------------------------------------------------------------------


async def test_socket_exposes_control_frames_to_monitor():
    """Control frames must reach the shared monitor to reset its silence deadline."""
    stream = _make_stream()

    class RecordingWebSocket:
        async def send_str(self, message: str) -> None:
            pass

    class RecordingSession:
        kwargs: dict[str, Any] | None = None

        async def ws_connect(self, url: str, **kwargs: Any) -> RecordingWebSocket:
            self.kwargs = kwargs
            return RecordingWebSocket()

    session = RecordingSession()
    stream._stt._http_session = session

    await stream._connect_ws()

    assert session.kwargs is not None
    assert session.kwargs.get("autoping") is False


@pytest.mark.parametrize("task_name", ["_keepalive_task", "_send_audio_task"])
async def test_write_drop_uses_retry_budget(task_name):
    """The keepalive used to catch every exception and return. When no audio is
    flowing that write is the only thing touching the socket, so swallowing it
    left the stream on a dead connection with nothing ever logged."""
    stream = _make_stream()

    class DeadWebSocket:
        closed = False

        async def send_str(self, message: str) -> None:
            raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")

        async def send_bytes(self, message: bytes) -> None:
            raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")

    stream._ws = DeadWebSocket()
    stream.audio_queue.put_nowait(b"audio")

    with pytest.raises(APIConnectionError) as exc_info:
        await asyncio.wait_for(getattr(stream, task_name)(), timeout=1.0)
    assert exc_info.value.retryable
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionResetError)


async def test_closed_socket_reconnect_exhausts_retry_budget(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.soniox import STT
    from livekit.plugins.soniox.stt import SpeechStream

    connections = 0

    class ClosedWebSocket:
        closed = True

        async def close(self):
            pass

        def exception(self):
            return aiohttp.ServerTimeoutError("liveness timeout")

    async def connect(self):
        nonlocal connections
        connections += 1
        return ClosedWebSocket()

    async def idle(self):
        await asyncio.Future()

    async def request_reconnect(self):
        self._reconnect_event.set()

    monkeypatch.setattr(SpeechStream, "_connect_ws", connect)
    monkeypatch.setattr(SpeechStream, "_prepare_audio_task", idle)
    monkeypatch.setattr(SpeechStream, "_send_audio_task", idle)
    monkeypatch.setattr(SpeechStream, "_keepalive_task", idle)
    monkeypatch.setattr(SpeechStream, "_recv_messages_task", request_reconnect)
    instance = STT(api_key="test-key")
    stream = instance.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    try:
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._task, 1)
        assert connections == 2
    finally:
        await stream.aclose()


async def test_error_frame_ends_recv_loop_and_keeps_the_cause(caplog):
    """A heartbeat timeout arrives as WSMsgType.ERROR, not as a close frame.
    Logging it as an unexpected type and continuing only worked because aiohttp
    happens to report CLOSED next, and it threw away the one value that says
    why the socket went."""
    stream = _make_stream()

    class HeartbeatTimeoutWebSocket:
        closed = False

        def __init__(self) -> None:
            self.receives = 0

        def exception(self) -> BaseException:
            return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

        def __aiter__(self) -> HeartbeatTimeoutWebSocket:
            return self

        async def receive(self) -> _FakeWSMessage:
            return await self.__anext__()

        async def __anext__(self) -> _FakeWSMessage:
            self.receives += 1
            # yield, so a recv loop that steps over the error instead of ending
            # fails the receive-count assertion rather than starving the loop
            await asyncio.sleep(0)
            if self.receives > 5:
                raise StopAsyncIteration
            return _FakeWSMessage("", msg_type=aiohttp.WSMsgType.ERROR)

    ws = HeartbeatTimeoutWebSocket()
    stream._ws = ws

    with caplog.at_level("WARNING", logger="livekit.plugins.soniox"):
        await asyncio.wait_for(stream._recv_messages_task(), timeout=1.0)

    assert ws.receives == 1
    assert stream._reconnect_event.is_set()
    assert any("No PONG received" in r.getMessage() for r in caplog.records)


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.soniox.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, stream._reconnect_event.set)
    finally:
        await stream.aclose()
        await provider.aclose()


@pytest.mark.parametrize(
    ("agent_text_input", "echo_cancellation", "requested"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
async def test_agent_text_input_websocket_config(
    agent_text_input: bool, echo_cancellation: bool, requested: bool
):
    from livekit.plugins.soniox import STTOptions

    stream = _make_stream()
    stream._stt._params = STTOptions(
        enable_agent_text_input=agent_text_input, enable_echo_cancellation=echo_cancellation
    )
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.send_str = AsyncMock()
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    stream._stt._http_session = session

    await stream._connect_ws()

    config = json.loads(ws.send_str.call_args.args[0])
    assert ("enable_agent_text_input" in config) is requested


def test_agent_text_input_advertises_chat_context():
    from livekit.plugins.soniox import STT, STTOptions

    assert not STT(api_key="test-key").capabilities.chat_context
    stt_instance = STT(api_key="test-key", params=STTOptions(enable_agent_text_input=True))
    assert stt_instance.capabilities.chat_context
    # with echo cancellation each reply is already sent while it plays
    stt_instance = STT(
        api_key="test-key",
        params=STTOptions(enable_agent_text_input=True, enable_echo_cancellation=True),
    )
    assert not stt_instance.capabilities.chat_context


async def test_assistant_reply_is_queued_as_agent_text():
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent
    from livekit.plugins.soniox import STT, STTOptions

    stt_instance = STT(api_key="test-key", params=STTOptions(enable_agent_text_input=True))

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt_instance.stream()

    stt_instance._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="user", content=["lecadol please"]))
    )
    stt_instance._push_conversation_item(
        ConversationItemAddedEvent(
            item=ChatMessage(role="assistant", content=["Sure, Lekadol 500 mg."])
        )
    )

    assert stream.audio_queue.qsize() == 1
    assert {"type": "agent_text", **stream.audio_queue.get_nowait()._asdict()} == {
        "type": "agent_text",
        "text": "Sure, Lekadol 500 mg.",
        "end_of_segment": True,
    }


async def test_agent_text_output_sends_played_text_and_ends_the_segment_when_playback_ends():
    from livekit.plugins.soniox import STT, STTOptions

    stt_instance = STT(api_key="test-key", params=STTOptions(enable_echo_cancellation=True))

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt_instance.stream()

    from .fake_io import FakeAudioOutput, FakeTextOutput

    next_in_chain = FakeTextOutput()
    output = stt_instance.agent_text_output(FakeAudioOutput(), next_in_chain=next_in_chain)

    output.flush()  # nothing played yet, so no segment to end
    await output.capture_text("Hello ")
    await output.capture_text("there.")
    output.flush()
    _played_out(output, position=1.0)

    sent = _sent(stream)
    assert sent == [
        {"type": "agent_text", "text": "Hello ", "end_of_segment": False},
        {"type": "agent_text", "text": "there.", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]
    assert "".join(message["text"] for message in sent) == "Hello there."
    assert next_in_chain._messages == ["", "Hello there."]


async def test_agent_text_output_sends_only_what_the_voice_says():
    from livekit.plugins.soniox import STT, STTOptions

    stt_instance = STT(api_key="test-key", params=STTOptions(enable_echo_cancellation=True))

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt_instance.stream()

    from .fake_io import FakeAudioOutput

    output = stt_instance.agent_text_output(FakeAudioOutput())

    # A bracket cue is performed, not said, and can be split across chunks;
    # Soniox TTS reads XML aloud, so that is sent.
    for chunk in ["Right. [clears ", "throat] <emphasis>Anyway</emphasis>."]:
        await output.capture_text(chunk)
    output.flush()
    _played_out(output, position=1.0)

    sent = _sent(stream)
    assert "".join(message["text"] for message in sent) == "Right. <emphasis>Anyway</emphasis>."
    assert sent[-1]["end_of_segment"] is True


def _agent_text_output(lead_ms: int = 700, transform=None):
    """An agent text output with one Soniox stream behind it, and its audio output."""
    from livekit.plugins.soniox import STT, STTOptions

    from .fake_io import FakeAudioOutput

    stt_instance = STT(
        api_key="test-key",
        params=STTOptions(enable_echo_cancellation=True, agent_text_lead_ms=lead_ms),
    )

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt_instance.stream()

    audio_output = FakeAudioOutput()
    kwargs = {"transform": transform} if transform else {}
    return stt_instance.agent_text_output(audio_output, **kwargs), stream, audio_output


def _heard(audio_output, seconds: float, *, offset: float = 0.0) -> None:
    """Report a run of played audio, as the audio output does once it cannot be discarded.

    `offset` is where the run starts in the reply, which the room output sets to
    the end of the run before it; `seconds` is how long this run lasted.
    """
    audio_output.on_playback_progressed(
        started_at=time.time() - seconds, offset=offset, duration=seconds
    )


def _played_out(output, *, position: float, interrupted: bool = False) -> None:
    """Report that the reply's playback finished at `position`, as the audio output does."""
    from livekit.agents.voice.io import PlaybackFinishedEvent

    output._audio_output.emit(
        "playback_finished",
        PlaybackFinishedEvent(playback_position=position, interrupted=interrupted),
    )


def _sent(stream) -> list[dict[str, Any]]:
    return [
        {"type": "agent_text", **stream.audio_queue.get_nowait()._asdict()}
        for _ in range(stream.audio_queue.qsize())
    ]


async def test_agent_text_is_held_until_the_lead_before_it_is_heard() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=700)

    _heard(audio_output, 0.1)
    # Spoken at 1.5s, so with a 0.7s lead it is not due at 0.1s.
    await output.capture_text(TimedString("later ", start_time=1.5, end_time=1.8))
    assert _sent(stream) == []

    _heard(audio_output, 0.9)
    output._send_due()

    assert [message["text"] for message in _sent(stream)] == ["later "]


@pytest.mark.virtual_time
async def test_agent_text_goes_out_early_when_playback_turns_out_to_be_ahead() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=700)
    _heard(audio_output, 0.0)

    # Spoken at 1.5s, so on the estimate at hand it is 0.8s away.
    await output.capture_text(TimedString("later ", start_time=1.5, end_time=1.8))
    assert _sent(stream) == []

    await asyncio.sleep(0.05)
    # The audio output reports a run: a whole second has played, which it only
    # says once the audio cannot be discarded, so the word is due right now.
    _heard(audio_output, 1.0)
    await asyncio.sleep(0.05)

    # Waiting out the original 0.8s would put it after the caller heard it.
    assert [message["text"] for message in _sent(stream)] == ["later "]


@pytest.mark.virtual_time
async def test_agent_text_waits_for_the_word_itself_not_for_a_fixed_interval() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)
    _heard(audio_output, 0.0)

    # Due at 0.22s, which no multiple of a 50ms interval lands on: polling would
    # hold it until 0.25s, while waiting for the word itself sends it at 0.22s.
    await output.capture_text(TimedString("soon ", start_time=0.22, end_time=0.4))
    assert _sent(stream) == []

    await asyncio.sleep(0.23)

    assert [message["text"] for message in _sent(stream)] == ["soon "]


async def test_agent_text_sends_a_script_without_spaces_as_it_was_spoken() -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    # Japanese, one word per character, with the whitespace piece a dropped tag
    # would leave between two of them.
    for word in ["今", "日", " ", "は"]:
        await output.capture_text(TimedString(word, start_time=0.0, end_time=0.1))

    # Soniox joins the pieces exactly as sent, so nothing may be added between
    # characters of a script that has no spaces, and the lone space rides with
    # the character that follows it.
    assert [message["text"] for message in _sent(stream)] == ["今", "日", " は"]


async def test_agent_text_counts_a_run_from_where_the_one_before_it_ended() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    # A second run of played audio: the output reports where it starts in the
    # reply, not from zero, so 2.0s had already played before this run's 0.5s.
    _heard(audio_output, 0.5, offset=2.0)
    await output.capture_text(TimedString("heard ", start_time=2.4, end_time=2.6))
    await output.capture_text(TimedString("later", start_time=9.0, end_time=9.2))

    # Reading the run's own duration as the position would place playback at
    # 0.5s and hold back a word the caller has already heard.
    assert [message["text"] for message in _sent(stream)] == ["heard "]

    output.on_detached()


async def test_agent_text_drops_the_tail_the_caller_never_heard() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("Heard. ", start_time=0.0, end_time=0.4))
    await output.capture_text(TimedString("Cut off.", start_time=5.0, end_time=5.4))
    _heard(audio_output, 0.4)

    # The reply was interrupted at 0.4s: the rest was never played.
    output.flush()
    _played_out(output, position=0.4, interrupted=True)

    assert _sent(stream) == [
        {"type": "agent_text", "text": "Heard. ", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]


async def test_agent_text_warns_once_when_there_is_nothing_to_pace_against(caplog) -> None:
    output, stream, _ = _agent_text_output()

    await output.capture_text("No timestamps ")
    await output.capture_text("here.")
    output.flush()
    _played_out(output, position=1.0)

    sent = [message["text"] for message in _sent(stream)]
    assert sent == ["No timestamps ", "here.", ""]
    assert "".join(sent) == "No timestamps here."
    assert sum("no word timestamps" in record.message for record in caplog.records) == 1


async def test_agent_text_lead_cannot_be_negative() -> None:
    from livekit.plugins.soniox import STTOptions

    with pytest.raises(ValueError, match="agent_text_lead_ms"):
        STTOptions(agent_text_lead_ms=-1)


async def test_agent_text_keeps_two_turns_apart() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("First turn.", start_time=0.0, end_time=0.5))
    output.flush()
    _played_out(output, position=0.5)
    await output.capture_text(TimedString("Second turn.", start_time=0.0, end_time=0.5))
    output.flush()
    _played_out(output, position=0.5)

    assert _sent(stream) == [
        {"type": "agent_text", "text": "First turn.", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
        {"type": "agent_text", "text": "Second turn.", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]


async def test_agent_text_drops_a_tag_cut_off_by_an_interruption() -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    # The reply's text ends inside a tag, then the caller cuts in after it: the
    # text is flushed before playback reports the interruption, as LiveKit does.
    await output.capture_text(TimedString("Ha ", start_time=0.0, end_time=0.3))
    await output.capture_text(TimedString("[lau", start_time=0.5, end_time=0.5))
    output.flush()
    _played_out(output, position=1.0, interrupted=True)
    await output.capture_text(TimedString("Next.", start_time=0.0, end_time=0.3))

    assert _sent(stream) == [
        {"type": "agent_text", "text": "Ha ", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
        {"type": "agent_text", "text": "Next.", "end_of_segment": False},
    ]
    output.on_detached()


async def test_agent_text_paces_the_next_reply_after_one_that_never_played() -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("Never played.", start_time=4.0, end_time=4.4))
    output.flush()
    first = output._pacing
    await asyncio.sleep(0)  # pacing is under way
    await output.capture_text(TimedString("Later.", start_time=5.0, end_time=5.4))
    second = output._pacing
    # Let the cancelled task finish: it must not forget the one that replaced it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert first is not None and first.done()
    assert output._pacing is second and second is not None
    output.on_detached()
    await asyncio.sleep(0)
    assert second.done()


async def test_agent_text_does_not_blame_the_tts_for_untimed_text(caplog) -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("Timed ", start_time=2.0, end_time=2.4))
    await output.capture_text("untimed.")
    output.on_detached()

    assert not any("earlier than the previous one" in r.message for r in caplog.records)


async def test_agent_text_sends_the_space_a_dropped_tag_leaves_with_the_next_word() -> None:
    output, stream, _ = _agent_text_output(
        lead_ms=0, transform=lambda words, held, final=False: words
    )

    # What drop_bracket_cues leaves of "Sure, [laughs] Lekadol." in one call.
    for word in ["Sure,", " ", "Lekadol."]:
        await output.capture_text(TimedString(word, start_time=0.0, end_time=0.1))

    assert [message["text"] for message in _sent(stream)] == ["Sure,", " Lekadol."]


async def test_agent_text_uses_the_transform_it_is_given() -> None:
    output, stream, _ = _agent_text_output(
        lead_ms=0, transform=lambda words, held, final=False: words
    )

    await output.capture_text(TimedString("[laughs] ", start_time=0.0, end_time=0.1))

    assert [message["text"] for message in _sent(stream)] == ["[laughs] "]


async def test_agent_text_leaves_no_state_behind_when_detached() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=700)

    await output.capture_text(TimedString("later", start_time=9.0, end_time=9.5))
    assert output._pacing is not None

    output.on_detached()
    # Playback reports after detaching must not move the position, or a stale
    # output would keep sending into a session it no longer belongs to.
    _heard(audio_output, 9.0)

    assert output._pacing is None
    assert output._playback.seconds() == 0.0
    # The open segment is closed, and the word nobody heard is not sent.
    assert _sent(stream) == [{"type": "agent_text", "text": "", "end_of_segment": True}]


async def test_agent_text_keeps_the_order_the_words_arrived_in(caplog) -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    # A timestamp that goes backwards is the Text-to-Speech's mistake: the words
    # still go in the order they were handed over, and it is reported.
    await output.capture_text(TimedString("help ", start_time=0.5, end_time=0.8))
    await output.capture_text(TimedString("you ", start_time=0.5, end_time=0.8))
    await output.capture_text(TimedString("today?", start_time=0.1, end_time=0.4))
    assert _sent(stream) == []

    _heard(audio_output, 1.0)
    output.flush()
    _played_out(output, position=1.0)

    assert "".join(message["text"] for message in _sent(stream)) == "help you today?"
    assert sum("earlier than the previous one" in r.message for r in caplog.records) == 1


async def test_agent_text_keeps_sending_after_the_text_is_complete() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    # The Text-to-Speech finishes the reply long before it is heard.
    await output.capture_text(TimedString("Thanks ", start_time=0.0, end_time=0.4))
    await output.capture_text(TimedString("for calling.", start_time=3.0, end_time=3.6))
    output.flush()
    assert _sent(stream) == [{"type": "agent_text", "text": "Thanks ", "end_of_segment": False}]

    _heard(audio_output, 3.0)
    output._send_due()
    assert [message["text"] for message in _sent(stream)] == ["for calling."]

    _played_out(output, position=3.6)
    assert _sent(stream) == [{"type": "agent_text", "text": "", "end_of_segment": True}]


async def test_agent_text_sends_the_whole_reply_when_playback_finishes() -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("one ", start_time=0.0, end_time=0.4))
    await output.capture_text(TimedString("two", start_time=5.0, end_time=5.4))
    output.flush()
    # A sink that reports no progress: the end of playback is all there is.
    _played_out(output, position=5.4)

    assert [message["text"] for message in _sent(stream)] == ["one ", "two", ""]


async def test_agent_text_closes_a_reply_that_never_played_when_the_next_begins() -> None:
    output, stream, _ = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("Never ", start_time=0.0, end_time=0.4))
    await output.capture_text(TimedString("played.", start_time=4.0, end_time=4.4))
    output.flush()
    await output.capture_text(TimedString("Next.", start_time=0.0, end_time=0.4))

    assert _sent(stream) == [
        {"type": "agent_text", "text": "Never ", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
        {"type": "agent_text", "text": "Next.", "end_of_segment": False},
    ]


async def test_agent_text_paces_the_next_reply_from_its_own_start() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("A long first reply.", start_time=0.0, end_time=8.0))
    _heard(audio_output, 8.0)
    output.flush()
    _played_out(output, position=8.0)
    _sent(stream)

    # Heard 5s into the next reply, which has not started playing.
    await output.capture_text(TimedString("Later.", start_time=5.0, end_time=5.4))
    assert _sent(stream) == []
    output.on_detached()


async def test_agent_text_lead_follows_a_runtime_change() -> None:
    output, stream, audio_output = _agent_text_output(lead_ms=0)

    await output.capture_text(TimedString("Soon.", start_time=0.5, end_time=0.9))
    assert _sent(stream) == []

    output._stt._params.agent_text_lead_ms = 700
    output._send_due()
    assert [message["text"] for message in _sent(stream)] == ["Soon."]


async def test_assistant_reply_is_not_sent_twice_with_echo_cancellation() -> None:
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent

    output, stream, _ = _agent_text_output()
    output._stt._params.enable_agent_text_input = True

    output._stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["Sent already."]))
    )

    assert _sent(stream) == []


async def test_agent_text_queued_before_connecting_survives_the_connection() -> None:
    stream = _make_stream()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.send_str = AsyncMock()
    session = MagicMock(spec=aiohttp.ClientSession)
    session.ws_connect = AsyncMock(return_value=ws)
    stream._stt._http_session = session

    stream.audio_queue.put_nowait(b"stale audio")
    stream._push_agent_text("Hello.", end_of_segment=False)
    await stream._connect_ws()

    assert stream.audio_queue.qsize() == 1
    assert stream.audio_queue.get_nowait().text == "Hello."


async def test_agent_text_end_of_segment_is_not_sent_to_a_connection_that_never_saw_it() -> None:
    stream = _make_stream()
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.send_str = AsyncMock()
    stream._ws = ws

    # The segment opened on a connection that has since been replaced.
    stream._push_agent_text("", end_of_segment=True)
    stream._push_agent_text("Next.", end_of_segment=False)
    stream._push_agent_text("", end_of_segment=True)

    sender = asyncio.create_task(stream._send_audio_task())
    while not stream.audio_queue.empty():
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    sender.cancel()
    await asyncio.gather(sender, return_exceptions=True)

    assert [json.loads(call.args[0]) for call in ws.send_str.call_args_list] == [
        {"type": "agent_text", "text": "Next.", "end_of_segment": False},
        {"type": "agent_text", "text": "", "end_of_segment": True},
    ]
