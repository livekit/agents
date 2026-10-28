"""Soniox protocol regressions through public streams and a local WebSocket peer."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIStatusError, LanguageCode
from livekit.agents.stt import SpeechEventType
from livekit.plugins.soniox import STT, STTOptions

pytestmark = pytest.mark.plugin("soniox")
END_TOKEN_FINAL: dict[str, Any] = {"text": "<end>", "is_final": True}


def _make_stream(translation: Any = None) -> STT:
    return STT(api_key="test-key", params=STTOptions(translation=translation))


async def _drive_recv(model: STT, messages: list[dict[str, Any]], *, expect_events: int):
    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()  # initial provider configuration
        for message in messages:
            await ws.send_json(message)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", peer)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        # The application supplies a transport and URL, then consumes the public stream.
        model._base_url = str(server.make_url("/transcribe"))
        model._http_session = session
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            return [await asyncio.wait_for(anext(stream), 2) for _ in range(expect_events)]
        finally:
            await stream.aclose()
            await model.aclose()


@pytest.mark.parametrize("level", [None, 0, 2])
async def test_authentication_and_endpoint_options_reach_the_peer(level: int | None) -> None:
    captured = {}
    connected = asyncio.Event()

    async def peer(request: web.Request) -> web.WebSocketResponse:
        captured["authorization"] = request.headers.get("Authorization")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        captured["config"] = await ws.receive_json()
        connected.set()
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", peer)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        model = STT(
            api_key="test-key",
            base_url=str(server.make_url("/transcribe")),
            http_session=session,
            params=STTOptions(endpoint_latency_adjustment_level=level),
        )
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            await asyncio.wait_for(connected.wait(), 2)
            assert captured["authorization"] == "Bearer test-key"
            config = captured["config"]
            assert "api_key" not in config
            if level is None:
                assert "endpoint_latency_adjustment_level" not in config
            else:
                assert config["endpoint_latency_adjustment_level"] == level
        finally:
            await stream.aclose()
            await model.aclose()


async def test_authentication_error_surfaces_without_retry() -> None:
    connections = 0

    async def peer(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({"error_code": 401, "error_message": "invalid key"})
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", peer)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        model = STT(
            api_key="test-key", base_url=str(server.make_url("/transcribe")), http_session=session
        )
        stream = model.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
        try:
            with pytest.raises(APIStatusError) as raised:
                await asyncio.wait_for(anext(stream), 2)
            assert raised.value.status_code == 401
            assert not raised.value.retryable
            assert connections == 1
        finally:
            await stream.aclose()
            await model.aclose()


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
