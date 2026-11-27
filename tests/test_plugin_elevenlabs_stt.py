"""Unit tests for ElevenLabs STT plugin configuration."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict
from yarl import URL

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIStatusError,
    LanguageCode,
    stt,
    vad as vad_module,
)
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import stt as elevenlabs_stt
from livekit.plugins.elevenlabs._turn_detection import TurnDetection
from livekit.plugins.elevenlabs._utils import trace_id_from_headers

pytestmark = pytest.mark.plugin("elevenlabs")


class _EventSink:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


def _new_stream(
    *,
    server_vad=NOT_GIVEN,
    commit_strategy=NOT_GIVEN,
    language: str | None = "en",
    include_timestamps: bool = False,
    include_language_detection=NOT_GIVEN,
    secondary_languages: list[str] | Any = NOT_GIVEN,
) -> elevenlabs_stt.SpeechStream:
    # mirrors what STT.__init__ stores: both language options are normalized on the way in
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = elevenlabs_stt.STTOptions(
        model_id="scribe_v2_realtime",
        api_key="test-key",
        base_url=elevenlabs_stt.API_BASE_URL_V1,
        language_code=None,
        secondary_languages=[LanguageCode(code) for code in secondary_languages]
        if secondary_languages is not NOT_GIVEN
        else NOT_GIVEN,
        include_language_detection=include_language_detection,
        tag_audio_events=True,
        include_timestamps=include_timestamps,
        sample_rate=16000,
        server_vad=server_vad,
        commit_strategy=commit_strategy,
        keyterms=NOT_GIVEN,
        no_verbatim=False,
        enable_logging=True,
        previous_text=None,
    )
    stream._language = LanguageCode(language) if language else None
    stream._event_ch = _EventSink()
    stream._speaking = False
    stream._final_alternatives = []
    stream._vad = None
    stream._reconnect_event = asyncio.Event()
    stream._stt = elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime")
    stream._stt._opts = stream._opts
    stream._stt._streams.add(stream)
    stream._turn_detection = TurnDetection(
        opts=stream._opts, vad=None, on_end_of_speech=stream._emit_end_of_speech
    )
    stream._input_ch = SimpleNamespace(start_time_offset=0.0)
    return stream


def _committed_transcript(
    text: str,
    *,
    with_timestamps: bool = False,
    language_code: str | None = None,
) -> dict:
    message: dict = {
        "message_type": "committed_transcript_with_timestamps"
        if with_timestamps
        else "committed_transcript",
        "text": text,
        "words": [
            {"text": text, "start": 0.1, "end": 0.4},
        ]
        if text
        else [],
    }
    # the server only carries the detected language on the delayed copy, and only when
    # language detection is enabled
    if language_code is not None:
        message["language_code"] = language_code
    return message


def _partial_transcript(text: str) -> dict:
    return {"message_type": "partial_transcript", "text": text, "words": []}


@pytest.mark.parametrize("server_vad", [None, {"vad_silence_threshold_secs": 0.5}])
@pytest.mark.parametrize("end_time", [0.4, 0.0])
def test_elevenlabs_endpoint_preserves_timing_across_empty_commit(server_vad, end_time):
    stream = _new_stream(server_vad=server_vad, include_timestamps=True)
    stream.start_time_offset = 7.0
    final = _committed_transcript("hello", with_timestamps=True)
    final["words"][0]["end"] = end_time
    stream._process_stream_event(final)
    if server_vad is None:
        stream._process_stream_event(_committed_transcript("", with_timestamps=True))

    endpoint = stream._event_ch.events[-1]
    assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert endpoint.alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert endpoint.alternatives == []

    stream._process_stream_event(_partial_transcript("next turn"))
    stream._process_stream_event(_committed_transcript("", with_timestamps=True))
    assert stream._event_ch.events[-1].alternatives == []


def _interim_texts(stream: elevenlabs_stt.SpeechStream) -> list[str]:
    return [
        event.alternatives[0].text
        for event in stream._event_ch.events
        if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    ]


def test_untimed_final_invalidates_previous_commit_timing() -> None:
    stream = _new_stream(server_vad=None, include_timestamps=True)
    stream._process_stream_event(_committed_transcript("hello", with_timestamps=True))
    final = _committed_transcript("again", with_timestamps=True)
    final["words"] = []
    stream._process_stream_event(final)
    stream._process_stream_event(_committed_transcript("", with_timestamps=True))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._event_ch.events[-1].alternatives == []


def test_advancing_partial_transcripts_are_forwarded() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_partial_transcript("yeah"))
    stream._process_stream_event(_partial_transcript("yeah please"))

    assert _interim_texts(stream) == ["yeah", "yeah please"]


def test_re_sent_partial_transcript_is_dropped() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    for _ in range(5):
        stream._process_stream_event(_partial_transcript("yeah please"))

    assert _interim_texts(stream) == ["yeah please"]
    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
    ]


def test_the_same_words_after_a_commit_are_forwarded_again() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_partial_transcript("right"))
    stream._process_stream_event(_committed_transcript("right"))
    stream._process_stream_event(_partial_transcript("right"))

    assert _interim_texts(stream) == ["right", "right"]


def test_the_same_words_after_an_empty_commit_are_forwarded_again() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_partial_transcript("right"))
    stream._process_stream_event(_committed_transcript(""))
    stream._process_stream_event(_partial_transcript("right"))

    assert _interim_texts(stream) == ["right", "right"]


def test_server_vad_commit_emits_end_of_speech() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[1].alternatives[0].text == "hello"
    assert stream._speaking is False


def test_manual_commit_still_waits_for_empty_commit() -> None:
    stream = _new_stream(server_vad=None)

    stream._process_stream_event(_committed_transcript("hello"))

    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert stream._speaking is True

    stream._process_stream_event(_committed_transcript(""))

    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert stream._speaking is False


def _stt(**kwargs: object) -> elevenlabs_stt.STT:
    return elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime", **kwargs)


def test_no_verbatim_defaults_to_false() -> None:
    assert _stt()._opts.no_verbatim is False


def test_no_verbatim_can_be_enabled() -> None:
    assert _stt(no_verbatim=True)._opts.no_verbatim is True


def test_update_options_sets_no_verbatim() -> None:
    instance = _stt()
    assert instance._opts.no_verbatim is False
    instance.update_options(no_verbatim=True)
    assert instance._opts.no_verbatim is True


def test_update_options_leaves_no_verbatim_untouched_when_not_given() -> None:
    instance = _stt(no_verbatim=True)
    instance.update_options(tag_audio_events=False)
    assert instance._opts.no_verbatim is True


def test_update_options_forwards_no_verbatim_to_active_streams() -> None:
    # no_verbatim is a WebSocket query param applied at connect time, so a live
    # realtime stream must be told to reconnect. Verify STT.update_options
    # forwards it to active streams (which trigger a reconnect).
    instance = _stt()
    captured: dict[str, object] = {}

    class _FakeStream:
        def update_options(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake = _FakeStream()
    instance._streams.add(fake)
    instance.update_options(no_verbatim=True)
    assert captured.get("no_verbatim") is True


def test_enable_logging_defaults_to_true() -> None:
    assert _stt()._opts.enable_logging is True


def test_enable_logging_can_be_disabled() -> None:
    assert _stt(enable_logging=False)._opts.enable_logging is False


def test_previous_text_is_kept_for_realtime_model() -> None:
    assert _stt(previous_text="prior context")._opts.previous_text == "prior context"


def test_previous_text_is_ignored_for_non_realtime_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key",
            model="scribe_v2",
            previous_text="prior context",
        )

    assert instance._opts.previous_text is None
    assert any("previous_text" in record.message for record in caplog.records)


@pytest.mark.parametrize(("enable_logging", "expected"), [(True, "true"), (False, "false")])
async def test_connect_ws_includes_enable_logging(enable_logging: bool, expected: str) -> None:
    # enable_logging is a WebSocket query param. Verify it is forwarded to the
    # realtime connect URL with the expected lowercase boolean string.
    stream = _new_stream()
    stream._opts.enable_logging = enable_logging

    class _ConnOptions:
        timeout = 5.0

    stream._conn_options = _ConnOptions()

    captured: dict[str, object] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: object) -> object:
            captured["url"] = url
            return object()

    stream._session = _FakeSession()

    await stream._connect_ws()

    assert f"enable_logging={expected}" in captured["url"]


async def _connect_ws_url(stream: elevenlabs_stt.SpeechStream) -> str:
    """Run _connect_ws against a fake session and return the realtime connect URL."""

    class _ConnOptions:
        timeout = 5.0

    stream._conn_options = _ConnOptions()

    captured: dict[str, str] = {}

    class _FakeSession:
        async def ws_connect(self, url: str, **kwargs: object) -> object:
            captured["url"] = url
            return object()

    stream._session = _FakeSession()

    await stream._connect_ws()
    return captured["url"]


@pytest.mark.parametrize(
    ("commit_strategy", "server_vad", "expected"),
    [
        (NOT_GIVEN, NOT_GIVEN, "manual"),
        (NOT_GIVEN, None, "manual"),
        (NOT_GIVEN, {}, "vad"),
        ("manual", {}, "manual"),
        ("vad", None, "vad"),
        ("turn_prediction", NOT_GIVEN, "turn_prediction"),
        ("turn_prediction", None, "turn_prediction"),
        ("turn_prediction", {}, "turn_prediction"),
    ],
)
async def test_commit_strategy_controls_connection_and_manual_flush(
    commit_strategy, server_vad, expected
) -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key",
        model="scribe_v3_live",
        commit_strategy=commit_strategy,
        server_vad=server_vad,
    )
    stream = _new_stream()
    stream._opts = instance._opts
    query = URL(await _connect_ws_url(stream)).query
    assert query["commit_strategy"] == expected
    assert instance.capabilities.manual_flush is (expected == "manual")


async def test_turn_prediction_passes_vad_fallback_options() -> None:
    options = {
        "vad_silence_threshold_secs": 2.0,
        "vad_threshold": 0.3,
        "min_speech_duration_ms": 100,
        "min_silence_duration_ms": 200,
    }
    stream = _new_stream(commit_strategy="turn_prediction", server_vad=options)
    stream._opts.model_id = "scribe_v3_live"
    query = URL(await _connect_ws_url(stream)).query
    assert query["commit_strategy"] == "turn_prediction"
    for name, value in options.items():
        assert query[name] == str(value)


@pytest.mark.parametrize("model", ["scribe_v1", "scribe_v2", "scribe_v2_realtime", "scribe_v3"])
def test_turn_prediction_rejects_unsupported_models(model: str) -> None:
    with pytest.raises(ValueError, match="turn_prediction is only supported for scribe_v3_live"):
        elevenlabs_stt.STT(api_key="test-key", model=model, commit_strategy="turn_prediction")

    instance = elevenlabs_stt.STT(api_key="test-key", model=model)
    with pytest.raises(ValueError, match="turn_prediction is only supported for scribe_v3_live"):
        instance.update_options(commit_strategy="turn_prediction", server_vad={})
    assert instance._opts.commit_strategy is NOT_GIVEN
    assert instance._opts.server_vad is NOT_GIVEN


@pytest.mark.parametrize("include_timestamps", [False, True])
@pytest.mark.parametrize("include_language_detection", [False, True])
def test_turn_prediction_waits_for_server_commit_and_emits_one_end_of_speech(
    include_timestamps: bool, include_language_detection: bool
) -> None:
    stream = _new_stream(
        commit_strategy="turn_prediction",
        include_timestamps=include_timestamps,
        include_language_detection=include_language_detection,
    )
    stream._opts.model_id = "scribe_v3_live"
    for probability in (0.08, 0.95):
        stream._process_stream_event(
            {**_partial_transcript("hello"), "turn_probability": probability}
        )
    assert [event.type for event in stream._event_ch.events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
    ]
    for with_timestamps in (False, True):
        stream._process_stream_event(
            _committed_transcript("hello", with_timestamps=with_timestamps)
        )
    assert [event.type for event in stream._event_ch.events[2:]] == [
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert stream._event_ch.events[-1].alternatives[0].text == "hello"
    assert stream._speaking is False


async def test_update_commit_strategy_reconnects_active_stream_and_updates_capabilities() -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3_live")
    stream = _new_stream()
    stream._opts = instance._opts
    stream._reconnect_event = asyncio.Event()
    instance._streams.add(stream)

    for strategy in ("turn_prediction", "manual", "vad"):
        stream._reconnect_event.clear()
        instance.update_options(commit_strategy=strategy)
        assert stream._reconnect_event.is_set()
        assert instance.capabilities.manual_flush is (strategy == "manual")
        assert URL(await _connect_ws_url(stream)).query["commit_strategy"] == strategy

    instance.update_options(server_vad=None)
    assert instance.capabilities.manual_flush is False
    assert URL(await _connect_ws_url(stream)).query["commit_strategy"] == "vad"


def test_stream_strategy_update_keeps_sibling_connections_and_capabilities_in_sync() -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3_live")
    streams = [_new_stream(), _new_stream()]
    for stream in streams:
        stream._stt = instance
        stream._opts = instance._opts
        stream._reconnect_event = asyncio.Event()
        instance._streams.add(stream)

    streams[0].update_options(commit_strategy="turn_prediction")

    assert not instance.capabilities.manual_flush
    assert all(stream._reconnect_event.is_set() for stream in streams)
    assert all(stream._opts.resolved_commit_strategy == "turn_prediction" for stream in streams)


def test_stream_commit_strategy_update_validates_before_changing_options() -> None:
    stream = _new_stream()
    stream._reconnect_event = asyncio.Event()
    with pytest.raises(ValueError, match="turn_prediction is only supported for scribe_v3_live"):
        stream.update_options(commit_strategy="turn_prediction", server_vad={})
    assert not stream._reconnect_event.is_set()
    assert stream._opts.server_vad is NOT_GIVEN

    stream._opts.model_id = "scribe_v3_live"
    stream.update_options(commit_strategy="turn_prediction")
    assert stream._reconnect_event.is_set()
    assert stream._opts.resolved_commit_strategy == "turn_prediction"


async def test_connect_ws_includes_keyterms() -> None:
    # keyterms bias the realtime model and are sent as repeated query params on
    # the connect URL.
    stream = _new_stream()
    stream._opts.keyterms = ["nginx", "Grafana Loki", "Ærø"]

    url = await _connect_ws_url(stream)

    assert "keyterms=nginx" in url
    assert "keyterms=Grafana%20Loki" in url
    assert "keyterms=%C3%86r%C3%B8" in url


async def test_connect_ws_escapes_query_delimiters_in_keyterms() -> None:
    # Keyterms are free-form text. Unescaped, "&" would split the term and inject
    # a bogus query param and "#" would truncate it, so both must be encoded and
    # must survive a round trip through the URL parser aiohttp uses.
    stream = _new_stream()
    stream._opts.keyterms = ["Smith & Sons", "C#"]

    url = await _connect_ws_url(stream)

    assert "keyterms=Smith%20%26%20Sons" in url
    assert "keyterms=C%23" in url
    assert URL(url).query.getall("keyterms") == ["Smith & Sons", "C#"]


async def test_connect_ws_omits_keyterms_when_not_given() -> None:
    url = await _connect_ws_url(_new_stream())

    assert "keyterms=" not in url


def test_update_options_forwards_keyterms_to_active_streams() -> None:
    # keyterms are a WebSocket query param applied at connect time, so a live
    # realtime stream must be told to reconnect. Verify STT.update_options
    # forwards them to active streams (which trigger a reconnect).
    instance = _stt()
    captured: dict[str, object] = {}

    class _FakeStream:
        def update_options(self, **kwargs: object) -> None:
            captured.update(kwargs)

    # _streams is a WeakSet: keep a strong reference so the fake survives.
    fake = _FakeStream()
    instance._streams.add(fake)
    instance.update_options(keyterms=["nginx"])
    assert captured.get("keyterms") == ["nginx"]


def test_stream_update_options_sets_keyterms_and_requests_reconnect() -> None:
    stream = _new_stream()
    stream._reconnect_event = asyncio.Event()

    stream.update_options(keyterms=["nginx"])

    assert stream._opts.keyterms == ["nginx"]
    assert stream._reconnect_event.is_set()


async def test_connect_ws_normalizes_the_primary_language() -> None:
    # LanguageCode keeps the region ("en-US") but the realtime API rejects it, so the primary
    # language goes on the wire through the same normalization the secondary ones get
    stream = _new_stream(language="en_US", secondary_languages=["ru-RU"])

    url = await _connect_ws_url(stream)

    assert URL(url).query.getall("language_code") == ["en"]
    assert URL(url).query.getall("secondary_languages") == ["ru"]


async def test_connect_ws_includes_secondary_languages() -> None:
    # secondary languages ride along with the pinned primary one and are sent as
    # repeated query params, the only serialization the realtime API accepts.
    stream = _new_stream(language="en", secondary_languages=["ru", "es"])

    url = await _connect_ws_url(stream)

    assert "language_code=en" in url
    assert URL(url).query.getall("secondary_languages") == ["ru", "es"]


async def test_connect_ws_normalizes_secondary_languages() -> None:
    # the realtime API takes ISO-639-1/3 and rejects anything else, including the region-tagged
    # tags LanguageCode produces, so names and regions are mapped before the connect URL
    stream = _new_stream(language="en", secondary_languages=["ru_RU", "french", "spa"])

    url = await _connect_ws_url(stream)

    assert URL(url).query.getall("secondary_languages") == ["ru", "fr", "es"]


async def test_connect_ws_omits_secondary_languages_when_not_given() -> None:
    url = await _connect_ws_url(_new_stream(language="en"))

    assert "secondary_languages=" not in url


def test_secondary_languages_are_normalized_on_the_options() -> None:
    instance = elevenlabs_stt.STT(
        api_key="test-key",
        model="scribe_v2_realtime",
        language_code="en_US",
        secondary_languages=["ru_RU", "french", "spa"],
    )

    assert instance._opts.language_code == LanguageCode("en-US")
    assert instance._opts.secondary_languages == ["ru-RU", "fr", "es"]
    assert all(
        isinstance(code, LanguageCode) for code in cast(list, instance._opts.secondary_languages)
    )


def test_secondary_languages_ignored_for_batch_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        instance = elevenlabs_stt.STT(
            api_key="test-key", model="scribe_v2", secondary_languages=["ru"]
        )

    assert instance._opts.secondary_languages is NOT_GIVEN
    assert "only supported for realtime models" in caplog.text


async def test_connect_ws_requests_language_detection_when_no_language_is_pinned() -> None:
    url = await _connect_ws_url(_new_stream(language=None))

    assert "include_language_detection=true" in url


async def test_connect_ws_omits_language_detection_when_language_is_pinned() -> None:
    url = await _connect_ws_url(_new_stream(language="en"))

    assert "include_language_detection" not in url


async def test_connect_ws_requests_language_detection_when_explicitly_enabled() -> None:
    stream = _new_stream(language="en", include_language_detection=True)

    url = await _connect_ws_url(stream)

    assert "include_language_detection=true" in url


async def test_connect_ws_omits_language_detection_when_explicitly_disabled() -> None:
    url = await _connect_ws_url(_new_stream(language=None, include_language_detection=False))

    assert "include_language_detection" not in url


def test_final_transcript_reports_the_detected_language() -> None:
    # with detection on, the detected language only reaches the delayed copy of the
    # commit, so that copy has to be the final one or every transcript is labelled
    # with the pinned language (or "en" when nothing is pinned).
    stream = _new_stream(
        server_vad={"vad_silence_threshold_secs": 0.5},
        language="en",
        include_language_detection=True,
        secondary_languages=["ru"],
    )

    stream._process_stream_event(_committed_transcript("привет"))
    stream._process_stream_event(
        _committed_transcript("привет", with_timestamps=True, language_code="ru")
    )

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "ru"


def test_autodetected_language_reaches_the_final_transcript() -> None:
    # without a pinned language the plugin asks the server to detect one; that language
    # only rides on the delayed copy, so dropping it labelled every transcript "en"
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5}, language=None)

    stream._process_stream_event(
        _committed_transcript("привет", with_timestamps=True, language_code="ru")
    )
    stream._process_stream_event(_committed_transcript("привет"))

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "ru"
    # reading the timestamped copy must not start handing out word timings the caller
    # never asked for
    assert finals[0].alternatives[0].words is None


def test_final_transcript_keeps_the_plain_copy_without_detection() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5}, language="es")

    stream._process_stream_event(_committed_transcript("hola"))
    stream._process_stream_event(_committed_transcript("hola", with_timestamps=True))

    finals = [
        event
        for event in stream._event_ch.events
        if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert len(finals) == 1
    assert finals[0].alternatives[0].language == "es"


@pytest.mark.parametrize(
    "message_type",
    ["partial_transcript", "committed_transcript", "committed_transcript_with_timestamps"],
)
@pytest.mark.parametrize(
    ("language", "language_data", "expected_language"),
    [
        ("es", {}, "es"),
        ("es", {"language_code": None}, "es"),
        ("es", {"language_code": ""}, "es"),
        ("es", {"language_code": "fra"}, "fr"),
        (None, {}, "en"),
        (None, {"language_code": None}, "en"),
        (None, {"language_code": ""}, "en"),
        (None, {"language_code": "fra"}, "fr"),
    ],
)
def test_transcript_language_falls_back_to_configured_language(
    message_type: str,
    language: str | None,
    language_data: dict[str, str | None],
    expected_language: str,
) -> None:
    stream = _new_stream(
        language=language,
        include_timestamps=message_type == "committed_transcript_with_timestamps",
        include_language_detection=False,
    )

    stream._process_stream_event({"message_type": message_type, "text": "hola", **language_data})

    transcript = stream._event_ch.events[-1]
    assert transcript.type == (
        stt.SpeechEventType.INTERIM_TRANSCRIPT
        if message_type == "partial_transcript"
        else stt.SpeechEventType.FINAL_TRANSCRIPT
    )
    assert transcript.alternatives[0].language == expected_language


class _FakeWS:
    """Records outgoing messages. receive() parks so recv_task stays alive."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._closed = asyncio.Event()
        self.received: asyncio.Queue[dict] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> Any:
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT, data=json.dumps(await self.received.get())
        )

    async def close(self) -> None:
        self._closed.set()


def _live_stream(
    ws: _FakeWS, *, model: str = "scribe_v2_realtime", **kwargs: Any
) -> elevenlabs_stt.SpeechStream:
    """A real SpeechStream running its real _run loop against a fake socket."""
    instance = elevenlabs_stt.STT(api_key="test-key", model=model, **kwargs)
    opts = dataclasses.replace(instance._opts)
    stream = elevenlabs_stt.SpeechStream(
        stt=instance,
        opts=opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        language=None,
        http_session=cast(Any, SimpleNamespace(closed=False)),
    )

    async def _fake_connect() -> Any:
        return ws

    # patched before the _run task gets its first tick, so no real socket is opened
    stream._connect_ws = _fake_connect  # type: ignore[method-assign]
    return stream


# Scribe rejects a commit with less audio than this since the previous one
_MIN_COMMIT_MS = 300


def _frame(ms: int, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream to send"
        await asyncio.sleep(0.01)


async def test_flush_commits_the_turn_when_no_audio_is_left_to_send() -> None:
    # 300ms is a whole number of repack chunks, so AudioByteStream.flush() returns no frames.
    # The commit must still go out: Scribe v2 does not finalize a turn without one, and
    # gating it on the per-frame loop dropped it here (roughly 1 flush in 5).
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(_MIN_COMMIT_MS))
        await _wait_until(lambda: len(ws.sent) == 6)

        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 7)

        assert [msg["commit"] for msg in ws.sent] == [False] * 6 + [True]
        assert ws.sent[-1]["audio_base_64"] == ""
    finally:
        await stream.aclose()


async def test_flush_commits_after_the_buffered_audio() -> None:
    # a partial chunk is still pending: it has to reach the server before the commit,
    # otherwise the tail of the turn is transcribed against the next one
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(_MIN_COMMIT_MS + 30))
        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 8)

        assert [msg["commit"] for msg in ws.sent] == [False] * 7 + [True]
        assert ws.sent[-2]["audio_base_64"] != ""
    finally:
        await stream.aclose()


def _audio_ms_before_each_commit(sent: list[dict], sample_rate: int = 16000) -> list[float]:
    audio_ms: list[float] = []
    samples = 0
    for msg in sent:
        if msg["commit"]:
            audio_ms.append(samples * 1000 / sample_rate)
            samples = 0
        else:
            samples += len(base64.b64decode(msg["audio_base_64"])) // 2
    return audio_ms


async def test_flush_pads_a_short_turn_and_skips_an_empty_commit() -> None:
    # Scribe answers a commit with less than 300ms of audio since the previous one with
    # commit_throttled and closes the session, dropping the uncommitted audio. The commit still
    # has to go out on the flush: the caller waits on its final transcript, and no more audio
    # may follow (push-to-talk, session close).
    ws = _FakeWS()
    stream = _live_stream(ws)
    speech = bytes(range(1, 201)) * 16  # 100ms without a zero sample
    try:
        # the first commit on a connection counts the audio since the connect
        stream.push_frame(
            rtc.AudioFrame(
                data=speech, sample_rate=16000, num_channels=1, samples_per_channel=len(speech) // 2
            )
        )
        stream.flush()
        await _wait_until(lambda: sum(msg["commit"] for msg in ws.sent) == 1)

        stream.flush()
        await asyncio.sleep(0.01)

        assert _audio_ms_before_each_commit(ws.sent) == [300]
        silence_ms = _MIN_COMMIT_MS - 100
        audio = b"".join(base64.b64decode(msg["audio_base_64"]) for msg in ws.sent)
        assert audio == speech + bytes(16000 * 2 * silence_ms // 1000)
    finally:
        await stream.aclose()


def test_trace_id_from_headers() -> None:
    # header lookup is case-insensitive, and an absent header returns None
    assert trace_id_from_headers(CIMultiDict({"X-Trace-Id": "trace-1"})) == "trace-1"
    assert trace_id_from_headers(CIMultiDict()) is None
    assert trace_id_from_headers(None) is None


def test_speech_confidence_from_word_logprobs() -> None:
    # confident transcription: word logprobs near 0 -> confidence near 1 (spacing tokens ignored)
    words = [
        {"type": "word", "logprob": -0.01},
        {"type": "spacing", "logprob": -2.0},
        {"type": "word", "logprob": -0.05},
    ]
    assert 0.9 < elevenlabs_stt._speech_confidence(words) <= 1.0


def test_speech_confidence_flags_low_quality_transcription() -> None:
    # uncertain transcription (very negative logprobs) -> low confidence
    words = [{"type": "word", "logprob": -2.5}, {"type": "word", "logprob": -3.0}]
    assert elevenlabs_stt._speech_confidence(words) < 0.2


def test_speech_confidence_defaults_to_zero_without_logprobs() -> None:
    # no words, or words without logprobs (e.g. non-timestamped commit) -> default 0.0
    assert elevenlabs_stt._speech_confidence(None) == 0.0
    assert elevenlabs_stt._speech_confidence([{"text": "hi", "start": 0.1, "end": 0.4}]) == 0.0


def test_committed_transcript_sets_confidence() -> None:
    stream = _new_stream(server_vad={"vad_silence_threshold_secs": 0.5})

    stream._process_stream_event(
        {
            "message_type": "committed_transcript",
            "text": "hello",
            "words": [
                {"text": "hello", "start": 0.1, "end": 0.4, "type": "word", "logprob": -0.02}
            ],
        }
    )

    final = stream._event_ch.events[1]
    assert final.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].confidence > 0.9


@pytest.mark.parametrize(
    ("model", "server_vad", "manual_flush"),
    [
        ("scribe_v2_realtime", NOT_GIVEN, True),
        ("scribe_v2_realtime", None, True),
        ("scribe_v2_realtime", {}, False),
        ("scribe_v2", NOT_GIVEN, False),
    ],
)
def test_manual_flush_capability(model, server_vad, manual_flush) -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model=model, server_vad=server_vad)
    assert instance.capabilities.manual_flush is manual_flush


def test_manual_flush_capability_tracks_server_vad_updates() -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime")
    instance.update_options(server_vad={})
    assert instance.capabilities.manual_flush is False
    instance.update_options(server_vad=None)
    assert instance.capabilities.manual_flush is True


@pytest.mark.parametrize(
    ("model", "strategy", "local_vad", "server_vad", "warns"),
    [
        ("scribe_v3_live", "turn_prediction", False, NOT_GIVEN, True),
        ("scribe_v3_live", "turn_prediction", False, {}, True),
        ("scribe_v3_live", "turn_prediction", True, NOT_GIVEN, False),
        ("scribe_v3_live", "manual", False, NOT_GIVEN, False),
        ("scribe_v2_realtime", "vad", False, NOT_GIVEN, False),
    ],
)
def test_turn_prediction_warns_without_local_vad(
    model, strategy, local_vad, server_vad, warns, caplog: pytest.LogCaptureFixture
) -> None:
    from .fake_vad import FakeVAD

    elevenlabs_stt.STT(
        api_key="test-key",
        model=model,
        commit_strategy=strategy,
        vad=FakeVAD() if local_vad else None,
        server_vad=server_vad,
    )
    assert ("approximately 36 seconds" in caplog.text) is warns
    if warns:
        assert "elevenlabs.STT(vad=...)" in caplog.text


def test_enabling_turn_prediction_warns_once_with_an_active_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v3_live")
    stream = _new_stream()
    stream._opts = instance._opts
    stream._reconnect_event = asyncio.Event()
    instance._streams.add(stream)
    instance.update_options(commit_strategy="turn_prediction")
    assert len(caplog.records) == 1
    assert "approximately 36 seconds" in caplog.text
    caplog.clear()
    instance.update_options(keyterms=["LiveKit"])
    assert not caplog.records


def test_stream_warns_when_enabling_turn_prediction_without_vad(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _new_stream()
    stream._opts.model_id = "scribe_v3_live"
    stream._reconnect_event = asyncio.Event()
    stream.update_options(commit_strategy="turn_prediction")
    assert "approximately 36 seconds" in caplog.text


@pytest.mark.parametrize("options", [{}, {"tag_audio_events": False}, {"keyterms": ["LiveKit"]}])
def test_unrelated_updates_preserve_manual_flush_override(options) -> None:
    instance = _stt()
    instance.capabilities.manual_flush = False
    instance.update_options(**options)
    assert instance.capabilities.manual_flush is False


class _ControlledVADStream(vad_module.VADStream):
    def __init__(self, vad: vad_module.VAD) -> None:
        super().__init__(vad)
        self.fail_push = False

    async def _main_task(self) -> None:
        async for _ in self._input_ch:
            pass

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self.fail_push:
            raise RuntimeError("VAD failed")
        super().push_frame(frame)

    def emit(self, event_type: vad_module.VADEventType) -> None:
        self._event_ch.send_nowait(
            vad_module.VADEvent(
                type=event_type,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.0,
                silence_duration=0.0,
                raw_accumulated_speech=0.0,
                raw_accumulated_silence=0.0,
            )
        )


class _ControlledVAD(vad_module.VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=vad_module.VADCapabilities(update_interval=0.1))
        self.streams: list[_ControlledVADStream] = []

    def stream(self) -> _ControlledVADStream:
        stream = _ControlledVADStream(self)
        self.streams.append(stream)
        return stream


async def _partial_detector(
    *, model: str = "scribe_v3_live", strategy: str = "turn_prediction", local_vad: bool = True
) -> tuple[TurnDetection, _ControlledVAD, list[None], asyncio.Task[None]]:
    vad = _ControlledVAD()
    ended: list[None] = []
    opts = dataclasses.replace(
        _new_stream(server_vad={"vad_silence_threshold_secs": 0.5}, commit_strategy=strategy)._opts,
        model_id=model,
    )
    detector = TurnDetection(
        opts=opts, vad=vad if local_vad else None, on_end_of_speech=lambda: ended.append(None)
    )
    await detector.start()
    task = asyncio.create_task(detector.run_vad())
    await asyncio.sleep(0)
    return detector, vad, ended, task


@pytest.mark.virtual_time
@pytest.mark.parametrize("speaking", [False, True])
async def test_partial_timeout_ignores_existing_turn_partial(speaking: bool) -> None:
    detector, vad, ended, task = await _partial_detector()
    try:
        if speaking:
            vad.streams[0].emit(vad_module.VADEventType.START_OF_SPEECH)
            await asyncio.sleep(0)
        detector.on_partial_transcript(0.5, text="continuation", starts_turn=False)
        if speaking:
            vad.streams[0].emit(vad_module.VADEventType.END_OF_SPEECH)
        await asyncio.sleep(3)
        assert not ended
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_uncommitted_partial_times_out_once_without_extending_for_updates() -> None:
    detector, _, ended, task = await _partial_detector()
    try:
        detector.on_partial_transcript(0.76, text="Yeah.", starts_turn=True)
        await asyncio.sleep(0.7)
        assert not detector.on_partial_transcript(0.8, text="Yeah.")
        detector.on_partial_transcript(0.8, text="Yeah please.")
        await asyncio.sleep(0.3)
        assert len(ended) == 1
        detector.on_partial_transcript(0.8, text="Yeah please.")
        await asyncio.sleep(2)
        assert len(ended) == 1
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_partial_timeout_restarts_after_local_vad_speech() -> None:
    detector, vad, ended, task = await _partial_detector()
    try:
        detector.on_partial_transcript(0.5, text="Hello", starts_turn=True)
        await asyncio.sleep(0.7)
        vad.streams[0].emit(vad_module.VADEventType.START_OF_SPEECH)
        await asyncio.sleep(2)
        detector.on_partial_transcript(0.5, text="Hello there")
        await asyncio.sleep(2)
        assert not ended
        vad.streams[0].emit(vad_module.VADEventType.END_OF_SPEECH)
        await asyncio.sleep(0)
        await asyncio.sleep(1.1)
        assert len(ended) == 1
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
@pytest.mark.parametrize("text", ["Hello", ""])
async def test_provider_commit_cancels_partial_timeout(text: str) -> None:
    detector, _, ended, task = await _partial_detector()
    try:
        detector.on_partial_transcript(0.5, text="Hello", starts_turn=True)
        await asyncio.sleep(0.7)
        detector.on_committed_transcript(
            {"message_type": "committed_transcript", "text": text},
            final_message_type="committed_transcript_with_timestamps",
        )
        await asyncio.sleep(2)
        assert not ended
        detector.on_committed_transcript(
            {"message_type": "committed_transcript_with_timestamps", "text": text},
            final_message_type="committed_transcript_with_timestamps",
        )
        detector.end_speech()
        await asyncio.sleep(2)
        assert len(ended) == 1
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_delayed_forced_timestamp_copy_does_not_guard_continuation() -> None:
    detector, vad, ended, task = await _partial_detector()
    try:
        detector._vad_speaking = True
        detector._audio_duration_since_commit = 35.3
        detector.on_partial_transcript(0.5, text="Old segment", starts_turn=True)
        detector.on_committed_transcript(
            {"message_type": "committed_transcript", "text": "Old segment"},
            final_message_type="committed_transcript_with_timestamps",
        )
        vad.streams[0].emit(vad_module.VADEventType.END_OF_SPEECH)
        await asyncio.sleep(0)
        detector.on_partial_transcript(0.5, text="New segment")
        await asyncio.sleep(0.7)
        assert detector.on_committed_transcript(
            {"message_type": "committed_transcript_with_timestamps", "text": "Old segment"},
            final_message_type="committed_transcript_with_timestamps",
        )
        await asyncio.sleep(3)
        assert not ended
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_unselected_timestamp_copy_does_not_cancel_new_partial_timeout() -> None:
    detector, _, ended, task = await _partial_detector()
    try:
        detector.on_partial_transcript(0.5, text="Old segment", starts_turn=True)
        detector.on_committed_transcript(
            {"message_type": "committed_transcript", "text": "Old segment"},
            final_message_type="committed_transcript",
        )
        detector.end_speech()
        detector.on_partial_transcript(0.5, text="New segment", starts_turn=True)
        detector.on_committed_transcript(
            {"message_type": "committed_transcript_with_timestamps", "text": "Old segment"},
            final_message_type="committed_transcript",
        )
        await asyncio.sleep(1)
        assert len(ended) == 2
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_fresh_partial_is_guarded_after_forced_commit_fallback() -> None:
    detector, vad, ended, task = await _partial_detector()
    try:
        detector._vad_speaking = True
        detector._audio_duration_since_commit = 35.3
        detector.on_partial_transcript(0.5, text="Old segment", starts_turn=True)
        detector.on_committed_transcript(
            {"message_type": "committed_transcript", "text": "Old segment"},
            final_message_type="committed_transcript_with_timestamps",
        )
        assert detector.on_committed_transcript(
            {"message_type": "committed_transcript_with_timestamps", "text": "Old segment"},
            final_message_type="committed_transcript_with_timestamps",
        )
        await asyncio.sleep(1)
        assert len(ended) == 1
        detector.on_partial_transcript(0.5, text="Fresh turn", starts_turn=True)
        vad.streams[0].emit(vad_module.VADEventType.END_OF_SPEECH)
        await asyncio.sleep(0)
        await asyncio.sleep(1.1)
        assert len(ended) == 2
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
@pytest.mark.parametrize("restart", [False, True])
async def test_partial_timeout_is_cancelled_on_close_or_restart(restart: bool) -> None:
    detector, _, ended, task = await _partial_detector()
    detector.on_partial_transcript(0.5, text="Old connection", starts_turn=True)
    await asyncio.sleep(0.7)
    if restart:
        await detector.start()
    else:
        await detector.aclose()
    assert len(ended) == 1
    await asyncio.sleep(2)
    assert len(ended) == 1
    if restart:
        detector.on_partial_transcript(0.5, text="New connection", starts_turn=True)
        await asyncio.sleep(1)
        assert len(ended) == 2
        await detector.aclose()
    await task


@pytest.mark.virtual_time
async def test_local_vad_failure_cancels_partial_timeout() -> None:
    detector, vad, ended, task = await _partial_detector()
    try:
        detector.on_partial_transcript(0.5, text="Hello", starts_turn=True)
        vad.streams[0].fail_push = True
        await detector.push_frame(_frame(50))
        await asyncio.sleep(2)
        detector.on_partial_transcript(0.5, text="Hello there")
        await asyncio.sleep(2)
        assert not ended
    finally:
        await detector.aclose()
        await task


@pytest.mark.virtual_time
async def test_clean_local_vad_completion_cancels_partial_timeout() -> None:
    detector, _, ended, task = await _partial_detector()
    detector.on_partial_transcript(0.5, text="Hello", starts_turn=True)
    await detector.end_input()
    await task
    await asyncio.sleep(2)
    assert not ended
    assert detector._vad_stream is None
    await detector.aclose()


@pytest.mark.virtual_time
@pytest.mark.parametrize(
    ("model", "strategy", "local_vad"),
    [
        ("scribe_v3_live", "manual", True),
        ("scribe_v2_realtime", "vad", True),
        ("scribe_v3_live", "turn_prediction", False),
    ],
)
async def test_partial_timeout_stays_disabled_without_eligible_local_vad(
    model: str, strategy: str, local_vad: bool
) -> None:
    detector, _, ended, task = await _partial_detector(
        model=model, strategy=strategy, local_vad=local_vad
    )
    detector.on_partial_transcript(0.5, text="Hello", starts_turn=True)
    await asyncio.sleep(10)
    assert not ended
    await detector.aclose()
    await task


@pytest.mark.virtual_time
@pytest.mark.parametrize("after_commit", [False, True])
async def test_stream_closes_dangling_partial_after_optional_committed_turn(
    after_commit: bool,
) -> None:
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 0.5},
        vad=_ControlledVAD(),
        include_language_detection=False,
    )
    events: list[stt.SpeechEvent] = []

    async def collect_events() -> None:
        async for event in stream:
            events.append(event)

    collector = asyncio.create_task(collect_events())
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: bool(ws.sent))
        if after_commit:
            ws.received.put_nowait(_committed_transcript("Tell me a story."))
            await _wait_until(
                lambda: any(event.type == stt.SpeechEventType.END_OF_SPEECH for event in events)
            )
            events.clear()

        ws.received.put_nowait(_partial_transcript("Yeah."))
        await _wait_until(
            lambda: any(event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for event in events)
        )
        await _wait_until(
            lambda: any(event.type == stt.SpeechEventType.END_OF_SPEECH for event in events)
        )
        assert [event.type for event in events] == [
            stt.SpeechEventType.START_OF_SPEECH,
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            stt.SpeechEventType.END_OF_SPEECH,
        ]
        assert events[-1].alternatives == []

        ws.received.put_nowait(_committed_transcript(""))
        ws.received.put_nowait(_committed_transcript("Next real turn"))
        await _wait_until(
            lambda: any(
                event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                and event.alternatives[0].text == "Next real turn"
                for event in events
            )
        )
        assert [
            event.type for event in events if event.type != stt.SpeechEventType.INTERIM_TRANSCRIPT
        ] == [
            stt.SpeechEventType.START_OF_SPEECH,
            stt.SpeechEventType.END_OF_SPEECH,
            stt.SpeechEventType.START_OF_SPEECH,
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.END_OF_SPEECH,
        ]
    finally:
        await stream.aclose()
        await collector


@pytest.mark.virtual_time
@pytest.mark.parametrize("termination", ["close", "provider_error"])
@pytest.mark.parametrize("local_vad", [True, False])
async def test_stream_finishes_guarded_partial_on_teardown(
    termination: str, local_vad: bool
) -> None:
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        vad=_ControlledVAD() if local_vad else None,
    )
    stream._conn_options = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)
    events: list[stt.SpeechEventType] = []

    async def collect_events() -> None:
        async for event in stream:
            events.append(event.type)

    collector = asyncio.create_task(collect_events())
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: bool(ws.sent))
        ws.received.put_nowait(_partial_transcript("Hello"))
        await _wait_until(lambda: stt.SpeechEventType.INTERIM_TRANSCRIPT in events)

        if termination == "provider_error":
            ws.received.put_nowait({"message_type": "transcriber_error"})
            with pytest.raises(APIConnectionError, match="transcriber_error"):
                await asyncio.wait_for(collector, timeout=1)
        else:
            await stream.aclose()
            await collector
    finally:
        await stream.aclose()
        await asyncio.gather(collector, return_exceptions=True)

    expected = [stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.INTERIM_TRANSCRIPT]
    if local_vad:
        expected.append(stt.SpeechEventType.END_OF_SPEECH)
    assert events == expected


@pytest.mark.virtual_time
@pytest.mark.parametrize("include_timestamps", [False, True])
async def test_stream_does_not_time_out_forced_commit_continuation(
    include_timestamps: bool,
) -> None:
    vad = _ControlledVAD()
    ws = _FakeWS()
    stream = _live_stream(
        ws,
        model="scribe_v3_live",
        commit_strategy="turn_prediction",
        server_vad={"vad_silence_threshold_secs": 0.5},
        vad=vad,
        include_timestamps=include_timestamps,
        include_language_detection=False,
    )
    events: list[stt.SpeechEvent] = []

    async def collect_events() -> None:
        async for event in stream:
            events.append(event)

    collector = asyncio.create_task(collect_events())
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: bool(ws.sent) and bool(vad.streams))
        stream._turn_detection._vad_speaking = True
        stream._turn_detection._audio_duration_since_commit = 35.3
        ws.received.put_nowait({**_partial_transcript("old segment"), "turn_probability": 0.5})
        ws.received.put_nowait(_committed_transcript("old segment"))
        if include_timestamps:
            ws.received.put_nowait(_committed_transcript("old segment", with_timestamps=True))
        await _wait_until(
            lambda: any(event.type == stt.SpeechEventType.FINAL_TRANSCRIPT for event in events)
        )

        ws.received.put_nowait(_partial_transcript("continuation"))
        await _wait_until(
            lambda: any(
                event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
                and event.alternatives[0].text == "continuation"
                for event in events
            )
        )
        vad.streams[0].emit(vad_module.VADEventType.END_OF_SPEECH)
        await asyncio.sleep(0)
        await asyncio.sleep(1.2)
        assert sum(event.type == stt.SpeechEventType.START_OF_SPEECH for event in events) == 1
        assert not any(event.type == stt.SpeechEventType.END_OF_SPEECH for event in events)

        ws.received.put_nowait(_committed_transcript("continuation"))
        if include_timestamps:
            ws.received.put_nowait(_committed_transcript("continuation", with_timestamps=True))
        await _wait_until(
            lambda: any(event.type == stt.SpeechEventType.END_OF_SPEECH for event in events)
        )
        assert [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["old segment", "continuation"]
    finally:
        await stream.aclose()
        await collector


@pytest.mark.virtual_time
@pytest.mark.parametrize("reconnect_cause", ["options", "provider_error"])
@pytest.mark.parametrize("old_final_delivered", [False, True])
@pytest.mark.parametrize("commit_strategy", ["manual", "turn_prediction"])
async def test_reconnect_balances_old_turn_before_new_partial(
    reconnect_cause: str, old_final_delivered: bool, commit_strategy: str
) -> None:
    sockets = [_FakeWS(), _FakeWS()]
    connections: list[_FakeWS] = []
    stream = _live_stream(
        sockets[0],
        model="scribe_v3_live",
        commit_strategy=commit_strategy,
        server_vad={"vad_silence_threshold_secs": 0.5},
        vad=_ControlledVAD(),
        include_language_detection=False,
    )
    stream._conn_options = dataclasses.replace(
        DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0, liveness_timeout=60.0
    )

    async def connect() -> _FakeWS:
        ws = sockets[len(connections)]
        connections.append(ws)
        return ws

    stream._connect_ws = connect
    events: list[stt.SpeechEvent] = []

    async def collect_events() -> None:
        async for event in stream:
            events.append(event)

    collector = asyncio.create_task(collect_events())
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: bool(sockets[0].sent))
        sockets[0].received.put_nowait(_partial_transcript("old turn"))
        await _wait_until(
            lambda: any(event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for event in events)
        )
        if old_final_delivered:
            sockets[0].received.put_nowait(_committed_transcript("old turn"))
            await _wait_until(
                lambda: bool(events) and events[-1].type != stt.SpeechEventType.INTERIM_TRANSCRIPT
            )

        if reconnect_cause == "options":
            stream.update_options(keyterms=["LiveKit"])
        else:
            sockets[0].received.put_nowait({"message_type": "transcriber_error"})
        await _wait_until(lambda: len(connections) == 2)

        sockets[1].received.put_nowait(_partial_transcript("new turn"))
        await _wait_until(
            lambda: sum(event.type == stt.SpeechEventType.START_OF_SPEECH for event in events) == 2
        )
        if commit_strategy == "manual":
            sockets[1].received.put_nowait(_committed_transcript(""))
        await _wait_until(
            lambda: sum(event.type == stt.SpeechEventType.END_OF_SPEECH for event in events) == 2
        )
        turn_types = [
            event.type
            for event in events
            if event.type
            in (
                stt.SpeechEventType.START_OF_SPEECH,
                stt.SpeechEventType.FINAL_TRANSCRIPT,
                stt.SpeechEventType.END_OF_SPEECH,
            )
        ]
        expected = [stt.SpeechEventType.START_OF_SPEECH]
        if old_final_delivered:
            expected.append(stt.SpeechEventType.FINAL_TRANSCRIPT)
        expected.extend(
            [
                stt.SpeechEventType.END_OF_SPEECH,
                stt.SpeechEventType.START_OF_SPEECH,
                stt.SpeechEventType.END_OF_SPEECH,
            ]
        )
        assert turn_types == expected
    finally:
        await stream.aclose()
        await collector


def _forced_commit_stream(**kwargs) -> elevenlabs_stt.SpeechStream:
    stream = _new_stream(commit_strategy="turn_prediction", **kwargs)
    stream._opts.model_id = "scribe_v3_live"
    stream._turn_detection._vad_speaking = True
    stream._turn_detection._audio_duration_since_commit = 35.3
    stream._process_stream_event({**_partial_transcript("still speaking"), "turn_probability": 0.0})
    return stream


@pytest.mark.virtual_time
@pytest.mark.parametrize("silence_threshold", [None, 0.5])
async def test_forced_commit_keeps_turn_open_then_times_out_without_more_text(
    silence_threshold: float | None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.elevenlabs")
    stream = _forced_commit_stream(server_vad={"vad_silence_threshold_secs": silence_threshold})
    stream._process_stream_event(_committed_transcript("still speaking"))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert stream._speaking
    deferred = [r for r in caplog.records if r.message.startswith("deferring end of speech")]
    assert len(deferred) == 1
    assert deferred[0].levelno == logging.DEBUG
    assert deferred[0].audio_duration_since_commit == pytest.approx(35.3)
    assert deferred[0].turn_probability == 0.0
    assert deferred[0].vad_speaking is True
    # VAD can stay positive due to background speech. It must not prevent the fallback.
    timeout = (silence_threshold if silence_threshold is not None else 1.5) + 0.5
    await asyncio.sleep(timeout - 0.1)
    stream._process_stream_event(_partial_transcript(""))
    assert stream._speaking
    await asyncio.sleep(0.2)
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert not stream._speaking
    await asyncio.sleep(2)
    assert sum(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events) == 1
    fallback = [r for r in caplog.records if r.message.startswith("emitting end of speech")]
    assert len(fallback) == 1
    assert fallback[0].levelno == logging.WARNING
    assert fallback[0].eos_timeout == timeout
    assert fallback[0].vad_speaking is True
    for record in deferred + fallback:
        assert "lk.pii.data" not in record.__dict__
        assert "still speaking" not in json.dumps(record.__dict__, default=str)


@pytest.mark.virtual_time
async def test_new_segment_text_cancels_forced_commit_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _forced_commit_stream(server_vad={"vad_silence_threshold_secs": 0.5})
    stream._process_stream_event(_committed_transcript("still speaking"))
    stream._process_stream_event({**_partial_transcript("and continuing"), "turn_probability": 0.1})
    await asyncio.sleep(4)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert stream._speaking
    assert not any(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events)
    # The next segment still needs its final transcript before the turn can close.
    stream._turn_detection._vad_speaking = False
    stream._process_stream_event(_committed_transcript("and continuing"))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH


@pytest.mark.parametrize(
    ("duration", "probability", "speaking", "client_commits"),
    [
        (10, 0.0, True, 0),
        (40, 0.0, True, 0),
        (35.3, 0.9, True, 0),
        (35.3, None, True, 0),
        (35.3, 0.0, False, 0),
        (35.3, 0.0, True, 1),
    ],
)
def test_other_automatic_commits_still_on_end_of_speech(
    duration, probability, speaking, client_commits
) -> None:
    stream = _forced_commit_stream()
    turn_detection = stream._turn_detection
    turn_detection._audio_duration_since_commit = duration
    turn_detection.on_partial_transcript(probability, text="still speaking")
    turn_detection._vad_speaking = speaking
    for _ in range(client_commits):
        turn_detection.on_commit_requested()
    stream._process_stream_event(_committed_transcript("hello"))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH


@pytest.mark.parametrize("probability", [0.9, 1.0])
def test_turn_probability_updates_even_when_partial_text_does_not_change(probability) -> None:
    stream = _forced_commit_stream()
    stream._process_stream_event(
        {**_partial_transcript("still speaking"), "turn_probability": probability}
    )
    stream._process_stream_event(_committed_transcript("still speaking"))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert len(_interim_texts(stream)) == 1


@pytest.mark.virtual_time
async def test_forced_commit_duration_restarts_for_each_segment() -> None:
    stream = _forced_commit_stream()
    try:
        stream._process_stream_event(_committed_transcript("still speaking"))
        stream._process_stream_event(
            {**_partial_transcript("next segment"), "turn_probability": 0.0}
        )
        await stream._turn_detection.push_frame(_frame(35300))
        stream._process_stream_event(_committed_transcript("next segment"))
        assert not any(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events)
        assert stream._turn_detection._eos_timeout_handle is not None
    finally:
        stream._turn_detection.cancel_eos_timeout()


@pytest.mark.virtual_time
async def test_forced_commit_decision_survives_delayed_timestamps_and_new_partials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.elevenlabs")
    stream = _forced_commit_stream(include_timestamps=True)
    stream._process_stream_event(_committed_transcript("still speaking"))
    stream._process_stream_event({**_partial_transcript("next segment"), "turn_probability": 0.95})
    stream._turn_detection._vad_speaking = False
    stream._process_stream_event(_committed_transcript("still speaking", with_timestamps=True))
    await asyncio.sleep(5)
    assert stream._speaking
    assert not any(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events)
    stream._process_stream_event(_committed_transcript("next segment"))
    stream._process_stream_event(_committed_transcript("next segment", with_timestamps=True))
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert sum(r.message.startswith("deferring end of speech") for r in caplog.records) == 1
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.virtual_time
@pytest.mark.parametrize("continuation", ["still speaking", "next segment"])
async def test_commit_resets_partial_deduplication_before_delayed_timestamps(
    continuation: str,
) -> None:
    stream = _forced_commit_stream(include_timestamps=True)
    stream._process_stream_event(_committed_transcript("still speaking"))
    stream._process_stream_event({**_partial_transcript(continuation), "turn_probability": 0.0})
    assert _interim_texts(stream) == ["still speaking", continuation]

    stream._process_stream_event(_committed_transcript("still speaking", with_timestamps=True))
    stream._process_stream_event({**_partial_transcript(continuation), "turn_probability": 0.0})
    assert _interim_texts(stream) == ["still speaking", continuation]
    await asyncio.sleep(3)
    assert stream._speaking
    assert not any(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events)
    await stream._turn_detection.aclose()


@pytest.mark.virtual_time
async def test_server_final_cancels_forced_commit_timer() -> None:
    stream = _forced_commit_stream()
    stream._process_stream_event(_committed_transcript("still speaking"))
    timer = stream._turn_detection._eos_timeout_handle
    assert timer is not None
    stream._process_stream_event(_committed_transcript(""))
    assert timer.cancelled()
    await asyncio.sleep(5)
    assert sum(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events) == 1


@pytest.mark.virtual_time
@pytest.mark.parametrize("include_timestamps", [False, True])
async def test_turn_detection_restart_finishes_deferred_turn_and_clears_delayed_commit(
    include_timestamps: bool,
) -> None:
    stream = _forced_commit_stream(include_timestamps=include_timestamps)
    turn_detection = stream._turn_detection
    stream._process_stream_event(_committed_transcript("still speaking"))
    stream._process_stream_event(_committed_transcript("still speaking", with_timestamps=True))
    await turn_detection.start()
    await asyncio.sleep(5)
    assert sum(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events) == 1
    assert not stream._speaking
    stream._process_stream_event({**_partial_transcript("new connection"), "turn_probability": 1.0})
    stream._process_stream_event(
        _committed_transcript("new connection", with_timestamps=include_timestamps)
    )
    assert stream._event_ch.events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    await turn_detection.aclose()


@pytest.mark.virtual_time
@pytest.mark.parametrize("completion", ["server_final", "timeout"])
async def test_completed_deferred_turn_does_not_end_next_partial_on_close(completion: str) -> None:
    stream = _forced_commit_stream()
    stream._process_stream_event(_committed_transcript("still speaking"))
    if completion == "server_final":
        stream._process_stream_event(_committed_transcript("done"))
    else:
        await asyncio.sleep(2.1)
    assert not stream._speaking
    stream._process_stream_event(_partial_transcript("new turn"))

    await stream._turn_detection.aclose()

    assert stream._speaking
    assert sum(e.type == stt.SpeechEventType.END_OF_SPEECH for e in stream._event_ch.events) == 1


@pytest.mark.virtual_time
@pytest.mark.parametrize("reconnect_cause", ["options", "provider_error"])
@pytest.mark.parametrize("continuation", [None, "before_final", "after_final"])
async def test_reconnect_finishes_deferred_turn_without_more_transcripts(
    reconnect_cause: str,
    continuation: str | None,
) -> None:
    sockets = [_FakeWS(), _FakeWS()]
    connections: list[_FakeWS] = []
    stream = _live_stream(sockets[0], model="scribe_v3_live", commit_strategy="turn_prediction")
    stream._conn_options = dataclasses.replace(
        DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0, liveness_timeout=60.0
    )

    async def connect():
        ws = sockets[len(connections)]
        connections.append(ws)
        return ws

    stream._connect_ws = connect
    try:
        stream.push_frame(_frame(35300))
        await _wait_until(lambda: bool(sockets[0].sent))
        stream._turn_detection._vad_speaking = True
        sockets[0].received.put_nowait({**_partial_transcript("hello"), "turn_probability": 0.0})
        sockets[0].received.put_nowait(_committed_transcript("hello"))
        if continuation == "before_final":
            sockets[0].received.put_nowait(_partial_transcript("continuing"))
        sockets[0].received.put_nowait(_committed_transcript("hello", with_timestamps=True))
        events = [
            await stream.__anext__() for _ in range(4 if continuation == "before_final" else 3)
        ]
        assert events[-1].type == stt.SpeechEventType.FINAL_TRANSCRIPT
        if continuation == "after_final":
            sockets[0].received.put_nowait(_partial_transcript("continuing"))
            assert (await stream.__anext__()).type == stt.SpeechEventType.INTERIM_TRANSCRIPT
        assert (stream._turn_detection._eos_timeout_handle is None) is (continuation is not None)

        if reconnect_cause == "options":
            stream.update_options(keyterms=["LiveKit"])
        else:
            sockets[0].received.put_nowait({"message_type": "transcriber_error"})

        endpoint = await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert endpoint.type == stt.SpeechEventType.END_OF_SPEECH
        assert endpoint.alternatives[0].text == "hello"
        await _wait_until(lambda: len(connections) == 2)
        await asyncio.sleep(3)
        assert not stream._speaking
        assert not stream._task.done()
    finally:
        await stream.aclose()


@pytest.mark.virtual_time
@pytest.mark.parametrize("failure_point", ["push_frame", "inference"])
@pytest.mark.parametrize("termination", ["close", "reconnect"])
async def test_teardown_waits_for_vad_cleanup_after_failure(
    failure_point: str,
    termination: str,
) -> None:
    from .fake_vad import FakeVAD, FakeVADStream

    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_done = asyncio.Event()

    class FailingVADStream(FakeVADStream):
        close_calls = 0

        async def _main_task(self):
            async for _ in self._input_ch:
                raise RuntimeError("VAD inference failed")

        def push_frame(self, frame):
            if failure_point == "push_frame":
                raise RuntimeError("VAD push failed")
            super().push_frame(frame)

        async def aclose(self):
            self.close_calls += 1
            cleanup_started.set()
            await allow_cleanup.wait()
            await super().aclose()
            cleanup_done.set()

    class FailingVAD(FakeVAD):
        def stream(self):
            if not hasattr(self, "first_stream"):
                self.first_stream = FailingVADStream(self)
                return self.first_stream
            return FakeVADStream(self)

    detector = FailingVAD()
    sockets = [_FakeWS(), _FakeWS()]
    connections: list[_FakeWS] = []
    stream = _live_stream(
        sockets[0], model="scribe_v3_live", commit_strategy="turn_prediction", vad=detector
    )
    closing = None

    async def connect():
        ws = sockets[len(connections)]
        connections.append(ws)
        return ws

    stream._connect_ws = connect
    try:
        stream.push_frame(_frame(50))
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        if termination == "close":
            closing = asyncio.create_task(stream.aclose())
        else:
            stream.update_options(keyterms=["LiveKit"])
        await asyncio.sleep(0.05)
        if closing is not None:
            assert not closing.done()
        else:
            assert len(connections) == 1

        allow_cleanup.set()
        if closing is not None:
            await asyncio.wait_for(closing, timeout=1)
        else:
            await _wait_until(lambda: len(connections) == 2)
        assert cleanup_done.is_set()
        assert detector.first_stream.close_calls == 1
        assert detector.first_stream._input_ch.closed
        assert detector.first_stream._task.done()
    finally:
        allow_cleanup.set()
        if closing is not None:
            await closing
        await stream.aclose()
        if not cleanup_done.is_set():
            await FakeVADStream.aclose(detector.first_stream)


@pytest.mark.virtual_time
@pytest.mark.parametrize("failure_point", ["start", "push_frame", "inference", "end_input"])
async def test_local_vad_failure_does_not_stop_transcription(
    failure_point: str, caplog: pytest.LogCaptureFixture
) -> None:
    from .fake_vad import FakeVAD, FakeVADStream

    class FailingVADStream(FakeVADStream):
        async def _main_task(self):
            async for _ in self._input_ch:
                if failure_point == "inference":
                    raise RuntimeError("VAD inference failed")

        def push_frame(self, frame):
            if failure_point == "push_frame":
                raise RuntimeError("VAD push failed")
            super().push_frame(frame)

        def end_input(self):
            if failure_point == "end_input":
                raise RuntimeError("VAD end_input failed")
            super().end_input()

    class FailingVAD(FakeVAD):
        def stream(self):
            if failure_point == "start":
                raise RuntimeError("VAD start failed")
            self.last_stream = FailingVADStream(self)
            return self.last_stream

    detector = FailingVAD()
    ws = _FakeWS()
    stream = _live_stream(
        ws, model="scribe_v3_live", commit_strategy="turn_prediction", vad=detector
    )
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: bool(ws.sent))
        if failure_point == "end_input":
            stream.end_input()
        await _wait_until(lambda: "disabling forced-commit guard" in caplog.text)
        assert stream._turn_detection._vad_stream is None
        assert not stream._turn_detection._vad_speaking
        if failure_point != "start":
            assert detector.last_stream._task.done()

        ws.received.put_nowait({**_partial_transcript("hello"), "turn_probability": 0.0})
        ws.received.put_nowait(_committed_transcript("hello"))
        ws.received.put_nowait(_committed_transcript("hello", with_timestamps=True))
        events = []
        while not events or events[-1].type != stt.SpeechEventType.END_OF_SPEECH:
            events.append(await asyncio.wait_for(stream.__anext__(), timeout=1))
        assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
        assert events[-2].alternatives[0].text == "hello"
        assert not stream._task.done()
        assert sum("disabling forced-commit guard" in r.message for r in caplog.records) == 1
    finally:
        await stream.aclose()


@pytest.mark.virtual_time
async def test_vad_drives_turn_detection_and_closes_with_stream() -> None:
    from .fake_stt import FakeUserSpeech
    from .fake_vad import FakeVAD

    class RecordingVAD(FakeVAD):
        def stream(self):
            self.last_stream = super().stream()
            return self.last_stream

    detector = RecordingVAD(
        fake_user_speeches=[FakeUserSpeech(start_time=0, end_time=50, transcript="", stt_delay=0)]
    )
    ws = _FakeWS()
    stream = _live_stream(
        ws, model="scribe_v3_live", commit_strategy="turn_prediction", vad=detector
    )
    try:
        stream.push_frame(_frame(35300))
        await asyncio.sleep(0.1)
        assert stream._turn_detection._vad_speaking
        assert stream._turn_detection._audio_duration_since_commit == pytest.approx(35.3)
        ws.received.put_nowait({**_partial_transcript("hello"), "turn_probability": 0.0})
        ws.received.put_nowait(_committed_transcript("hello"))
        ws.received.put_nowait(_committed_transcript("hello", with_timestamps=True))
        assert (await stream.__anext__()).type == stt.SpeechEventType.START_OF_SPEECH
        assert (await stream.__anext__()).type == stt.SpeechEventType.INTERIM_TRANSCRIPT
        assert (await stream.__anext__()).type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert stream._speaking
        timer = stream._turn_detection._eos_timeout_handle
        assert timer is not None
    finally:
        await stream.aclose()
    assert timer.cancelled()
    assert detector.last_stream._task.done()


@pytest.mark.parametrize(
    "message_type", ["auth_error", "quota_exceeded", "transcriber_error", "input_error", "error"]
)
def test_provider_error_content_stays_in_pii_attributes(
    message_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    payload = {
        "message_type": message_type,
        "message": "customer transcript",
        "details": "provider credential",
    }
    stream = _new_stream()
    with caplog.at_level(logging.ERROR), pytest.raises(APIConnectionError) as exc:
        stream._process_stream_event(payload)

    assert str(exc.value) == f"ElevenLabs STT error [{message_type}]"
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == f"ElevenLabs STT error [{message_type}]"
    assert record.__dict__["lk.pii.data"] == payload
    assert record.exc_info is None


@pytest.mark.parametrize(
    ("message_type", "retryable"),
    [
        ("auth_error", False),
        ("quota_exceeded", False),
        ("input_error", False),
        ("transcriber_error", True),
        ("error", True),
    ],
)
async def test_provider_error_reaches_stream_consumer(message_type: str, retryable: bool) -> None:
    ws = _FakeWS()
    stream = _live_stream(ws)
    if retryable:
        stream._conn_options = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)
    try:
        ws.received.put_nowait({"message_type": message_type, "message": "customer content"})
        with pytest.raises(
            APIConnectionError, match=rf"ElevenLabs STT error \[{message_type}\]"
        ) as exc:
            await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert exc.value.retryable is retryable
        assert stream._num_retries == 0
        assert ws._closed.is_set()
    finally:
        await stream.aclose()


def test_audio_chunk_duration_defaults_to_50ms() -> None:
    assert _stt()._opts.audio_chunk_duration_ms == 50


@pytest.mark.parametrize("duration", [0, -1, 0.5, 100.0, True, False, None, "100"])
def test_invalid_audio_chunk_duration_is_rejected(duration: Any) -> None:
    with pytest.raises(ValueError, match="audio_chunk_duration_ms must be a positive integer"):
        _stt(audio_chunk_duration_ms=duration)


@pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
@pytest.mark.parametrize("duration", [1, 50, 75, 100, 200])
@pytest.mark.parametrize("tail_ms", [0, 1])
async def test_configured_audio_chunks_preserve_audio_and_commit(
    sample_rate: int, duration: int, tail_ms: int
) -> None:
    ws = _FakeWS()
    stream = _live_stream(ws, sample_rate=sample_rate, audio_chunk_duration_ms=duration)
    # Distinct PCM bytes reveal drops, duplication and reordering across input frames.
    total_samples = sample_rate * (max(duration * 2, _MIN_COMMIT_MS) + tail_ms) // 1000
    audio = bytes(i % 251 for i in range(total_samples * 2))
    input_frame_bytes = sample_rate * 20 // 1000 * 2
    chunk_bytes = sample_rate * duration // 1000 * 2
    expected_chunks = [audio[i : i + chunk_bytes] for i in range(0, len(audio), chunk_bytes)]
    try:
        for offset in range(0, len(audio), input_frame_bytes):
            data = audio[offset : offset + input_frame_bytes]
            stream.push_frame(
                rtc.AudioFrame(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(data) // 2,
                )
            )
        # Complete chunks must be released before the caller flushes.
        await _wait_until(lambda: len(ws.sent) >= len(audio) // chunk_bytes)
        stream.flush()
        await _wait_until(lambda: len(ws.sent) >= len(expected_chunks) + 1)

        assert [base64.b64decode(msg["audio_base_64"]) for msg in ws.sent[:-1]] == expected_chunks
        assert [msg["commit"] for msg in ws.sent] == [False] * len(expected_chunks) + [True]
        assert all(msg["sample_rate"] == sample_rate for msg in ws.sent)
        assert ws.sent[-1]["audio_base_64"] == ""
    finally:
        await stream.aclose()


_INVALID_KEY = {"detail": {"code": "invalid_api_key", "message": "Invalid API key"}}


@pytest.mark.parametrize(
    ("body", "message", "parsed"),
    [
        (json.dumps(_INVALID_KEY), "Invalid API key", _INVALID_KEY),
        ("", "Unknown ElevenLabs error", ""),
        ("{not json", "{not json", "{not json"),
    ],
)
async def test_recognize_does_not_retry_a_client_error(
    body: str, message: str, parsed: Any
) -> None:
    requests = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(status=401, text=body, content_type="application/json")

    app = web.Application()
    app.router.add_post("/speech-to-text", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url(""))
        batch_stt = elevenlabs_stt.STT(api_key="bad-key", base_url=url, http_session=session)
        with pytest.raises(APIStatusError) as exc_info:
            await batch_stt.recognize([_frame(100)])

    assert exc_info.value.status_code == 401
    assert exc_info.value.message == message
    assert exc_info.value.body == parsed
    assert requests == 1


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.elevenlabs.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", use_realtime=True, http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(keyterms=["LiveKit"]))
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_recognize_sets_a_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # without a timeout the request falls back to the session's 5 minute default
    timeouts: list[aiohttp.ClientTimeout | None] = []
    post = aiohttp.ClientSession.post

    def recording_post(self: aiohttp.ClientSession, *args: Any, **kwargs: Any) -> Any:
        timeouts.append(kwargs.get("timeout"))
        return post(self, *args, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "post", recording_post)

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"text": "hello", "words": []})

    app = web.Application()
    app.router.add_post("/speech-to-text", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url(""))
        batch_stt = elevenlabs_stt.STT(api_key="test-key", base_url=url, http_session=session)
        await batch_stt.recognize(
            [_frame(100)],
            conn_options=dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, timeout=2.5),
        )

    assert timeouts == [aiohttp.ClientTimeout(total=30, sock_connect=2.5)]


async def test_recognize_language_override_applies_to_one_call() -> None:
    # StreamAdapter passes its stream's language to every recognize() call, so an
    # override must not leak into the STT's own configuration
    sent_languages: list[str] = []

    async def handler(request: web.Request) -> web.Response:
        form = await request.post()
        sent_languages.append(str(form.get("language_code")))
        return web.json_response({"text": "hello", "words": []})

    app = web.Application()
    app.router.add_post("/speech-to-text", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        url = str(server.make_url(""))
        batch_stt = elevenlabs_stt.STT(
            api_key="test-key", base_url=url, language_code="en", http_session=session
        )
        overridden = await batch_stt.recognize([_frame(100)], language="fr")
        default = await batch_stt.recognize([_frame(100)])

    assert sent_languages == ["fr", "en"]
    assert overridden.alternatives[0].language == "fr"
    assert default.alternatives[0].language == "en"
    assert batch_stt._opts.language_code == "en"


@pytest.mark.parametrize("include_timestamps", [False, True])
async def test_end_input_waits_for_all_commits(include_timestamps: bool) -> None:
    ws = _FakeWS()
    stream = _live_stream(
        ws, include_timestamps=include_timestamps, include_language_detection=False
    )
    try:
        stream.push_frame(_frame(300))
        stream.flush()
        stream.push_frame(_frame(300))
        stream.end_input()
        await _wait_until(lambda: sum(message["commit"] for message in ws.sent) == 2)
        for text in ("first", "last"):
            if include_timestamps:
                ws.received.put_nowait(_committed_transcript(text))
                await asyncio.sleep(0.01)
                assert not stream._task.done(), "the basic copy is not the final acknowledgement"
            ws.received.put_nowait(_committed_transcript(text, with_timestamps=include_timestamps))
            await asyncio.sleep(0.01)
            if text == "first":
                assert not stream._task.done(), "a later commit is still pending"
        await asyncio.wait_for(asyncio.shield(stream._task), 1)
        texts = [
            event.alternatives[0].text
            async for event in stream
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert texts == ["first", "last"]
        assert ws._closed.is_set()
    finally:
        await stream.aclose()


@pytest.mark.parametrize("failure", ["write", "close"])
async def test_final_commit_failure_is_reported(failure: str) -> None:
    ws = _FakeWS()
    cause = aiohttp.ClientConnectionError("connection reset")
    send = ws.send_str

    async def send_commit(data: str) -> None:
        if json.loads(data).get("commit") and failure == "write":
            raise cause
        await send(data)

    ws.send_str = send_commit
    stream = _live_stream(ws)
    stream._conn_options = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)
    if failure == "close":

        async def close_message():
            await _wait_until(lambda: any(message.get("commit") for message in ws.sent))
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, 1000, "")

        ws.receive = close_message
        ws.close_code = 1000
    try:
        stream.push_frame(_frame(300))
        stream.end_input()
        with pytest.raises((APIConnectionError, APIStatusError)):
            await asyncio.wait_for(stream._task, 1)
    finally:
        await stream.aclose()


@pytest.mark.usefixtures("job_process")
async def test_server_vad_final_cannot_confirm_the_last_manual_commit(monkeypatch):
    from .stt_harness.discovery import provider_by_name
    from .stt_harness.drivers import make_audio_segments, open_stream, push_audio
    from .stt_harness.protocol import Transcript
    from .stt_harness.server import STTServer, json_message

    provider = provider_by_name("elevenlabs_server_vad")

    class DelayedCommits(provider.protocol):
        async def receive(self, message):
            if json_message(message).get("commit"):
                self.connection.server.final_requested.set()
            else:
                await super().receive(message)

    provider = dataclasses.replace(provider, protocol=DelayedCommits)
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.processing_stalled = True
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        peer = server.connections[0]
        await peer.protocol.send_transcript(Transcript("first", True, 0, 0.4, 0))
        await peer.replies.join()
        await asyncio.sleep(0.01)
        assert not collector.done(), "an earlier VAD final consumed the manual commit"
        await peer.protocol.send_transcript(Transcript("tail", True, 0.4, 0.8, 1))
        await asyncio.wait_for(collector, 1)
        assert [
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["first", "tail"]
        assert not errors
