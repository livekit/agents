"""Unit tests for ElevenLabs STT plugin configuration."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from multidict import CIMultiDict
from yarl import URL

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import stt as elevenlabs_stt
from livekit.plugins.elevenlabs._utils import trace_id_from_headers

pytestmark = pytest.mark.plugin("elevenlabs")


class _EventSink:
    def __init__(self) -> None:
        self.events: list[stt.SpeechEvent] = []

    def send_nowait(self, event: stt.SpeechEvent) -> None:
        self.events.append(event)


def _new_stream(*, server_vad=NOT_GIVEN) -> elevenlabs_stt.SpeechStream:
    stream = object.__new__(elevenlabs_stt.SpeechStream)
    stream._opts = elevenlabs_stt.STTOptions(
        model_id="scribe_v2_realtime",
        api_key="test-key",
        base_url=elevenlabs_stt.API_BASE_URL_V1,
        language_code=None,
        tag_audio_events=True,
        include_timestamps=False,
        sample_rate=16000,
        server_vad=server_vad,
        keyterms=NOT_GIVEN,
        no_verbatim=False,
        enable_logging=True,
        previous_text=None,
    )
    stream._language = None
    stream._event_ch = _EventSink()
    stream._speaking = False
    stream._start_time_offset = 0.0
    return stream


def _committed_transcript(text: str) -> dict:
    return {
        "message_type": "committed_transcript",
        "text": text,
        "words": [
            {"text": text, "start": 0.1, "end": 0.4},
        ]
        if text
        else [],
    }


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


class _FakeWS:
    """Records outgoing messages. receive() parks so recv_task stays alive."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self) -> Any:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self._closed.set()


def _live_stream(ws: _FakeWS) -> elevenlabs_stt.SpeechStream:
    """A real SpeechStream running its real _run loop against a fake socket."""
    instance = elevenlabs_stt.STT(api_key="test-key", model="scribe_v2_realtime")
    opts = dataclasses.replace(instance._opts, sample_rate=16000)
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
    # 50ms is exactly one repack chunk, so AudioByteStream.flush() returns no frames.
    # The commit must still go out: Scribe v2 does not finalize a turn without one, and
    # gating it on the per-frame loop dropped it here (roughly 1 flush in 5).
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(50))
        await _wait_until(lambda: len(ws.sent) == 1)

        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 2)

        assert [msg["commit"] for msg in ws.sent] == [False, True]
        assert ws.sent[-1]["audio_base_64"] == ""
    finally:
        await stream.aclose()


async def test_flush_commits_after_the_buffered_audio() -> None:
    # a partial chunk is still pending: it has to reach the server before the commit,
    # otherwise the tail of the turn is transcribed against the next one
    ws = _FakeWS()
    stream = _live_stream(ws)
    try:
        stream.push_frame(_frame(30))
        stream.flush()
        await _wait_until(lambda: len(ws.sent) == 2)

        assert [msg["commit"] for msg in ws.sent] == [False, True]
        assert ws.sent[0]["audio_base_64"] != ""
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
