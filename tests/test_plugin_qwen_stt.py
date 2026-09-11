"""Hermetic tests for the Qwen (Alibaba Cloud Model Studio) STT plugin.

The plugin is a protocol translator, so these tests drive it against an in-process aiohttp
WebSocket server speaking Model Studio's documented realtime events (see
``tests/fake_qwen_realtime.py``) rather than a mocked socket.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.plugins.qwen import STT
from livekit.plugins.qwen.models import REALTIME_BASE_URLS

from .fake_qwen_realtime import FakeASRServer

pytestmark = pytest.mark.unit

NO_RETRY = APIConnectOptions(max_retry=0, timeout=5.0)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: ignore a DASHSCOPE_API_KEY from the dev machine."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)


@pytest.fixture
async def server() -> AsyncIterator[FakeASRServer]:
    srv = FakeASRServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    http = aiohttp.ClientSession()
    yield http
    await http.close()


def make_stt(server: FakeASRServer, session: aiohttp.ClientSession, **kwargs: Any) -> STT:
    return STT(api_key="sk-test", base_url=server.url, http_session=session, **kwargs)


async def collect(
    stt_: STT, frames: Iterable[rtc.AudioFrame] = (), *, flush: bool = False
) -> list[stt.SpeechEvent]:
    """Drive a stream to completion and return the emitted SpeechEvents."""
    stream = stt_.stream(conn_options=NO_RETRY)
    for frame in frames:
        stream.push_frame(frame)
    if flush:
        stream.flush()
    stream.end_input()
    events = [event async for event in stream]
    await stream.aclose()
    return events


def types_of(events: list[stt.SpeechEvent]) -> list[str]:
    return [event.type.value for event in events]


def text_of(event: stt.SpeechEvent) -> str:
    return event.alternatives[0].text


def audio_frame(samples: int, *, value: int = 7) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=int(value).to_bytes(2, "little", signed=True) * samples,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


def transcription_text(text: str, *, stash: str = "", language: str = "") -> dict[str, Any]:
    return {
        "event_id": "srv_t",
        "type": "conversation.item.input_audio_transcription.text",
        "text": text,
        "stash": stash,
        "language": language,
    }


def transcription_completed(transcript: str, *, language: str = "") -> dict[str, Any]:
    return {
        "event_id": "srv_c",
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": transcript,
        "language": language,
    }


def speech_started() -> dict[str, Any]:
    return {"event_id": "srv_s", "type": "input_audio_buffer.speech_started"}


def failed_event(message: str = "garbled frame") -> dict[str, Any]:
    return {
        "event_id": "srv_f",
        "type": "conversation.item.input_audio_transcription.failed",
        "error": {"code": "AudioDecodeFailed", "message": message},
    }


def error_event(**error: Any) -> dict[str, Any]:
    return {"event_id": "srv_e", "type": "error", "error": error}


# --- construction ------------------------------------------------------------------------


def test_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        STT()


def test_reads_the_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    assert STT()._api_key == "sk-env"


def test_region_selects_the_public_endpoint() -> None:
    assert STT(api_key="sk-x")._base_url == REALTIME_BASE_URLS["intl"]
    assert STT(api_key="sk-x", region="cn")._base_url == REALTIME_BASE_URLS["cn"]


def test_base_url_overrides_the_region() -> None:
    url = "wss://ws-1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
    assert STT(api_key="sk-x", region="cn", base_url=url)._base_url == url


def test_provider_and_model_are_labelled() -> None:
    stt_ = STT(api_key="sk-x")
    assert stt_.provider == "Qwen"
    assert stt_.model == "qwen3-asr-flash-realtime"


# --- handshake ---------------------------------------------------------------------------
# Model Studio takes the model in the query string and the key as a bearer header, then
# expects one session.update before any audio.


async def test_model_is_sent_in_the_query_string(server, session) -> None:
    await collect(make_stt(server, session))
    assert server.query.get("model") == "qwen3-asr-flash-realtime"


async def test_api_key_is_sent_as_a_bearer_token(server, session) -> None:
    await collect(make_stt(server, session))
    assert server.headers.get("Authorization") == "Bearer sk-test"


async def test_session_update_declares_pcm_at_16k(server, session) -> None:
    await collect(make_stt(server, session))
    updates = server.events_of_type("session.update")
    assert len(updates) == 1
    config = updates[0]["session"]
    assert config["input_audio_format"] == "pcm"
    # 16000 or 8000 are the only rates the model accepts; LiveKit's default would be
    # 24000, which the server rejects.
    assert config["sample_rate"] == 16000


async def test_language_is_omitted_when_unset_so_mixed_speech_is_detected(server, session) -> None:
    await collect(make_stt(server, session))
    transcription = server.events_of_type("session.update")[0]["session"][
        "input_audio_transcription"
    ]
    assert "language" not in transcription


async def test_language_is_forwarded_when_set(server, session) -> None:
    await collect(make_stt(server, session, language="zh"))
    transcription = server.events_of_type("session.update")[0]["session"][
        "input_audio_transcription"
    ]
    assert transcription["language"] == "zh"


async def test_stream_language_argument_wins_over_the_default(server, session) -> None:
    stt_ = make_stt(server, session, language="zh")
    stream = stt_.stream(language="en", conn_options=NO_RETRY)
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()
    transcription = server.events_of_type("session.update")[0]["session"][
        "input_audio_transcription"
    ]
    assert transcription["language"] == "en"


async def test_update_options_applies_to_the_next_stream(server, session) -> None:
    stt_ = make_stt(server, session)
    stt_.update_options(language="en", vad_silence_duration_ms=1200)
    await collect(stt_)
    config = server.events_of_type("session.update")[0]["session"]
    assert config["input_audio_transcription"]["language"] == "en"
    assert config["turn_detection"] == {"type": "server_vad", "silence_duration_ms": 1200}


async def test_finish_handshake_is_sent_when_input_ends(server, session) -> None:
    await collect(make_stt(server, session))
    assert len(server.events_of_type("session.finish")) == 1


# --- event translation -------------------------------------------------------------------
# Model Studio's own event names, mapped onto LiveKit SpeechEvents.


async def test_speech_started_becomes_start_of_speech(server, session) -> None:
    server.script(speech_started())
    events = await collect(make_stt(server, session))
    assert types_of(events) == ["start_of_speech"]


async def test_text_event_becomes_an_interim_transcript(server, session) -> None:
    server.script(transcription_text("你好"))
    events = await collect(make_stt(server, session))
    assert types_of(events) == ["interim_transcript"]
    assert text_of(events[0]) == "你好"


async def test_interim_transcript_appends_the_tentative_stash(server, session) -> None:
    # `text` is the confirmed prefix, `stash` the part the model may still revise.
    # Both belong in an interim result.
    server.script(transcription_text("我在", stash="写代码"))
    events = await collect(make_stt(server, session))
    assert text_of(events[0]) == "我在写代码"


async def test_completed_becomes_a_final_transcript(server, session) -> None:
    server.script(transcription_completed("I use Python."))
    events = await collect(make_stt(server, session))
    assert types_of(events) == ["final_transcript"]
    assert text_of(events[0]) == "I use Python."


async def test_detected_language_lands_on_the_transcript(server, session) -> None:
    server.script(transcription_completed("你好", language="zh"))
    events = await collect(make_stt(server, session))
    assert events[0].alternatives[0].language == "zh"


async def test_end_of_speech_follows_a_final_transcript_after_speech_started(
    server, session
) -> None:
    # LiveKit's turn detection keys off END_OF_SPEECH; Model Studio never sends one, so
    # the plugin has to close the pair itself.
    server.script(
        speech_started(),
        transcription_text("hello"),
        transcription_completed("Hello there."),
    )
    events = await collect(make_stt(server, session))
    assert types_of(events) == [
        "start_of_speech",
        "interim_transcript",
        "final_transcript",
        "end_of_speech",
    ]


async def test_empty_text_events_are_not_forwarded(server, session) -> None:
    # The model emits a bare `.text` frame with nothing confirmed yet; forwarding it would
    # clear the UI's interim line for no reason.
    server.script(transcription_text(""))
    events = await collect(make_stt(server, session))
    assert events == []


# --- audio forwarding --------------------------------------------------------------------


async def test_pushed_audio_arrives_as_base64_pcm(server, session) -> None:
    frame = audio_frame(1600)
    await collect(make_stt(server, session), frames=[frame])
    assert bytes(server.audio) == frame.data.tobytes()


async def test_audio_is_chunked_into_100ms_appends(server, session) -> None:
    # One 300 ms frame must go up as three appends, not one giant one: a manual-mode
    # event caps at 15 MiB and steady chunking is what keeps transcription latency low.
    await collect(make_stt(server, session), frames=[audio_frame(4800)])
    assert len(server.events_of_type("input_audio_buffer.append")) == 3


async def test_trailing_partial_chunk_is_flushed_not_dropped(server, session) -> None:
    # 150 ms: one full 100 ms chunk plus a 50 ms remainder that would be lost if the
    # byte stream were never flushed.
    frame = audio_frame(2400)
    await collect(make_stt(server, session), frames=[frame])
    assert bytes(server.audio) == frame.data.tobytes()


# --- turn detection ----------------------------------------------------------------------
# The live stream must let Model Studio's VAD close each utterance. LiveKit never calls
# flush() on a streaming STT (its stt_node only push_frame()s and aclose()s), so a
# manual-commit stream would never commit and never yield a transcript. Manual mode is
# reserved for the one-shot recognize() path, where end_input() does flush.


async def test_streaming_uses_server_vad_by_default(server, session) -> None:
    await collect(make_stt(server, session))
    config = server.events_of_type("session.update")[0]["session"]
    assert config["turn_detection"] == {"type": "server_vad"}


async def test_streaming_forwards_vad_tuning(server, session) -> None:
    await collect(make_stt(server, session, vad_silence_duration_ms=2500, vad_threshold=0.3))
    config = server.events_of_type("session.update")[0]["session"]
    assert config["turn_detection"] == {
        "type": "server_vad",
        "silence_duration_ms": 2500,
        "threshold": 0.3,
    }


async def test_streaming_never_commits_even_when_flushed(server, session) -> None:
    await collect(make_stt(server, session), frames=[audio_frame(1600)], flush=True)
    assert server.events_of_type("input_audio_buffer.commit") == []


async def test_recognize_disables_server_vad_and_commits_the_whole_buffer(server, session) -> None:
    # One-shot recognition has a known utterance boundary, so it commits explicitly
    # instead of waiting on the server's silence window.
    server.emit_script_after("input_audio_buffer.commit")
    server.script(transcription_completed("One shot."))
    await make_stt(server, session).recognize(audio_frame(2400), conn_options=NO_RETRY)

    config = server.events_of_type("session.update")[0]["session"]
    assert config["turn_detection"] is None
    # 150 ms leaves a 50 ms remainder in the chunker; every byte of it must be appended
    # before the single commit.
    appended_before_commit = 0
    commits = 0
    for event in server.client_events:
        if event["type"] == "input_audio_buffer.commit":
            commits += 1
            break
        if event["type"] == "input_audio_buffer.append":
            appended_before_commit += len(base64.b64decode(event["audio"]))
    assert commits == 1
    assert appended_before_commit == 2400 * 2


# --- teardown ----------------------------------------------------------------------------
# LiveKit tears a live STT stream down with aclose(), never end_input(), which cancels
# _run() before send() reaches session.finish. Model Studio counts a socket dropped
# without the handshake as a failed request, so it has to happen on the way out: bounded,
# and never on error paths where the FallbackAdapter is waiting to move on.


async def test_aclose_still_completes_the_finish_handshake(server, session) -> None:
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(1600))
    await asyncio.sleep(0.2)  # let the audio go out first
    await stream.aclose()
    assert len(server.events_of_type("session.finish")) == 1


async def test_aclose_against_a_wedged_server_still_returns(server, session) -> None:
    server.ignore_finish()
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(1600))
    await asyncio.sleep(0.2)
    await asyncio.wait_for(stream.aclose(), timeout=10)


async def test_a_server_error_closes_without_a_finish_handshake(server, session) -> None:
    # After an error the FallbackAdapter wants the next provider now; a courtesy
    # handshake would only delay it.
    server.script(error_event(message="boom"))
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(1600))
    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()
    assert server.events_of_type("session.finish") == []


# --- per-utterance failure ---------------------------------------------------------------


async def test_a_failed_utterance_is_logged_and_the_stream_carries_on(
    server, session, caplog: pytest.LogCaptureFixture
) -> None:
    # `.failed` is per-utterance. Tearing the socket down for it would make the
    # FallbackAdapter abandon the provider over one bad segment, so the plugin logs it
    # and keeps listening. The provider payload rides in the PII-tagged extra, never in
    # the message body (REVIEW.md).
    failed = {
        "event_id": "srv_f",
        "type": "conversation.item.input_audio_transcription.failed",
        "error": {"code": "AudioDecodeFailed", "message": "garbled frame"},
    }
    server.script(failed, transcription_completed("still here"))
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.qwen"):
        events = await collect(make_stt(server, session))

    assert types_of(events) == ["final_transcript"]
    assert text_of(events[0]) == "still here"
    warning = next(
        r
        for r in caplog.records
        if r.name == "livekit.plugins.qwen" and r.levelno == logging.WARNING
    )
    assert "garbled frame" not in warning.getMessage()
    assert warning.__dict__["lk.pii.data"]["error"]["message"] == "garbled frame"


# --- failures ----------------------------------------------------------------------------


async def drain(stt_: STT) -> None:
    stream = stt_.stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(1600))
    stream.end_input()
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


async def test_server_error_event_surfaces_as_an_api_error(server, session) -> None:
    server.script(error_event(code="InvalidParameter", message="bad sample_rate"))
    with pytest.raises(APIStatusError) as exc_info:
        await drain(make_stt(server, session))
    assert "bad sample_rate" in str(exc_info.value)


async def test_a_request_error_is_not_retried(server, session) -> None:
    # Model Studio reports failures as an error event, not an HTTP status. A bad request
    # will be just as bad on the second try; retrying it three times only delays the
    # failover to the next STT.
    server.script(
        error_event(
            type="invalid_request_error", code="InvalidParameter", message="unsupported language"
        )
    )
    with pytest.raises(APIStatusError) as exc_info:
        await drain(make_stt(server, session))
    assert exc_info.value.retryable is False


async def test_a_server_error_stops_being_retryable_once_audio_is_consumed(server, session) -> None:
    # Classification itself is unchanged; `status_error_from` still calls a server error
    # retryable (see tests/test_plugin_qwen_realtime.py). What changes is the stream: audio
    # already pulled off the channel cannot be replayed, so a retry would transcribe
    # silence. The error is handed to the FallbackAdapter instead.
    server.script(error_event(type="server_error", message="upstream busy"))
    with pytest.raises(APIStatusError) as exc_info:
        await drain(make_stt(server, session))
    assert exc_info.value.retryable is False
    assert "upstream busy" in str(exc_info.value)


async def test_hangup_before_finish_surfaces_as_a_connection_error(server, session) -> None:
    # A silent close would otherwise look like a clean end of stream and the
    # FallbackAdapter would never try the next provider.
    server.close_after_script()
    with pytest.raises(APIConnectionError):
        await drain(make_stt(server, session))


async def test_unreachable_endpoint_surfaces_as_a_connection_error(session) -> None:
    stt_ = STT(
        api_key="sk-test",
        base_url="ws://127.0.0.1:1/api-ws/v1/realtime",
        http_session=session,
    )
    with pytest.raises(APIConnectionError):
        await drain(stt_)


async def test_a_wedged_upstream_times_out_instead_of_hanging(server, session) -> None:
    # conn_options.timeout only covers the connect. A server that holds the socket open
    # but never answers session.finish would otherwise stall the stream, and the session
    # behind it, forever.
    server.ignore_finish()
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(drain(make_stt(server, session, finish_timeout=0.3)), timeout=10)


# --- usage -------------------------------------------------------------------------------
# Model Studio bills ASR per second of input audio, so the plugin reports duration rather
# than tokens.


def usage_seconds(events: list[stt.SpeechEvent]) -> float:
    return sum(
        e.recognition_usage.audio_duration
        for e in events
        if e.type is stt.SpeechEventType.RECOGNITION_USAGE and e.recognition_usage
    )


async def test_recognition_usage_reports_the_audio_duration_sent(server, session) -> None:
    server.script(transcription_completed("hello"))
    events = await collect(make_stt(server, session), frames=[audio_frame(1600), audio_frame(1600)])
    assert usage_seconds(events) == pytest.approx(0.2, abs=1e-3)


async def test_usage_is_reported_per_final_so_aclose_cannot_lose_it(server, session) -> None:
    # In a live session LiveKit tears the stream down with aclose(), which cancels _run()
    # outright; anything emitted only after the socket drains never happens. Usage has to
    # ride on each final transcript.
    server.emit_script_after("input_audio_buffer.append", count=2)
    server.script(transcription_completed("hello"))
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(1600))
    stream.push_frame(audio_frame(1600))

    seen: list[stt.SpeechEvent] = []
    while not any(e.type is stt.SpeechEventType.RECOGNITION_USAGE for e in seen):
        seen.append(await asyncio.wait_for(stream.__anext__(), timeout=5))
    await stream.aclose()

    assert usage_seconds(seen) == pytest.approx(0.2, abs=1e-3)


async def test_usage_is_not_double_counted_across_finals(server, session) -> None:
    server.emit_script_after("input_audio_buffer.append", count=2)
    server.script(transcription_completed("one"), transcription_completed("two"))
    events = await collect(make_stt(server, session), frames=[audio_frame(1600), audio_frame(1600)])
    assert usage_seconds(events) == pytest.approx(0.2, abs=1e-3)


# --- one-shot recognize ------------------------------------------------------------------
# The FallbackAdapter probes a provider with recognize() before trusting it, so the
# one-shot path has to work even though the transport streams.


async def test_recognize_returns_the_final_transcript(server, session) -> None:
    server.script(transcription_completed("One shot.", language="en"))
    event = await make_stt(server, session).recognize(audio_frame(1600), conn_options=NO_RETRY)
    assert event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "One shot."
    assert event.alternatives[0].language == "en"


async def test_recognize_with_no_transcript_returns_an_empty_final(server, session) -> None:
    event = await make_stt(server, session).recognize(audio_frame(1600), conn_options=NO_RETRY)
    assert event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == ""


# --- review follow-ups (livekit/agents#7224) ----------------------------------------------


async def test_a_retryable_error_after_audio_is_not_retried(server, session) -> None:
    # `RecognizeStream` keeps no replay buffer, so audio already pulled off `_input_ch` is
    # gone for good. Retrying would open a fresh socket with no audio, finish cleanly, and
    # turn a provider failure into an empty transcript. Hand the error to the
    # FallbackAdapter instead.
    server.script(error_event(type="server_error", message="upstream busy"))
    retrying = APIConnectOptions(max_retry=3, retry_interval=0.0, timeout=5.0)
    stream = make_stt(server, session).stream(conn_options=retrying)
    stream.push_frame(audio_frame(1600))
    stream.end_input()

    with pytest.raises(APIError) as exc_info:
        async for _ in stream:
            pass
    await stream.aclose()

    assert exc_info.value.retryable is False
    assert server.connections == 1


async def test_a_connect_failure_before_any_audio_is_still_retried(session) -> None:
    # The guard above must not disable retries for failures that predate any audio, which
    # are exactly the ones a retry can fix.
    stt_ = STT(
        api_key="sk-test",
        base_url="ws://127.0.0.1:1/api-ws/v1/realtime",
        http_session=session,
    )
    stream = stt_.stream(
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)
    )
    stream.push_frame(audio_frame(1600))
    stream.end_input()
    with pytest.raises(APIConnectionError):
        async for _ in stream:
            pass
    await stream.aclose()
    assert stream._num_retries == 2


async def test_a_failed_utterance_closes_the_open_turn(server, session) -> None:
    # LiveKit commits a user turn on END_OF_SPEECH when turn_detection="stt". Logging the
    # failure and moving on would leave that turn open forever, so the plugin promotes the
    # interim text to a final and closes the pair itself.
    server.script(speech_started(), transcription_text("hello"), failed_event())
    events = await collect(make_stt(server, session))

    assert types_of(events) == [
        "start_of_speech",
        "interim_transcript",
        "final_transcript",
        "end_of_speech",
    ]
    assert text_of(events[2]) == "hello"


async def test_a_failed_utterance_without_an_open_turn_emits_nothing(server, session) -> None:
    # No speech_started means no turn to close, so a failure stays a log line.
    server.script(failed_event(), transcription_completed("still here"))
    events = await collect(make_stt(server, session))
    assert types_of(events) == ["final_transcript"]
    assert text_of(events[0]) == "still here"


async def test_usage_is_reported_when_a_stream_closes_without_a_final(server, session) -> None:
    # aclose() cancels _run before the normal epilogue. Audio streamed since the last final
    # would otherwise never be billed, and a stream that produced no final at all would
    # report nothing.
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    collected: list[stt.SpeechEvent] = []

    async def drain() -> None:
        async for event in stream:
            collected.append(event)

    task = asyncio.create_task(drain())
    stream.push_frame(audio_frame(1600))
    stream.push_frame(audio_frame(1600))
    for _ in range(200):
        if len(server.events_of_type("input_audio_buffer.append")) >= 2:
            break
        await asyncio.sleep(0.02)
    await stream.aclose()
    await asyncio.wait_for(task, timeout=5)

    assert usage_seconds(collected) == pytest.approx(0.2, abs=1e-3)


async def test_usage_is_reported_when_recognition_fails_after_audio(server, session) -> None:
    # Model Studio billed the audio it received before the error, so it belongs in the
    # usage metrics whether or not a transcript ever came back.
    server.emit_script_after("input_audio_buffer.append", count=2)
    server.script(error_event(type="server_error", message="upstream busy"))
    stream = make_stt(server, session).stream(conn_options=NO_RETRY)
    collected: list[stt.SpeechEvent] = []

    async def drain() -> None:
        async for event in stream:
            collected.append(event)

    task = asyncio.create_task(drain())
    stream.push_frame(audio_frame(1600))
    stream.push_frame(audio_frame(1600))
    with pytest.raises(APIStatusError):
        await asyncio.wait_for(task, timeout=5)
    await stream.aclose()

    assert usage_seconds(collected) == pytest.approx(0.2, abs=1e-3)
