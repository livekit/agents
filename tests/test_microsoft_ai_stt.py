from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from examples.microsoft import microsoft_ai_echo as echo_example, microsoft_ai_smoke as smoke
from livekit import rtc
from livekit.agents import (
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    JobContext,
    StopResponse,
    inference,
    llm,
    stt,
    vad,
)
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai
from livekit.plugins.microsoft_ai._http import HTTPClient

from .fake_io import FakeAudioOutput
from .microsoft_ai_fakes import (
    DUMMY_CONFIG,
    FakeResponse,
    FakeSocket,
    ScriptedVAD,
    audio_frame,
    fake_session,
    no_http_session as no_http_session,
    no_network as no_network,
    wav_bytes,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

OPTIONS = APIConnectOptions(max_retry=0, timeout=0.5)
STT_URL = "wss://stt.example.invalid/v1/realtime?intent=transcription&deployment=dummy"


def provider(
    socket: FakeSocket,
    *,
    detector: vad.VAD | None = None,
    max_buffered_audio: float = 5.0,
) -> tuple[microsoft_ai.STT, MagicMock]:
    session = fake_session()
    session.ws_connect = AsyncMock(return_value=socket)
    instance = microsoft_ai.STT(
        vad=detector,
        url=STT_URL,
        model="test-transcriber",
        api_key="dummy-stt-key",
        http_session=session,
        max_buffered_audio=max_buffered_audio,
    )
    return instance, session


async def collect(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    async def run() -> list[stt.SpeechEvent]:
        return [event async for event in stream]

    return await asyncio.wait_for(run(), 3.0)


def finals(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        event.alternatives[0].text
        for event in events
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]


async def next_event(stream: stt.RecognizeStream) -> stt.SpeechEvent:
    return await asyncio.wait_for(stream.__anext__(), 1.0)


def test_no_provider_defaults_or_extra_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    with pytest.raises(ValueError, match="MICROSOFT_AI_STT_MODEL"):
        microsoft_ai.STT(vad=None)
    with pytest.raises(ValueError, match="MICROSOFT_AI_STT_URL"):
        microsoft_ai.STT(vad=None, model="test")
    with pytest.raises(ValueError, match="MICROSOFT_AI_STT_API_KEY"):
        microsoft_ai.STT(vad=None, model="test", url=STT_URL)
    assert not hasattr(microsoft_ai, "LLM")
    assert not hasattr(microsoft_ai, "realtime")
    assert not hasattr(microsoft_ai.STT, "with_azure")


async def test_handshake_gates_audio_and_sends_exact_config() -> None:
    socket = FakeSocket(auto_update=False)
    instance, session = provider(socket)
    async with instance, instance.stream(language="en", conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        stream.end_input()
        await socket.wait_sent("session.update")
        assert len(socket.sent) == 1
        assert socket.sent[0] == {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {"model": "test-transcriber", "language": "en"},
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }
        socket.emit({"type": "session.updated"})
        assert finals(await collect(stream)) == ["turn 1"]
        assert instance.capabilities.streaming
        assert instance.capabilities.interim_results
        assert not instance.capabilities.offline_recognize
        assert not instance.capabilities.aligned_transcript
        assert not instance.capabilities.diarization
        assert instance.provider == "Microsoft AI"
    session.ws_connect.assert_awaited_once()
    args, kwargs = session.ws_connect.call_args
    assert args == (STT_URL,)
    assert kwargs["headers"]["Authorization"] == "Bearer dummy-stt-key"
    assert socket.closed
    session.close.assert_not_awaited()


async def test_batch_recognition_is_explicitly_unsupported() -> None:
    instance, session = provider(FakeSocket())
    async with instance:
        with pytest.raises(NotImplementedError, match="not batch"):
            await instance.recognize(audio_frame(), conn_options=OPTIONS)
    session.ws_connect.assert_not_called()


async def test_environment_and_explicit_custom_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeSocket()
    session = fake_session()
    session.ws_connect = AsyncMock(return_value=socket)
    monkeypatch.setenv("MICROSOFT_AI_STT_URL", STT_URL)
    monkeypatch.setenv("MICROSOFT_AI_STT_MODEL", "environment-model")
    monkeypatch.setenv("MICROSOFT_AI_STT_API_KEY", "unused-environment-key")
    instance = microsoft_ai.STT(vad=None, headers={"api-key": "dummy"}, http_session=session)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.end_input()
        assert await collect(stream) == []
    headers = session.ws_connect.call_args.kwargs["headers"]
    assert headers == {"api-key": "dummy", "User-Agent": "LiveKit Agents"}
    assert instance.model == "environment-model"
    with pytest.raises(ValueError, match="either api_key or headers"):
        microsoft_ai.STT(vad=None, api_key="dummy", headers={})


async def test_api_key_from_selected_file_uses_exact_ga_transcription_url(tmp_path: Path) -> None:
    url = "wss://stt.example.invalid/openai/v1/realtime?intent=transcription"
    config = tmp_path / "endpoints.env"
    config.write_text(
        f"MICROSOFT_AI_STT_URL={url}\n"
        "MICROSOFT_AI_STT_MODEL=dummy-deployment\n"
        "MICROSOFT_AI_STT_API_KEY=dummy-raw-key\n"
        "MICROSOFT_AI_STT_AUTH_HEADER=api-key\n",
        encoding="utf-8",
    )
    socket = FakeSocket()
    http = fake_session()
    http.ws_connect = AsyncMock(return_value=socket)
    instance = microsoft_ai.STT(vad=None, env_file=config, http_session=http)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(803))
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
    args, kwargs = http.ws_connect.call_args
    assert args == (url,)
    assert kwargs["headers"] == {"api-key": "dummy-raw-key", "User-Agent": "LiveKit Agents"}
    assert "Authorization" not in kwargs["headers"]
    assert "params" not in kwargs
    assert socket.sent[0]["session"]["audio"]["input"]["transcription"] == {
        "model": "dummy-deployment"
    }
    assert socket.closed


async def test_api_key_auth_failure_never_falls_back_or_discloses_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    http = fake_session()
    http.ws_connect = AsyncMock(
        side_effect=aiohttp.WSServerHandshakeError(
            request_info=MagicMock(),
            history=(),
            status=401,
            message="dummy-private-key-and-url",
        )
    )
    instance = microsoft_ai.STT(
        vad=None,
        url=STT_URL,
        model="test",
        api_key="dummy-private-key",
        auth_header="api-key",
        http_session=http,
    )
    async with instance, instance.stream(conn_options=APIConnectOptions(max_retry=3)) as stream:
        with pytest.raises(APIStatusError) as caught:
            await collect(stream)
    assert caught.value.status_code == 401
    http.ws_connect.assert_awaited_once()
    assert http.ws_connect.call_args.kwargs["headers"]["api-key"] == "dummy-private-key"
    assert "Authorization" not in http.ws_connect.call_args.kwargs["headers"]
    assert "dummy-private-key" not in str(caught.value)
    assert "dummy-private-key" not in caplog.text
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("samples", [1, 157, 799, 800, 801, 2417])
async def test_end_input_drains_exact_pcm_tail_without_padding(samples: int) -> None:
    socket = FakeSocket()
    instance, _ = provider(socket)
    frame = audio_frame(samples)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        events = await collect(stream)
    assert socket.commits == [frame.data.tobytes()]
    assert finals(events) == ["turn 1"]
    chunks = [
        base64.b64decode(event["audio"])
        for event in socket.sent
        if event["type"] == "input_audio_buffer.append"
    ]
    assert all(len(chunk) == 1600 for chunk in chunks[:-1])
    assert len(chunks[-1]) == ((samples - 1) % 800 + 1) * 2
    usage = [event.recognition_usage for event in events if event.recognition_usage]
    assert len(usage) == 1 and usage[0].audio_duration == samples / 16000


async def test_revisions_deltas_and_final_are_not_duplicated() -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        await socket.wait_sent("input_audio_buffer.append")
        socket.transcript("intermediate", intermediate="helo", event_id="revision-1")
        assert (await next_event(stream)).type == stt.SpeechEventType.START_OF_SPEECH
        assert (await next_event(stream)).alternatives[0].text == "helo"
        socket.transcript("intermediate", intermediate="hello")
        assert (await next_event(stream)).alternatives[0].text == "hello"
        socket.transcript("delta", delta="hello ", event_id="delta-1")
        assert (await next_event(stream)).alternatives[0].text == "hello "
        socket.transcript("delta", delta="hello ", event_id="delta-1")
        socket.transcript("intermediate", intermediate="wurld")
        assert (await next_event(stream)).alternatives[0].text == "hello wurld"
        socket.transcript("intermediate", intermediate="world")
        assert (await next_event(stream)).alternatives[0].text == "hello world"
        socket.transcript("delta", delta="world")
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        socket.transcript("completed", transcript="hello world")
        socket.transcript("completed", transcript="hello world")
        events = await collect(stream)
    assert finals(events) == ["hello world"]
    assert not any(event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for event in events)
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH


async def test_flush_keeps_socket_open_for_multiple_utterances() -> None:
    socket = FakeSocket()
    instance, session = provider(socket)
    first, second = audio_frame(917), audio_frame(83, value=b"\x01\x00")
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(first)
        stream.flush()
        first_events = []
        while True:
            event = await next_event(stream)
            first_events.append(event)
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                break
        assert finals(first_events) == ["turn 1"]
        assert not socket.closed
        socket.transcript("completed", transcript="turn 1", item="item-1")
        stream.push_frame(second)
        stream.flush()
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 2"]
    assert socket.commits == [first.data.tobytes(), second.data.tobytes()]
    session.ws_connect.assert_awaited_once()


async def test_vad_drains_pending_frames_before_each_commit_under_backpressure() -> None:
    detector = ScriptedVAD(
        {
            512: vad.VADEventType.START_OF_SPEECH,
            1024: vad.VADEventType.END_OF_SPEECH,
            1536: vad.VADEventType.START_OF_SPEECH,
            2048: vad.VADEventType.END_OF_SPEECH,
        }
    )
    socket = FakeSocket()
    socket.append_gate = asyncio.Event()
    instance, _ = provider(socket, detector=detector)
    first, second = audio_frame(1024), audio_frame(1024, value=b"\x02\x00")
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(first)
        stream.push_frame(second)
        stream.end_input()
        await asyncio.wait_for(socket.append_started.wait(), 1.0)
        assert not socket.commits
        socket.append_gate.set()
        assert finals(await collect(stream)) == ["turn 1", "turn 2"]
    assert socket.commits == [first.data.tobytes(), second.data.tobytes()]
    assert all(item.closed and item._task.done() for item in detector.streams)


async def test_vad_end_input_preserves_incomplete_inference_window() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH})
    instance, _ = provider(socket, detector=detector)
    frame = audio_frame(1307)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
    assert socket.commits == [frame.data.tobytes()]


async def test_bundled_vad_timestamps_and_tail_match_the_current_sdk() -> None:
    socket = FakeSocket()
    instance, _ = provider(socket, detector=inference.VAD(model="silero"))
    frame = audio_frame(1307)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
    assert socket.commits == [frame.data.tobytes()]


async def test_manual_flush_resets_vad_without_merging_turns() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH})
    instance, _ = provider(socket, detector=detector)
    frames = [audio_frame(917), audio_frame(613, value=b"\x03\x00")]
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frames[0])
        stream.flush()
        stream.push_frame(frames[1])
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1", "turn 2"]
    assert socket.commits == [frame.data.tobytes() for frame in frames]
    assert len(detector.streams) == 2


async def test_resampling_drains_the_sdk_resampler_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = FakeSocket()
    instance, _ = provider(socket)
    frame = audio_frame(4817, sample_rate=48000)
    reference = rtc.AudioResampler(48000, 16000, quality=rtc.AudioResamplerQuality.HIGH)
    resampled: list[rtc.AudioFrame] = []
    original_push, original_flush = reference.push, reference.flush

    def push(frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        frames = original_push(frame)
        resampled.extend(frames)
        return frames

    def flush() -> list[rtc.AudioFrame]:
        frames = original_flush()
        resampled.extend(frames)
        return frames

    # The SDK dithers PCM output. Capture the actual instance's frames rather than
    # byte-comparing two independent resamplers with different dither.
    monkeypatch.setattr(reference, "push", push)
    monkeypatch.setattr(reference, "flush", flush)
    monkeypatch.setattr(rtc, "AudioResampler", lambda *args, **kwargs: reference)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        await collect(stream)
    assert socket.commits == [b"".join(f.data.tobytes() for f in resampled)]
    assert sum(f.samples_per_channel for f in resampled) == round(4817 / 3)
    assert all(f.sample_rate == 16000 and f.num_channels == 1 for f in resampled)


async def test_incomplete_backend_tail_is_not_fabricated_as_final() -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(901))
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.transcript("delta", delta="hello ")
        socket.transcript("intermediate", intermediate="tail")
        socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        socket.transcript("completed", transcript="hello ")
        with pytest.raises(APIError, match="audio-tail contract") as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert len(socket.commits[0]) == 1802


@pytest.mark.parametrize("ack", [False, True])
async def test_missing_final_or_ack_has_finite_timeout(ack: bool) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    options = APIConnectOptions(timeout=0.03, max_retry=3, retry_interval=0)
    async with instance, instance.stream(conn_options=options) as stream:
        stream.push_frame(audio_frame())
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        if ack:
            socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        with pytest.raises(APITimeoutError) as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert socket.closed
    assert len(socket.commits) == 1


@pytest.mark.parametrize(
    "event",
    [
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "missing id"},
        {
            "type": "conversation.item.input_audio_transcription.intermediate",
            "item_id": "one",
            "intermediate": 1,
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "one",
            "transcript": "unsolicited",
        },
        {"type": "input_audio_buffer.committed", "item_id": "one"},
    ],
)
async def test_invalid_item_events_fail_explicitly(event: dict[str, object]) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        await socket.wait_sent("input_audio_buffer.append")
        socket.emit(event)
        with pytest.raises(APIError) as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert socket.closed


async def test_item_ids_cannot_change_before_completion() -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        await socket.wait_sent("input_audio_buffer.append")
        socket.transcript("delta", item="one", delta="first")
        socket.transcript("delta", item="two", delta="second")
        with pytest.raises(APIError, match="changed item"):
            await collect(stream)


async def test_disconnect_after_audio_does_not_retry_or_replay() -> None:
    socket = FakeSocket()
    instance, session = provider(socket)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=3, timeout=0.2)) as stream,
    ):
        stream.push_frame(audio_frame(817))
        await socket.wait_sent("input_audio_buffer.append")
        socket.disconnect()
        with pytest.raises(APIConnectionError) as caught:
            await collect(stream)
        assert not caught.value.retryable
    session.ws_connect.assert_awaited_once()
    assert not socket.commits


async def test_connect_retry_preserves_queued_audio_and_sanitizes_transport_errors() -> None:
    socket = FakeSocket()
    instance, session = provider(socket)
    session.ws_connect.side_effect = [
        aiohttp.ClientConnectionError("do-not-log-this-dummy-endpoint"),
        socket,
    ]
    errors = []
    instance.on("error", errors.append)
    frame = audio_frame(33)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=1, timeout=0.2)) as stream,
    ):
        stream.push_frame(frame)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
    assert session.ws_connect.await_count == 2
    assert socket.commits == [frame.data.tobytes()]
    assert errors[0].recoverable
    assert "do-not-log" not in str(errors[0].error)


async def test_handshake_timeout_retry_budget_is_finite() -> None:
    instance, session = provider(FakeSocket())
    sockets = [FakeSocket(auto_update=False) for _ in range(3)]
    session.ws_connect.side_effect = sockets
    async with (
        instance,
        instance.stream(
            conn_options=APIConnectOptions(max_retry=2, timeout=0.01, retry_interval=0)
        ) as stream,
    ):
        stream.end_input()
        with pytest.raises(APITimeoutError):
            await collect(stream)
    assert session.ws_connect.await_count == 3
    assert all(socket.closed for socket in sockets)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 503])
async def test_handshake_http_error_status_is_preserved(status: int) -> None:
    instance, session = provider(FakeSocket())
    session.ws_connect.side_effect = aiohttp.WSServerHandshakeError(
        request_info=MagicMock(),
        history=(),
        status=status,
        message="do-not-log-this-dummy-secret",
    )
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        with pytest.raises(APIStatusError) as caught:
            await collect(stream)
    assert caught.value.status_code == status
    assert caught.value.retryable == (status in (429, 503))
    assert "do-not-log" not in str(caught.value)


@pytest.mark.parametrize(
    ("code", "status"),
    [("invalid_api_key", 401), ("rate_limit_exceeded", 429), ("content_filter", 403)],
)
async def test_provider_errors_preserve_status_without_echoing_body(code: str, status: int) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, session = provider(socket)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=2, timeout=0.2)) as stream,
    ):
        stream.push_frame(audio_frame())
        await socket.wait_sent("input_audio_buffer.append")
        socket.emit({"type": "error", "error": {"code": code, "message": "do-not-log-transcript"}})
        with pytest.raises(APIStatusError) as caught:
            await collect(stream)
    assert caught.value.status_code == status
    assert not caught.value.retryable
    assert "do-not-log" not in str(caught.value)
    session.ws_connect.assert_awaited_once()


async def test_audio_overflow_is_bounded_and_surfaces_to_both_callers() -> None:
    socket = FakeSocket()
    socket.append_gate = asyncio.Event()
    instance, _ = provider(socket, max_buffered_audio=0.06)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(800))
        await asyncio.wait_for(socket.append_started.wait(), 1.0)
        with pytest.raises(APIConnectionError, match="buffer is full"):
            stream.push_frame(audio_frame(800))
        socket.append_gate.set()
        with pytest.raises(APIConnectionError, match="buffer is full"):
            await collect(stream)
    assert not socket.commits


async def test_input_format_and_timeout_validation() -> None:
    instance, _ = provider(FakeSocket())
    async with instance:
        with pytest.raises(ValueError, match="greater than zero"):
            instance.stream(conn_options=APIConnectOptions(timeout=0))
        async with instance.stream(conn_options=OPTIONS) as stream:
            with pytest.raises(ValueError, match="mono"):
                stream.push_frame(
                    rtc.AudioFrame(
                        data=b"\0" * 3200,
                        sample_rate=16000,
                        num_channels=2,
                        samples_per_channel=800,
                    )
                )
            stream.push_frame(audio_frame())
            with pytest.raises(ValueError, match="sample rate"):
                stream.push_frame(audio_frame(sample_rate=48000))


async def test_aclose_cancels_without_committing_or_late_events() -> None:
    socket = FakeSocket(auto_commit=False)
    instance, session = provider(socket)
    stream = instance.stream(conn_options=OPTIONS)
    stream.push_frame(audio_frame())
    await socket.wait_sent("input_audio_buffer.append")
    socket.transcript("delta", delta="queued")
    await instance.aclose()
    socket.transcript("completed", transcript="late")
    assert await collect(stream) == []
    assert not socket.commits
    assert socket.closed and stream._task.done() and stream._metrics_task.done()
    session.close.assert_not_awaited()
    with pytest.raises(RuntimeError, match="closed"):
        instance.stream()


@pytest.mark.parametrize("body", ["{", "[]", '{"event":"missing-type"}'])
async def test_malformed_json_events_are_terminal(body: str) -> None:
    socket = FakeSocket(auto_update=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        await socket.wait_sent("session.update")
        socket.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, body, ""))
        with pytest.raises(APIError) as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert socket.closed


async def test_realtime_rate_limit_before_audio_can_retry() -> None:
    first = FakeSocket(auto_update=False)
    first.emit({"type": "error", "error": {"status_code": 429}})
    second = FakeSocket()
    instance, session = provider(first)
    session.ws_connect.side_effect = [first, second]
    options = APIConnectOptions(max_retry=1, timeout=0.2)
    async with instance, instance.stream(conn_options=options) as stream:
        stream.push_frame(audio_frame(17))
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
    assert session.ws_connect.await_count == 2
    assert first.closed and second.closed


async def test_authentication_failure_does_not_spend_retry_budget() -> None:
    socket = FakeSocket(auto_update=False)
    socket.emit({"type": "error", "error": {"code": "invalid_api_key"}})
    instance, session = provider(socket)
    async with instance, instance.stream(conn_options=APIConnectOptions(max_retry=3)) as stream:
        with pytest.raises(APIStatusError) as caught:
            await collect(stream)
    assert caught.value.status_code == 401
    session.ws_connect.assert_awaited_once()


async def test_aclose_during_handshake_is_immediate() -> None:
    socket = FakeSocket(auto_update=False)
    instance, session = provider(socket)
    stream = instance.stream(conn_options=APIConnectOptions(timeout=10))
    await socket.wait_sent("session.update")
    await asyncio.wait_for(instance.aclose(), 0.5)
    assert await collect(stream) == []
    assert socket.closed
    assert len(socket.sent) == 1
    session.close.assert_not_awaited()


@pytest.mark.usefixtures("no_http_session")
def test_explicit_external_file_loads_without_mutating_environment(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    recognizer = microsoft_ai.STT(vad=None, env_file=path)
    synthesizer = microsoft_ai.TTS(env_file=path)
    assert recognizer.model == "file-transcriber"
    assert recognizer._language == "en"
    assert recognizer._client.headers["Authorization"] == "Bearer dummy-stt-key"
    assert synthesizer.model == "file-synthesizer"
    assert synthesizer.sample_rate == 24000
    assert synthesizer._opts.voice == "en-US-Dummy:file-synthesizer"
    assert synthesizer._client.headers["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in synthesizer._client.headers
    assert synthesizer._client.headers["Accept"] == "audio/wav"
    assert "MICROSOFT_AI_STT_API_KEY" not in os.environ
    assert "MICROSOFT_AI_TTS_API_KEY" not in os.environ


@pytest.mark.usefixtures("no_http_session")
def test_explicit_arguments_then_environment_then_file_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_ENV_FILE", str(path))
    monkeypatch.setenv("MICROSOFT_AI_STT_MODEL", "environment-transcriber")
    monkeypatch.setenv("MICROSOFT_AI_TTS_MODEL", "environment-synthesizer")
    recognizer = microsoft_ai.STT(vad=None)
    synthesizer = microsoft_ai.TTS(
        model="argument-synthesizer", voice="en-US-Dummy:argument-synthesizer", sample_rate=48000
    )
    assert recognizer.model == "environment-transcriber"
    assert synthesizer.model == "argument-synthesizer"
    assert synthesizer.sample_rate == 48000
    other = microsoft_ai.STT(vad=None, language="fr", model="argument-transcriber")
    assert other._language == "fr" and other.model == "argument-transcriber"


@pytest.mark.usefixtures("no_http_session")
def test_dotenv_values_are_literal_not_shell_sourced_or_interpolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace('"dummy-stt-key"', "'literal-${SHOULD_NOT_EXPAND}'"),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOULD_NOT_EXPAND", "not-a-credential")
    recognizer = microsoft_ai.STT(vad=None, env_file=path)
    assert recognizer._client.headers["Authorization"] == "Bearer literal-${SHOULD_NOT_EXPAND}"


@pytest.mark.usefixtures("no_http_session")
def test_empty_template_fails_without_network_or_secret_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "empty.env"
    path.write_text("MICROSOFT_AI_STT_API_KEY=\nMICROSOFT_AI_TTS_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MICROSOFT_AI_STT_MODEL"):
        microsoft_ai.STT(vad=None, env_file=path)
    with pytest.raises(ValueError, match="MICROSOFT_AI_TTS_SAMPLE_RATE"):
        microsoft_ai.TTS(env_file=path)
    assert not caplog.records


@pytest.mark.usefixtures("no_http_session")
def test_missing_selected_file_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read the selected") as caught:
        microsoft_ai.STT(vad=None, env_file=tmp_path / "private-location.env")
    assert "private-location" not in str(caught.value)


@pytest.mark.usefixtures("no_http_session")
async def test_smoke_preflights_both_services_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "incomplete.env"
    path.write_text(
        DUMMY_CONFIG.replace(
            "MICROSOFT_AI_TTS_API_KEY='dummy-tts-key'", "MICROSOFT_AI_TTS_API_KEY="
        ),
        encoding="utf-8",
    )
    create_session = MagicMock()
    monkeypatch.setattr(HTTPClient, "session", create_session)
    with pytest.raises(ValueError, match="MICROSOFT_AI_TTS_API_KEY"):
        await smoke._run(pcm=b"\0\0", expected="test", check_tts=True, env_file=path)
    create_session.assert_not_called()


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize(
    ("auth_header", "value"),
    [("Authorization", "Bearer dummy-stt-key"), ("api-key", "dummy-stt-key")],
)
def test_stt_auth_selector_from_file(
    tmp_path: Path, auth_header: Literal["Authorization", "api-key"], value: str
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG + f"MICROSOFT_AI_STT_AUTH_HEADER={auth_header}\n", encoding="utf-8"
    )
    instance = microsoft_ai.STT(vad=None, env_file=path)
    assert instance._client.headers == {auth_header: value, "User-Agent": "LiveKit Agents"}
    assert "dummy-stt-key" not in instance._client.url
    tts = microsoft_ai.TTS(env_file=path)
    assert tts._client.headers["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in tts._client.headers and "api-key" not in tts._client.headers


@pytest.mark.usefixtures("no_http_session")
def test_stt_auth_selector_uses_argument_environment_file_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG + "MICROSOFT_AI_STT_AUTH_HEADER=api-key\n", encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", "Authorization")
    assert microsoft_ai.STT(vad=None, env_file=path)._client.headers["Authorization"] == (
        "Bearer dummy-stt-key"
    )
    instance = microsoft_ai.STT(
        vad=None, env_file=path, auth_header="api-key", api_key="argument-key"
    )
    assert instance._client.headers["api-key"] == "argument-key"
    assert "Authorization" not in instance._client.headers


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize(
    "selector",
    ["", " ", "Api-Key", "Bearer", "Ocp-Apim-Subscription-Key", "api-key\r\nx-secret: dummy"],
)
def test_invalid_auth_selector_fails_without_echoing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", selector)
    with pytest.raises(ValueError, match="must be Authorization or api-key") as caught:
        microsoft_ai.STT(vad=None, env_file=path)
    assert "dummy" not in str(caught.value)
    assert not caplog.records


@pytest.mark.usefixtures("no_http_session")
def test_custom_headers_override_selector_environment_without_loading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace('MICROSOFT_AI_STT_API_KEY="dummy-stt-key"', ""), encoding="utf-8"
    )
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", "invalid-unused-value")
    instance = microsoft_ai.STT(vad=None, env_file=path, headers={"Authorization": "Bearer custom"})
    assert instance._client.headers == {
        "Authorization": "Bearer custom",
        "User-Agent": "LiveKit Agents",
    }
    assert microsoft_ai.STT(vad=None, env_file=path, headers={})._client.headers == {
        "User-Agent": "LiveKit Agents"
    }
    with pytest.raises(ValueError, match="either auth_header or headers"):
        microsoft_ai.STT(vad=None, env_file=path, headers={}, auth_header="api-key")


@pytest.mark.usefixtures("no_http_session")
@pytest.mark.parametrize(
    "credential", ["dummy\r\nx-header:value", "dummy\n", "dummy\x00", "dummy\x7f"]
)
def test_stt_credentials_cannot_inject_header_values(
    tmp_path: Path, credential: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    with pytest.raises(ValueError, match="control characters") as caught:
        microsoft_ai.STT(vad=None, env_file=path, api_key=credential, auth_header="api-key")
    assert "dummy" not in str(caught.value)
    assert not caplog.records


@pytest.mark.usefixtures("no_http_session")
async def test_stt_only_smoke_reads_auth_selector_without_tts_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        "\n".join(line for line in DUMMY_CONFIG.splitlines() if "MICROSOFT_AI_STT_" in line)
        + "\nMICROSOFT_AI_STT_AUTH_HEADER=api-key\n",
        encoding="utf-8",
    )
    checked = False

    async def check(provider: microsoft_ai.STT, pcm: bytes, expected: str) -> None:
        nonlocal checked
        assert provider._client.headers["api-key"] == "dummy-stt-key"
        assert "Authorization" not in provider._client.headers
        assert pcm == b"\0\0" and expected == "test"
        checked = True

    monkeypatch.setattr(smoke, "_check_stt", check)
    monkeypatch.setattr(
        microsoft_ai, "TTS", MagicMock(side_effect=AssertionError("TTS is not selected"))
    )
    await smoke._run(pcm=b"\0\0", expected="test", check_tts=False, env_file=path)
    assert checked


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--tts"],
        ["--run-live"],
        ["--run-live", "--stt-wav", "unused.wav"],
        ["--run-live", "--expected-text-file", "unused.txt"],
    ],
)
def test_smoke_requires_explicit_opt_in_and_complete_input_selection(
    args: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stt_factory, tts_factory = MagicMock(), MagicMock()
    monkeypatch.setattr(microsoft_ai, "STT", stt_factory)
    monkeypatch.setattr(microsoft_ai, "TTS", tts_factory)
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", *args])
    with pytest.raises(SystemExit) as caught:
        smoke.main()
    assert caught.value.code == 2
    stt_factory.assert_not_called()
    tts_factory.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_live_smoke_requires_owner_only_dotenv_permissions(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions 0600"):
        smoke._check_env_file_permissions(path)
    path.chmod(0o600)
    smoke._check_env_file_permissions(path)


@pytest.mark.parametrize("samples", [1, 1307, 80000])
def test_approved_fixture_is_bounded_without_padding(tmp_path: Path, samples: int) -> None:
    path = tmp_path / "synthetic.wav"
    pcm = b"\x01\x00" * samples
    path.write_bytes(wav_bytes(pcm, sample_rate=16000))
    assert smoke._read_fixture(path) == pcm


@pytest.mark.parametrize(
    "data",
    [
        wav_bytes(b"", sample_rate=16000),
        wav_bytes(b"\x00\x00" * 80001, sample_rate=16000),
        wav_bytes(b"\x00\x00" * 100, sample_rate=24000),
        wav_bytes(b"\x00\x00" * 100, sample_rate=16000, channels=2),
        wav_bytes(b"\x00\x00" * 100, sample_rate=16000)[:-2],
        b"\x00" * (1024 * 1024 + 1),
    ],
)
def test_invalid_or_over_limit_fixtures_fail_before_network(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "synthetic.wav"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        smoke._read_fixture(path)


@pytest.mark.parametrize("text", ["", "  ", "...", "a" * 257, "b" * 4097])
def test_expected_text_is_bounded_and_contains_words(tmp_path: Path, text: str) -> None:
    path = tmp_path / "synthetic.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        smoke._read_expected(path)


@pytest.mark.parametrize("expected", ["Turn 1.", "Turn 1 missing tail"])
async def test_stt_smoke_checks_complete_words_without_printing_them(
    expected: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    socket = FakeSocket()
    session = fake_session()
    session.ws_connect = AsyncMock(return_value=socket)
    instance = microsoft_ai.STT(
        vad=None,
        url="wss://stt.example.invalid/realtime",
        model="test",
        headers={},
        http_session=session,
    )
    pcm = b"\x01\x00" * 337
    async with instance:
        if "missing" in expected:
            with pytest.raises(ValueError, match="including its tail"):
                await smoke._check_stt(instance, pcm, expected)
        else:
            await smoke._check_stt(instance, pcm, expected)
    assert socket.commits == [pcm]
    assert expected not in capsys.readouterr().out
    assert socket.closed
    session.ws_connect.assert_awaited_once()


@pytest.mark.parametrize("text", ["Ready now.", "The final word.", "Repeat this."])
async def test_completed_turn_echoes_exact_text_once_without_an_llm(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    session = MagicMock(spec=AgentSession)
    handle = MagicMock(spec=SpeechHandle)
    session.say.return_value = handle
    monkeypatch.setattr(echo_example.EchoAgent, "session", property(lambda _: session))
    agent = echo_example.EchoAgent()
    message = llm.ChatMessage(role="user", content=[text])
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(llm.ChatContext(), message)
    session.say.assert_called_once_with(text, allow_interruptions=True, add_to_chat_ctx=False)
    handle.add_done_callback.assert_called_once_with(agent._speech_done)
    session.generate_reply.assert_not_called()


async def test_repeated_words_in_distinct_turns_are_not_silently_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=AgentSession)
    monkeypatch.setattr(echo_example.EchoAgent, "session", property(lambda _: session))
    agent = echo_example.EchoAgent()
    for _ in range(2):
        with pytest.raises(StopResponse):
            await agent.on_user_turn_completed(
                llm.ChatContext(), llm.ChatMessage(role="user", content=["Same words."])
            )
    assert session.say.call_count == 2


async def test_empty_completed_turn_does_not_synthesize(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock(spec=AgentSession)
    monkeypatch.setattr(echo_example.EchoAgent, "session", property(lambda _: session))
    with pytest.raises(StopResponse):
        await echo_example.EchoAgent().on_user_turn_completed(
            llm.ChatContext(), llm.ChatMessage(role="user", content=[])
        )
    session.say.assert_not_called()


def test_synthesis_error_is_reported_without_transcript_or_provider_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handle = MagicMock(spec=SpeechHandle)
    handle.exception.return_value = RuntimeError("dummy-private-transcript")
    echo_example.EchoAgent._speech_done(handle)
    assert "Echo synthesis failed (RuntimeError)" in caplog.text
    assert "dummy-private-transcript" not in caplog.text


def _mock_entrypoint(monkeypatch: pytest.MonkeyPatch):
    recognizer = MagicMock(spec=microsoft_ai.STT)
    recognizer.__aenter__.return_value = recognizer
    recognizer.__aexit__.return_value = False
    speech = MagicMock(spec=microsoft_ai.TTS)
    speech.sample_rate = 24000
    speech.__aenter__.return_value = speech
    speech.__aexit__.return_value = False
    stt_factory = MagicMock(return_value=recognizer)
    tts_factory = MagicMock(return_value=speech)
    monkeypatch.setattr(echo_example.microsoft_ai, "STT", stt_factory)
    monkeypatch.setattr(echo_example.microsoft_ai, "TTS", tts_factory)
    detector = MagicMock(spec=inference.VAD)
    vad_factory = MagicMock(return_value=detector)
    monkeypatch.setattr(echo_example.inference, "VAD", vad_factory)
    session = MagicMock(spec=AgentSession)
    session.room_io = SimpleNamespace(wait_for_ready=AsyncMock())
    callbacks = {}

    def on(event):
        def register(callback):
            callbacks[event] = callback
            return callback

        return register

    session.on.side_effect = on
    started = asyncio.Event()

    async def start(**kwargs):
        started.set()

    session.start = AsyncMock(side_effect=start)
    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr(echo_example, "AgentSession", session_factory)
    ctx = MagicMock(spec=JobContext)
    ctx.room = MagicMock(spec=rtc.Room)
    return SimpleNamespace(
        ctx=ctx,
        recognizer=recognizer,
        speech=speech,
        session=session,
        callbacks=callbacks,
        started=started,
        session_factory=session_factory,
        stt_factory=stt_factory,
        tts_factory=tts_factory,
        detector=detector,
        vad_factory=vad_factory,
    )


async def test_shares_vad_with_native_stt_and_closes_on_participant_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_entrypoint(monkeypatch)
    monkeypatch.setenv("MICROSOFT_AI_ENV_FILE", "selected-private-config.env")
    task = asyncio.create_task(echo_example.entrypoint(fake.ctx))
    try:
        await asyncio.wait_for(fake.started.wait(), 1)
        fake.callbacks["close"](SimpleNamespace(reason="participant_disconnected"))
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    fake.vad_factory.assert_called_once_with(min_silence_duration=0.5)
    fake.stt_factory.assert_called_once_with(
        vad=fake.detector, env_file="selected-private-config.env"
    )
    fake.tts_factory.assert_called_once_with(env_file="selected-private-config.env")
    options = fake.session_factory.call_args.kwargs
    assert options["stt"] is fake.recognizer and options["tts"] is fake.speech
    assert options["vad"] is fake.detector
    assert "llm" not in options
    turn = options["turn_handling"]
    assert turn["turn_detection"] == "stt"
    assert turn["preemptive_generation"] == {"enabled": False}
    assert turn["interruption"]["enabled"]
    assert turn["interruption"]["mode"] == "vad"
    assert turn["interruption"]["resume_false_interruption"] is False
    assert options["conn_options"].stt_conn_options.max_retry == 0
    assert options["conn_options"].tts_conn_options.max_retry == 0
    room = fake.session.start.call_args.kwargs
    assert room["session_host"] is False and room["record"] is False
    assert room["room_options"].audio_input.sample_rate == 16000
    assert room["room_options"].audio_input.pre_connect_audio is False
    assert room["room_options"].video_input is False
    assert room["room_options"].text_input is False
    assert room["room_options"].text_output is True
    fake.session.say.assert_not_called()
    fake.session.generate_reply.assert_not_called()
    fake.session.room_io.wait_for_ready.assert_awaited_once()
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()
    fake.ctx.shutdown.assert_called_once_with(reason="Echo session ended")


async def test_cancelled_echo_session_releases_both_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_entrypoint(monkeypatch)
    task = asyncio.create_task(echo_example.entrypoint(fake.ctx))
    await asyncio.wait_for(fake.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()


async def test_session_time_limit_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _mock_entrypoint(monkeypatch)
    monkeypatch.setattr(echo_example, "SESSION_LIMIT", 0.01)
    await asyncio.wait_for(echo_example.entrypoint(fake.ctx), 1)
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()
    fake.ctx.shutdown.assert_called_once_with(reason="Echo session ended")


async def test_room_readiness_failure_closes_without_waiting_for_a_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_entrypoint(monkeypatch)
    fake.session.room_io.wait_for_ready.side_effect = asyncio.TimeoutError()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(echo_example.entrypoint(fake.ctx), 1)
    fake.session.say.assert_not_called()
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()
    fake.ctx.shutdown.assert_not_called()


async def test_real_model_less_session_speaks_final_text_and_can_interrupt() -> None:
    http = fake_session()
    http.post.side_effect = [
        FakeResponse(wav_bytes(b"\x81\x01" * 48000)),
        FakeResponse(wav_bytes(b"\x82\x02" * 1200)),
    ]
    speech = microsoft_ai.TTS(
        url="https://tts.example.invalid/cognitiveservices/v1",
        model="dummy",
        voice="en-US-Dummy:dummy",
        sample_rate=24000,
        headers={},
        http_session=http,
    )
    output = FakeAudioOutput(sample_rate=24000)
    began = asyncio.Event()
    output.on("playback_started", lambda _: began.set())
    async with speech:
        session = AgentSession(
            tts=speech,
            vad=None,
            turn_handling={"turn_detection": None},
            conn_options=SessionConnectOptions(tts_conn_options=APIConnectOptions(max_retry=0)),
        )
        session.output.audio = output
        agent = echo_example.EchoAgent()
        handles = []
        session.on("speech_created", lambda event: handles.append(event.speech_handle))
        try:
            await session.start(agent, session_host=False, record=False)
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(
                    llm.ChatContext(), llm.ChatMessage(role="user", content=["First echo."])
                )
            await asyncio.wait_for(began.wait(), 2)
            await session.interrupt()
            assert handles[0].interrupted
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(
                    llm.ChatContext(), llm.ChatMessage(role="user", content=["Next echo."])
                )
            await asyncio.wait_for(handles[1], 2)
            assert handles[1].exception() is None and not handles[1].interrupted
            assert session.llm is None
        finally:
            await session.aclose()
    assert http.post.call_count == 2
    assert [
        next(iter(ElementTree.fromstring(call.kwargs["data"]))).text
        for call in http.post.call_args_list
    ] == ["First echo.", "Next echo."]
