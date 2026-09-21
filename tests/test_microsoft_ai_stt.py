from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    inference,
    stt,
    vad,
)
from livekit.plugins import microsoft_ai

from .microsoft_ai_fakes import FakeSocket, ScriptedVAD, audio_frame, fake_session

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

OPTIONS = APIConnectOptions(max_retry=0, timeout=0.5)
STT_URL = "wss://stt.example.invalid/v1/realtime?intent=transcription&deployment=dummy"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hermetic Microsoft AI tests must not make network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.delenv("MICROSOFT_AI_ENV_FILE", raising=False)
    for name in ("URL", "MODEL", "API_KEY", "LANGUAGE"):
        monkeypatch.delenv(f"MICROSOFT_AI_STT_{name}", raising=False)


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
