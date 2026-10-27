from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
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

from .microsoft_ai_fakes import (
    DUMMY_CONFIG,
    FakeSocket,
    ScriptedVAD,
    audio_frame,
    fake_session,
    no_http_session as no_http_session,
    no_network as no_network,
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


async def test_bundled_vad_silence_does_not_create_an_empty_turn() -> None:
    socket = FakeSocket()
    instance, _ = provider(socket, detector=inference.VAD(model="silero"))
    frame = audio_frame(1307, value=b"\0\0")
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        assert await collect(stream) == []
        assert stream._processed_samples == frame.samples_per_channel
    assert socket.commits == []
    assert all(event["type"] == "session.update" for event in socket.sent)


async def test_bundled_vad_start_frames_preserve_delayed_onset_and_tail() -> None:
    # A public zero activation threshold makes synthetic PCM deterministically
    # count as speech. This checks the VAD frame contract, not model accuracy.
    detector = inference.VAD(
        activation_threshold=0.0, min_speech_duration=0.736, prefix_padding_duration=0.0
    )
    socket = FakeSocket()
    instance, _ = provider(socket, detector=detector)
    frame = audio_frame(16437)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2)) as stream,
    ):
        stream.push_frame(frame)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
        assert stream._processed_samples == frame.samples_per_channel
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


async def test_prolonged_vad_idle_sends_no_audio_and_keeps_local_retention_bounded() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD(prefix_samples=16000)
    instance, _ = provider(socket, detector=detector, max_buffered_audio=1.1)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        for second in range(180):
            stream.push_frame(audio_frame(16000, value=b"\0\0"))
            await detector.wait_consumed(((second + 1) * 16000 // 512) * 512)
            assert stream._input_duration - stream._processed_samples / 16000 <= 0.032
            assert detector.streams[0].retained_samples <= 16000
        assert socket.pending_audio == b""
        assert socket.commits == []
        assert all(event["type"] == "session.update" for event in socket.sent)
        assert stream._segment_samples == 0
        stream.end_input()
        assert await collect(stream) == []
    assert socket.closed


async def test_prolonged_interturn_silence_does_not_grow_the_remote_buffer() -> None:
    idle_samples = 16000 * 180
    detector = ScriptedVAD(
        {
            512: vad.VADEventType.START_OF_SPEECH,
            2048: vad.VADEventType.END_OF_SPEECH,
            2048 + idle_samples + 512: vad.VADEventType.START_OF_SPEECH,
            4096 + idle_samples: vad.VADEventType.END_OF_SPEECH,
        },
        prefix_samples=1024,
    )
    socket = FakeSocket()
    instance, _ = provider(socket, detector=detector, max_buffered_audio=1.1)
    first, second = audio_frame(2048), audio_frame(2048, value=b"\x01\0")
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(first)
        first_events = []
        while True:
            event = await next_event(stream)
            first_events.append(event)
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                break
        assert finals(first_events) == ["turn 1"]
        for second_index in range(180):
            stream.push_frame(audio_frame(16000, value=b"\0\0"))
            await detector.wait_consumed((2048 + (second_index + 1) * 16000) // 512 * 512)
            assert stream._input_duration - stream._processed_samples / 16000 <= 0.032
            assert detector.streams[0].retained_samples <= 1024
        assert socket.commits == [first.data.tobytes()]
        assert socket.pending_audio == b""
        assert stream._segment_samples == 0
        stream.push_frame(second)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 2"]
        assert stream._processed_samples == 4096 + idle_samples
    assert socket.commits == [first.data.tobytes(), b"\0\0" * 512 + second.data.tobytes()]
    assert socket.closed


async def test_delayed_vad_start_uses_all_public_prefix_frames_not_a_fixed_preroll() -> None:
    detector = ScriptedVAD(
        {16384: vad.VADEventType.START_OF_SPEECH, 32768: vad.VADEventType.END_OF_SPEECH},
        prefix_samples=12000,
    )
    socket = FakeSocket()
    instance, _ = provider(socket, detector=detector)
    # Speech starts 724 ms before START_OF_SPEECH. A guessed 500 ms
    # pre-roll would lose 224 ms, but the VAD's actual prefix contains it.
    pcm = b"\0\0" * 4800 + b"\x23\x01" * 19200 + b"\x34\x02" * 64
    pcm += b"\0\0" * (32768 + 123 - 24064)
    frame = rtc.AudioFrame(
        data=pcm, sample_rate=16000, num_channels=1, samples_per_channel=len(pcm) // 2
    )
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        assert finals(await collect(stream)) == ["turn 1"]
        assert stream._processed_samples == len(pcm) // 2
    assert socket.commits == [pcm[(16384 - 12000) * 2 : 32768 * 2]]
    assert b"\x34\x02" * 64 in socket.commits[0]


async def test_recovered_prefix_is_not_double_counted_against_backpressure() -> None:
    before = asyncio.all_tasks()
    socket = FakeSocket()
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH}, prefix_samples=512)
    instance, _ = provider(socket, detector=detector, max_buffered_audio=0.1)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(512))
        await socket.wait_sent("input_audio_buffer.append")
        assert stream._processed_samples == 512
        socket.append_started.clear()
        socket.append_gate = asyncio.Event()
        stream.push_frame(audio_frame(1600))
        await asyncio.wait_for(socket.append_started.wait(), 1)
        with pytest.raises(APIConnectionError, match="buffer is full"):
            stream.push_frame(audio_frame(1))
        with pytest.raises(APIConnectionError, match="buffer is full"):
            await asyncio.wait_for(collect(stream), 1)
        assert socket.closed
    assert socket.commits == []
    assert not (asyncio.all_tasks() - before)


async def test_vad_start_without_prefix_fails_instead_of_dropping_onset() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH}, prefix_samples=0)
    instance, _ = provider(socket, detector=detector)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(512))
        with pytest.raises(APIError, match="preserve the speech prefix") as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert not socket.commits and not socket.pending_audio


async def test_oversized_vad_prefix_fails_explicitly_not_by_truncating_audio() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD({2048: vad.VADEventType.START_OF_SPEECH})
    instance, _ = provider(socket, detector=detector, max_buffered_audio=0.1)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        for position in (512, 1024, 1536):
            stream.push_frame(audio_frame(512))
            await detector.wait_consumed(position)
        stream.push_frame(audio_frame(512))
        with pytest.raises(APIError, match="prefix exceeds max_buffered_audio") as caught:
            await collect(stream)
        assert not caught.value.retryable
        assert stream._input_error is None
    assert not socket.commits and not socket.pending_audio


@pytest.mark.parametrize("ending", ["flush", "end_input", "aclose"])
async def test_idle_vad_boundaries_do_not_send_audio_or_empty_commits(ending: str) -> None:
    socket = FakeSocket()
    detector = ScriptedVAD()
    instance, _ = provider(socket, detector=detector)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(1307, value=b"\0\0"))
        await detector.wait_consumed(1024)
        if ending == "flush":
            stream.flush()
            stream.end_input()
            assert await collect(stream) == []
        elif ending == "end_input":
            stream.end_input()
            assert await collect(stream) == []
        else:
            await stream.aclose()
            assert await collect(stream) == []
    assert not socket.commits and not socket.pending_audio
    assert all(event["type"] == "session.update" for event in socket.sent)
    assert socket.closed and all(item.closed for item in detector.streams)


@pytest.mark.parametrize("use_vad", [False, True])
async def test_resampling_drains_the_sdk_resampler_tail(
    monkeypatch: pytest.MonkeyPatch, use_vad: bool
) -> None:
    socket = FakeSocket()
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH}) if use_vad else None
    instance, _ = provider(socket, detector=detector)
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
        assert stream._processed_samples == sum(frame.samples_per_channel for frame in resampled)
    assert socket.commits == [b"".join(f.data.tobytes() for f in resampled)]
    assert sum(f.samples_per_channel for f in resampled) == round(4817 / 3)
    assert all(f.sample_rate == 16000 and f.num_channels == 1 for f in resampled)


@pytest.mark.parametrize(
    ("delta", "hypotheses", "transcript"),
    [
        pytest.param("I agree", [" completely"], "I agree", id="retracted-interim"),
        pytest.param(
            "I agree", [" completly", " completely"], "I agree completely.", id="corrected-interim"
        ),
        pytest.param(
            "I agree", [" completely", " in part"], "I agree in part.", id="replaced-interim"
        ),
        pytest.param("", ["spurious speech"], "", id="empty-final"),
        pytest.param("I agree", [" completely"], "", id="empty-final-after-delta"),
    ],
)
async def test_acknowledged_final_is_authoritative_and_next_turn_continues(
    delta: str, hypotheses: list[str], transcript: str
) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, http = provider(socket)
    first, second = audio_frame(901), audio_frame(83, value=b"\x01\0")
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(first)
        stream.flush()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.transcript("delta", delta=delta)
        for hypothesis in hypotheses:
            socket.transcript("intermediate", intermediate=hypothesis)
        socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        socket.transcript("completed", transcript=transcript)
        socket.transcript("completed", transcript=transcript)
        events = []
        while True:
            event = await next_event(stream)
            events.append(event)
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                break
        assert finals(events) == ([transcript] if transcript else [])
        interims = [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
        ]
        assert interims[-1] == delta + hypotheses[-1]
        assert sum(event.type == stt.SpeechEventType.RECOGNITION_USAGE for event in events) == 1
        assert not socket.closed and not stream._input_ch.closed

        stream.push_frame(second)
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit", count=2)
        socket.complete(item="item-2", text="Next turn.")
        following = await collect(stream)
        assert finals(following) == ["Next turn."]
        assert sum(event.type == stt.SpeechEventType.RECOGNITION_USAGE for event in following) == 1
    assert socket.commits == [first.data.tobytes(), second.data.tobytes()]
    http.ws_connect.assert_awaited_once()
    assert socket.closed


async def test_empty_completion_without_interim_does_not_invent_user_text() -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        socket.transcript("completed", transcript="")
        events = await collect(stream)
    assert [event.type for event in events] == [stt.SpeechEventType.RECOGNITION_USAGE]
    assert all(not event.alternatives for event in events)
    assert socket.closed


@pytest.mark.parametrize(
    ("ack_item", "completed_item"),
    [(None, "item-1"), ("item-2", "item-1"), ("item-1", "item-2")],
    ids=["missing-ack", "wrong-ack-item", "wrong-completed-item"],
)
async def test_authoritative_final_still_requires_the_matching_commit_ack(
    ack_item: str | None, completed_item: str
) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.transcript("delta", delta="I agree")
        socket.transcript("intermediate", intermediate=" completely")
        if ack_item is not None:
            socket.emit({"type": "input_audio_buffer.committed", "item_id": ack_item})
        socket.transcript("completed", item=completed_item, transcript="I agree")
        with pytest.raises(APIError) as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert socket.closed


@pytest.mark.parametrize(
    "payload",
    [{}, {"transcript": None}, {"transcript": 123}, {"transcript": []}],
    ids=["missing", "null", "number", "list"],
)
async def test_acknowledged_completed_event_still_requires_string_transcript(
    payload: dict[str, object],
) -> None:
    socket = FakeSocket(auto_commit=False)
    instance, _ = provider(socket)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame())
        stream.end_input()
        await socket.wait_sent("input_audio_buffer.commit")
        socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        socket.transcript("completed", **payload)
        with pytest.raises(APIError, match="string transcript") as caught:
            await collect(stream)
        assert not caught.value.retryable
    assert socket.closed


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


@pytest.mark.parametrize("ack", [False, True])
async def test_live_vad_commit_timeout_propagates_with_input_still_open(ack: bool) -> None:
    before = asyncio.all_tasks()
    socket = FakeSocket(auto_commit=False)
    detector = ScriptedVAD(
        {512: vad.VADEventType.START_OF_SPEECH, 1024: vad.VADEventType.END_OF_SPEECH}
    )
    instance, http = provider(socket, detector=detector)
    options = APIConnectOptions(timeout=0.1, max_retry=2)
    async with instance, instance.stream(conn_options=options) as stream:
        stream.push_frame(audio_frame(1024))
        await socket.wait_sent("input_audio_buffer.commit")
        if ack:
            socket.emit({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        with pytest.raises(APITimeoutError) as caught:
            await asyncio.wait_for(collect(stream), timeout=1.0)
        assert not stream._input_ch.closed
        assert stream._input_error is None
        assert not caught.value.retryable
        assert socket.closed
        assert detector.streams[0]._task.done()
        assert detector.streams[0]._metrics_task.done()
    http.ws_connect.assert_awaited_once()
    assert len(socket.commits) == 1
    assert not (asyncio.all_tasks() - before)


async def test_vad_feed_error_propagates_while_consumer_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = asyncio.all_tasks()
    socket = FakeSocket()
    detector = ScriptedVAD()
    instance, http = provider(socket, detector=detector)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(audio_frame(1024))
        await detector.wait_consumed(1024)
        failure = RuntimeError("VAD input failed")

        def fail_push(frame: rtc.AudioFrame) -> None:
            raise failure

        monkeypatch.setattr(detector.streams[0], "push_frame", fail_push)
        stream.push_frame(audio_frame(512))
        with pytest.raises(RuntimeError, match="VAD input failed") as caught:
            await asyncio.wait_for(collect(stream), timeout=1.0)
        assert caught.value is failure
        assert not stream._input_ch.closed
        assert socket.closed
    http.ws_connect.assert_awaited_once()
    assert detector.streams[0].closed
    assert not (asyncio.all_tasks() - before)


async def test_vad_finite_input_drains_exact_tail_and_waits_for_final() -> None:
    socket = FakeSocket(auto_commit=False)
    detector = ScriptedVAD({512: vad.VADEventType.START_OF_SPEECH})
    instance, _ = provider(socket, detector=detector)
    frame = audio_frame(1307)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(frame)
        stream.end_input()
        reader = asyncio.create_task(collect(stream))
        try:
            await socket.wait_sent("input_audio_buffer.commit")
            assert socket.commits == [frame.data.tobytes()]
            assert not reader.done() and not socket.closed
            socket.complete(text="all final words")
            assert finals(await asyncio.wait_for(reader, timeout=1.0)) == ["all final words"]
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
    assert socket.closed and detector.streams[0].closed


async def test_vad_finite_input_has_a_bounded_consumer_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = asyncio.all_tasks()
    socket = FakeSocket()
    detector = ScriptedVAD()
    instance, _ = provider(socket, detector=detector)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1)) as stream,
    ):
        stream.push_frame(audio_frame(1024))
        await detector.wait_consumed(1024)
        # Simulate a VAD that never completes after receiving its end-of-input signal.
        monkeypatch.setattr(detector.streams[0], "end_input", lambda: None)
        stream.end_input()
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(collect(stream), timeout=1.0)
        assert stream._input_ch.closed
        assert socket.closed
    assert detector.streams[0].closed
    assert not (asyncio.all_tasks() - before)


async def test_vad_cancellation_releases_a_pending_commit_and_open_input() -> None:
    before = asyncio.all_tasks()
    socket = FakeSocket(auto_commit=False)
    detector = ScriptedVAD(
        {512: vad.VADEventType.START_OF_SPEECH, 1024: vad.VADEventType.END_OF_SPEECH}
    )
    instance, _ = provider(socket, detector=detector)
    async with instance:
        stream = instance.stream(conn_options=APIConnectOptions(timeout=10.0))
        stream.push_frame(audio_frame(1024))
        await socket.wait_sent("input_audio_buffer.commit")
        assert not stream._input_ch.closed
        await asyncio.wait_for(stream.aclose(), timeout=1.0)
        assert await collect(stream) == []
        assert socket.closed
        assert stream._task.done() and stream._metrics_task.done()
        assert detector.streams[0].closed
    assert len(socket.commits) == 1
    assert not (asyncio.all_tasks() - before)


async def test_live_vad_input_does_not_get_a_stream_idle_timeout() -> None:
    socket = FakeSocket()
    detector = ScriptedVAD()
    instance, _ = provider(socket, detector=detector)
    async with (
        instance,
        instance.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05)) as stream,
    ):
        stream.push_frame(audio_frame(1024))
        await detector.wait_consumed(1024)
        reader = asyncio.create_task(collect(stream))
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(reader), timeout=0.15)
            assert not reader.done()
            assert not stream._input_ch.closed and not socket.closed
            assert not socket.commits
        finally:
            await stream.aclose()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
    assert detector.streams[0].closed


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
