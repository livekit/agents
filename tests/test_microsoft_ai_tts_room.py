from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from examples.voice_agents import microsoft_ai_tts_room as example
from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectOptions, APIStatusError, JobContext
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai

from .fake_io import FakeAudioOutput
from .microsoft_ai_fakes import FakeResponse, fake_session, wav_bytes

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("TTS room example tests must not contact providers or a room server")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.delenv("MICROSOFT_AI_ENV_FILE", raising=False)


class _Handle:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.awaited = False

    def __await__(self):
        async def finish() -> None:
            self.awaited = True

        return finish().__await__()

    def exception(self) -> Exception | None:
        assert self.awaited
        return self.error


def _mock_example(monkeypatch: pytest.MonkeyPatch):
    speech = MagicMock(spec=microsoft_ai.TTS)
    speech.sample_rate = 24000
    speech.__aenter__.return_value = speech
    speech.__aexit__.return_value = False
    provider_factory = MagicMock(return_value=speech)
    monkeypatch.setattr(example.microsoft_ai, "TTS", provider_factory)
    monkeypatch.setattr(
        example.microsoft_ai,
        "STT",
        MagicMock(side_effect=AssertionError("TTS-only example must not construct STT")),
    )
    started = asyncio.Event()
    ready = asyncio.Event()
    handle = _Handle()
    session = MagicMock(spec=AgentSession)
    session.room_io = SimpleNamespace(wait_for_ready=AsyncMock(side_effect=ready.wait))
    session.say.return_value = handle

    async def start(**kwargs: object) -> None:
        started.set()

    session.start = AsyncMock(side_effect=start)
    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr(example, "AgentSession", session_factory)
    context = MagicMock(spec=JobContext)
    context.room = MagicMock(spec=rtc.Room)
    return context, session, speech, provider_factory, session_factory, started, ready, handle


async def test_waits_for_subscription_and_says_once_without_input_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, factory, session_factory, started, ready, handle = _mock_example(
        monkeypatch
    )
    monkeypatch.setenv("MICROSOFT_AI_ENV_FILE", "selected-private-config.env")
    task = asyncio.create_task(example.entrypoint(context))
    try:
        await asyncio.wait_for(started.wait(), 1)
        session.say.assert_not_called()
        ready.set()
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    factory.assert_called_once_with(env_file="selected-private-config.env")
    kwargs = session_factory.call_args.kwargs
    assert kwargs["tts"] is speech
    assert kwargs["vad"] is None
    assert kwargs["turn_handling"] == {"turn_detection": None}
    assert kwargs["user_away_timeout"] is None
    assert "stt" not in kwargs and "llm" not in kwargs
    assert kwargs["conn_options"].tts_conn_options.max_retry == 0
    start = session.start.call_args.kwargs
    assert start["room"] is context.room
    assert start["record"] is False and start["session_host"] is False
    options = start["room_options"]
    assert options.audio_input is False and options.video_input is False
    assert options.text_input is False and options.text_output is False
    assert options.audio_output.sample_rate == 24000
    session.say.assert_called_once_with(
        "Hello, this is a Microsoft AI voice test.",
        allow_interruptions=False,
        add_to_chat_ctx=False,
    )
    assert handle.awaited
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_cancelling_before_room_ready_closes_without_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, started, _, _ = _mock_example(monkeypatch)
    task = asyncio.create_task(example.entrypoint(context))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    session.say.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_readiness_timeout_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, _, _, _ = _mock_example(monkeypatch)
    session.room_io.wait_for_ready.side_effect = asyncio.TimeoutError()
    with pytest.raises(asyncio.TimeoutError):
        await example.entrypoint(context)
    session.say.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_speech_handle_error_is_checked_and_resources_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, speech, _, _, _, ready, handle = _mock_example(monkeypatch)
    error = APIStatusError("Microsoft AI TTS request failed", status_code=401)
    handle.error = error
    ready.set()
    with pytest.raises(APIStatusError) as caught:
        await example.entrypoint(context)
    assert caught.value is error
    assert handle.awaited
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_start_failure_closes_session_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    context, session, speech, _, _, _, _, _ = _mock_example(monkeypatch)
    session.start.side_effect = RuntimeError("room setup failed")
    with pytest.raises(RuntimeError, match="room setup failed"):
        await example.entrypoint(context)
    session.say.assert_not_called()
    session.aclose.assert_awaited_once()
    speech.__aexit__.assert_awaited_once()


async def test_actual_agent_session_say_uses_only_tts_and_emits_audio() -> None:
    pcm = b"\x81\x01" * 1200
    response = FakeResponse(wav_bytes(pcm))
    http = fake_session()
    http.post.return_value = response
    speech = microsoft_ai.TTS(
        url="https://tts.example.invalid/cognitiveservices/v1",
        model="test-model",
        voice="en-US-Dummy:test-model",
        sample_rate=24000,
        headers={},
        http_session=http,
    )
    output = FakeAudioOutput(sample_rate=24000)
    capture = AsyncMock(wraps=output.capture_frame)
    output.capture_frame = capture
    async with speech:
        session = AgentSession(
            tts=speech,
            vad=None,
            turn_handling={"turn_detection": None},
            user_away_timeout=None,
            conn_options=SessionConnectOptions(
                tts_conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
            ),
        )
        session.output.audio = output
        try:
            await session.start(
                Agent(instructions="Say the supplied text."), session_host=False, record=False
            )
            handle = session.say(example.GREETING, allow_interruptions=False, add_to_chat_ctx=False)
            await asyncio.wait_for(handle, 3)
            assert handle.exception() is None
            assert session.stt is None and session.llm is None and session.vad is None
        finally:
            await session.aclose()
    http.post.assert_called_once()
    root = ElementTree.fromstring(http.post.call_args.kwargs["data"])
    assert next(iter(root)).text == example.GREETING
    frames = [call.args[0] for call in capture.await_args_list]
    assert frames
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
    assert b"".join(frame.data.tobytes() for frame in frames).startswith(pcm)
    assert response.closed
