from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from examples.voice_agents import microsoft_ai_echo as example
from livekit import rtc
from livekit.agents import (
    AgentSession,
    APIConnectOptions,
    JobContext,
    StopResponse,
    inference,
    llm,
)
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import microsoft_ai

from .fake_io import FakeAudioOutput
from .microsoft_ai_fakes import FakeResponse, fake_session, wav_bytes

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Echo example unit tests must not make network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.delenv("MICROSOFT_AI_ENV_FILE", raising=False)


@pytest.mark.parametrize("text", ["Ready now.", "The final word.", "Repeat this."])
async def test_completed_turn_echoes_exact_text_once_without_an_llm(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    session = MagicMock(spec=AgentSession)
    handle = MagicMock(spec=SpeechHandle)
    session.say.return_value = handle
    monkeypatch.setattr(example.EchoAgent, "session", property(lambda _: session))
    agent = example.EchoAgent()
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
    monkeypatch.setattr(example.EchoAgent, "session", property(lambda _: session))
    agent = example.EchoAgent()
    for _ in range(2):
        with pytest.raises(StopResponse):
            await agent.on_user_turn_completed(
                llm.ChatContext(), llm.ChatMessage(role="user", content=["Same words."])
            )
    assert session.say.call_count == 2


async def test_empty_completed_turn_does_not_synthesize(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock(spec=AgentSession)
    monkeypatch.setattr(example.EchoAgent, "session", property(lambda _: session))
    with pytest.raises(StopResponse):
        await example.EchoAgent().on_user_turn_completed(
            llm.ChatContext(), llm.ChatMessage(role="user", content=[])
        )
    session.say.assert_not_called()


def test_synthesis_error_is_reported_without_transcript_or_provider_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handle = MagicMock(spec=SpeechHandle)
    handle.exception.return_value = RuntimeError("dummy-private-transcript")
    example.EchoAgent._speech_done(handle)
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
    monkeypatch.setattr(example.microsoft_ai, "STT", stt_factory)
    monkeypatch.setattr(example.microsoft_ai, "TTS", tts_factory)
    detector = MagicMock(spec=inference.VAD)
    vad_factory = MagicMock(return_value=detector)
    monkeypatch.setattr(example.inference, "VAD", vad_factory)
    session = MagicMock(spec=AgentSession)
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
    monkeypatch.setattr(example, "AgentSession", session_factory)
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
    task = asyncio.create_task(example.entrypoint(fake.ctx))
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
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()
    fake.ctx.shutdown.assert_called_once_with(reason="Echo session ended")


async def test_cancelled_echo_session_releases_both_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_entrypoint(monkeypatch)
    task = asyncio.create_task(example.entrypoint(fake.ctx))
    await asyncio.wait_for(fake.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()


async def test_session_time_limit_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _mock_entrypoint(monkeypatch)
    monkeypatch.setattr(example, "SESSION_LIMIT", 0.01)
    await asyncio.wait_for(example.entrypoint(fake.ctx), 1)
    fake.session.aclose.assert_awaited_once()
    fake.recognizer.__aexit__.assert_awaited_once()
    fake.speech.__aexit__.assert_awaited_once()
    fake.ctx.shutdown.assert_called_once_with(reason="Echo session ended")


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
        agent = example.EchoAgent()
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
