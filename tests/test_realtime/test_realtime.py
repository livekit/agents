from __future__ import annotations

import asyncio
import os
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from openai.types import realtime

from livekit import rtc
from livekit.agents import RunContext, function_tool, llm, utils
from livekit.plugins import openai, xai

TESTS_DIR = Path(__file__).parent
SAMPLE_RATE = 24000


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv(override=True)


def _openai_model() -> openai.realtime.RealtimeModel:
    return openai.realtime.RealtimeModel(
        voice="alloy",
        input_audio_transcription=realtime.AudioTranscription(model="gpt-4o-mini-transcribe"),
    )


def _azure_model() -> openai.realtime.RealtimeModel:
    return openai.realtime.RealtimeModel(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"] + "/openai",
        voice="alloy",
    )


def _xai_model() -> xai.realtime.RealtimeModel:
    return xai.realtime.RealtimeModel()


_skip_xai = pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="XAI_API_KEY not set")

REALTIME_MODELS: list = [
    pytest.param(_openai_model, id="openai"),
    pytest.param(_azure_model, id="azure"),
    pytest.param(_xai_model, id="xai", marks=_skip_xai),
]

# xAI doesn't support conversation.item.delete or sequential response.create
OPENAI_AND_AZURE: list = [
    pytest.param(_openai_model, id="openai"),
    pytest.param(_azure_model, id="azure"),
]


@pytest.fixture
async def rt_session(request, job_process):
    factory: Callable = request.param
    model = factory()
    session = model.session()
    yield session
    await session.aclose()
    # small delay between tests to avoid rate limiting (especially xAI)
    await asyncio.sleep(0.5)


# -- Helpers --


async def _collect_text(gen_ev: llm.GenerationCreatedEvent) -> str:
    parts: list[str] = []
    async for msg_gen in gen_ev.message_stream:
        async for chunk in msg_gen.text_stream:
            parts.append(chunk)
    return "".join(parts)


def _silence(duration_s: float = 2.0) -> rtc.AudioFrame:
    n = int(SAMPLE_RATE * duration_s)
    return rtc.AudioFrame(
        data=bytes(n * 2), sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=n
    )


def _load_wav(name: str) -> list[rtc.AudioFrame]:
    with wave.open(str(TESTS_DIR / f"{name}.wav"), "rb") as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    chunk = sr // 50  # 20ms
    frames, off = [], 0
    while off + chunk <= len(raw) // (ch * 2):
        frames.append(
            rtc.AudioFrame(
                data=raw[off * 2 * ch : (off + chunk) * 2 * ch],
                sample_rate=sr,
                num_channels=ch,
                samples_per_channel=chunk,
            )
        )
        off += chunk
    return frames


async def _push_speech(session: llm.RealtimeSession, wav_name: str = "hello_world") -> None:
    for frame in _load_wav(wav_name):
        session.push_audio(frame)
        await asyncio.sleep(0.02)
    session.push_audio(_silence())


# -- Basic generation --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_generate_reply(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    assert gen_ev.user_initiated
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert len(text) > 0


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_generate_reply_with_instructions(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="Say exactly: pineapple"),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "pineapple" in text.lower()


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_generate_reply_preserves_session_instructions(rt_session: llm.RealtimeSession):
    await rt_session.update_instructions("Your name is Kelly. Always respond in English.")
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="Tell the user what your name is."),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "kelly" in text.lower(), f"Expected 'kelly' in response, got: {text}"


@pytest.mark.parametrize("rt_session", OPENAI_AND_AZURE, indirect=True)
async def test_multiple_sequential_replies(rt_session: llm.RealtimeSession):
    for i in range(2):
        gen_ev = await asyncio.wait_for(
            rt_session.generate_reply(instructions=f"Say the number {i}"),
            timeout=20,
        )
        text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
        assert len(text) > 0, f"Reply {i} was empty"


# -- Audio I/O --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_response_includes_audio(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)

    got_audio = False
    async for msg_gen in gen_ev.message_stream:
        modalities = await asyncio.wait_for(msg_gen.modalities, timeout=10)
        assert "audio" in modalities
        async for frame in msg_gen.audio_stream:
            assert frame.sample_rate == SAMPLE_RATE
            assert frame.num_channels == 1
            assert len(frame.data) > 0
            got_audio = True
            break
    assert got_audio


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_push_audio(rt_session: llm.RealtimeSession):
    t = np.linspace(0, 0.5, SAMPLE_RATE // 2, dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    frame = rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(samples),
    )
    rt_session.push_audio(frame)
    rt_session.commit_audio()


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_push_real_audio_triggers_response(rt_session: llm.RealtimeSession):
    generation_received = asyncio.Event()
    gen_events: list[llm.GenerationCreatedEvent] = []

    def on_gen(ev: llm.GenerationCreatedEvent):
        gen_events.append(ev)
        generation_received.set()

    rt_session.on("generation_created", on_gen)
    await _push_speech(rt_session)
    await asyncio.wait_for(generation_received.wait(), timeout=15)

    assert not gen_events[0].user_initiated
    text = await asyncio.wait_for(_collect_text(gen_events[0]), timeout=15)
    assert len(text) > 0


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_vad_speech_events(rt_session: llm.RealtimeSession):
    speech_started = asyncio.Event()
    speech_stopped = asyncio.Event()
    rt_session.on("input_speech_started", lambda _: speech_started.set())
    rt_session.on("input_speech_stopped", lambda _: speech_stopped.set())

    await _push_speech(rt_session)

    await asyncio.wait_for(speech_started.wait(), timeout=10)
    await asyncio.wait_for(speech_stopped.wait(), timeout=10)


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_input_audio_transcription(rt_session: llm.RealtimeSession):
    transcripts: list[str] = []
    transcript_received = asyncio.Event()

    def on_transcript(ev: llm.InputTranscriptionCompleted):
        transcripts.append(ev.transcript)
        transcript_received.set()

    rt_session.on("input_audio_transcription_completed", on_transcript)
    await _push_speech(rt_session, "weather_question")
    rt_session.commit_audio()

    await asyncio.wait_for(transcript_received.wait(), timeout=15)
    full = " ".join(transcripts).lower()
    assert "weather" in full or "paris" in full


# -- Chat context --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_update_chat_ctx(rt_session: llm.RealtimeSession):
    chat_ctx = llm.ChatContext()
    # TODO(long): fix assistant message for openai azure
    # chat_ctx.add_message(role="assistant", content="What is your favorite number?")
    chat_ctx.add_message(role="user", content="My favorite number is seven")
    await asyncio.wait_for(rt_session.update_chat_ctx(chat_ctx), timeout=10)

    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions="What is the user's favorite number? Reply with just the number."
        ),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "seven" in text.lower() or "7" in text


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_chat_ctx_populated_after_reply(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    await asyncio.wait_for(_collect_text(gen_ev), timeout=15)

    ctx = rt_session.chat_ctx
    assert len(ctx.items) > 0
    assert any(item.type == "message" and item.role == "assistant" for item in ctx.items)


# xAI doesn't support conversation.item.delete
@pytest.mark.parametrize("rt_session", OPENAI_AND_AZURE, indirect=True)
async def test_update_chat_ctx_replaces_history(rt_session: llm.RealtimeSession):
    ctx1 = llm.ChatContext()
    ctx1.add_message(role="user", content="Remember: color is red")
    await asyncio.wait_for(rt_session.update_chat_ctx(ctx1), timeout=10)

    ctx2 = llm.ChatContext()
    ctx2.add_message(role="user", content="Remember: color is blue")
    await asyncio.wait_for(rt_session.update_chat_ctx(ctx2), timeout=10)

    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What color did the user mention?"),
        timeout=15,
    )
    text = await asyncio.wait_for(_collect_text(gen_ev), timeout=15)
    assert "blue" in text.lower()


# -- Interruption --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_interrupt(rt_session: llm.RealtimeSession):
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions="Write a very long essay about the history of computing."
        ),
        timeout=15,
    )
    got_chunk = False
    async for msg_gen in gen_ev.message_stream:
        async for _ in msg_gen.text_stream:
            got_chunk = True
            rt_session.interrupt()
            break
        break
    assert got_chunk


# -- Function tools --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_function_tool(rt_session: llm.RealtimeSession):
    @function_tool
    async def get_weather(ctx: RunContext, city: str) -> str:
        """Get the weather for a city.
        Args:
            city: The city name
        """
        return f"Weather in {city}: sunny, 72F"

    await rt_session.update_tools([get_weather])
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What's the weather in Paris? Use the tool."),
        timeout=15,
    )

    got_function_call = False
    async for fn_call in gen_ev.function_stream:
        assert fn_call.name == "get_weather"
        got_function_call = True
    assert got_function_call


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_function_tool_reply(rt_session: llm.RealtimeSession):
    @function_tool
    async def get_capital(ctx: RunContext, country: str) -> str:
        """Get the capital of a country.
        Args:
            country: The country name
        """
        return "Paris"

    await rt_session.update_tools([get_capital])
    gen_ev = await asyncio.wait_for(
        rt_session.generate_reply(instructions="What is the capital of France? Use the tool."),
        timeout=15,
    )

    fn_call: llm.FunctionCall | None = None
    async for fc in gen_ev.function_stream:
        fn_call = fc
    assert fn_call is not None

    chat_ctx = rt_session.chat_ctx.copy()
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            id=utils.shortuuid(),
            call_id=fn_call.call_id,
            output="Paris",
            is_error=False,
        )
    )
    await asyncio.wait_for(rt_session.update_chat_ctx(chat_ctx), timeout=10)

    gen_ev2 = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    text = await asyncio.wait_for(_collect_text(gen_ev2), timeout=15)
    assert "paris" in text.lower()


# -- Session reuse across handoffs --


@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(_openai_model, id="openai"),
    ],
)
async def test_reuse_session_across_handoff(model_factory: Callable[[], llm.RealtimeModel]):
    """Populate a session with long history, simulate agent handoffs in four modes,
    validate correctness, and print handoff + first reply timing."""
    model = model_factory()
    loop = asyncio.get_event_loop()

    # build a long chat context
    long_ctx = llm.ChatContext()
    for i in range(10):
        long_ctx.add_message(role="user", content=f"Tell me fact #{i + 1}")
        long_ctx.add_message(role="assistant", content=f"Fact #{i + 1}: {i * 7} is interesting.")

    new_instructions = "You are a math tutor. Answer in one sentence."
    question = "What is 2 + 2?"

    timings: list[tuple[str, float, float]] = []  # (label, handoff_time, reply_time)

    async def _run_trial(label: str, *, reuse: bool, chat_ctx: llm.ChatContext) -> None:
        if reuse:
            session = model.session()
            await asyncio.wait_for(session.update_chat_ctx(long_ctx), timeout=15)
        else:
            session = model.session()

        try:
            t0 = loop.time()

            # measure handoff
            await asyncio.wait_for(
                session._update_session(instructions=new_instructions, chat_ctx=chat_ctx),
                timeout=15,
            )
            handoff_time = loop.time() - t0

            # measure first reply after handoff
            ctx = session.chat_ctx.copy()
            ctx.add_message(role="user", content=question)
            await asyncio.wait_for(session.update_chat_ctx(ctx), timeout=15)

            # t1 = loop.time()
            gen = await asyncio.wait_for(session.generate_reply(), timeout=15)
            reply_time = loop.time() - t0
            text = await asyncio.wait_for(_collect_text(gen), timeout=15)

            assert "4" in text
            print(f"{label} handoff time: {handoff_time:.3f}s, reply time: {reply_time:.3f}s")
            timings.append((label, handoff_time, reply_time))
        finally:
            await session.aclose()
            await asyncio.sleep(0.5)

    await _run_trial("reuse (copy ctx)", reuse=True, chat_ctx=long_ctx.copy())
    await _run_trial("reuse (empty ctx)", reuse=True, chat_ctx=llm.ChatContext())
    await _run_trial("fresh (copy ctx)", reuse=False, chat_ctx=long_ctx.copy())
    await _run_trial("fresh (empty ctx)", reuse=False, chat_ctx=llm.ChatContext())

    # print timing table
    print("\n┌──────────────────────┬──────────────┬──────────────┐")
    print("│ Mode                 │ Handoff (s)  │ Reply (s)    │")
    print("├──────────────────────┼──────────────┼──────────────┤")
    for label, ht, rt in timings:
        print(f"│ {label:<20} │ {ht:>12.3f} │ {rt:>12.3f} │")
    print("└──────────────────────┴──────────────┴──────────────┘")


# -- Remote item tracking --


@pytest.mark.parametrize("rt_session", REALTIME_MODELS, indirect=True)
async def test_remote_item_added_event(rt_session: llm.RealtimeSession):
    items_added: list[llm.RemoteItemAddedEvent] = []
    rt_session.on("remote_item_added", lambda ev: items_added.append(ev))

    gen_ev = await asyncio.wait_for(rt_session.generate_reply(), timeout=15)
    await asyncio.wait_for(_collect_text(gen_ev), timeout=15)

    assert len(items_added) > 0
    assert any(
        item.item.type == "message" and item.item.role == "assistant" for item in items_added
    )
