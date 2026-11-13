"""One smoke per Python plugin base class driven through the native bridge, and the fallback
adapters wrapping them."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
)

from livekit import rtc
from livekit.agents import Agent, APIError, TokenUsage, VoiceSession, llm, stt, tts
from livekit.agents.metrics import TTSMetrics
from livekit.plugins.openai.responses import LLM as ResponsesLLM
from livekit.plugins.openai.tools import WebSearch
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeAvatar,
    FakeLLM,
    FakeRealtime,
    FakeSentenceTokenizer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    Hang,
    until,
)

HEARD = "my name is Theo"


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "batch"])
async def test_a_heard_turn_is_answered(streaming: bool) -> None:
    """A Python VAD and STT (a stream, or batch recognize behind the core's VAD) hear a turn."""
    session = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT([HEARD], streaming=streaming),
        llm=FakeLLM({HEARD: f"You said: {HEARD}"}),
        tts=FakeTTS(streaming=False),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    microphone = FakeAudioInput()
    session.input.audio = microphone
    session.output.audio = FakeAudioOutput()

    def said() -> list[tuple[str, str | None]]:
        return [(i.role, i.text_content) for i in session.history.items if i.type == "message"]

    async with session:
        await session.start(Agent(instructions="Answer.", name="desk"))
        microphone.speak(1.0)
        reply = ("assistant", f"You said: {HEARD}")
        await until(lambda: reply in said(), f"no reply to the heard turn: {said()}")
    assert ("user", HEARD) in said()


async def test_llm_fallback_marks_a_repeated_provider_unavailable_once() -> None:
    """A provider listed twice in an LLM FallbackAdapter shares one health state."""
    failure = APIError("scripted failure", retryable=True)
    primary, backup = FakeLLM([failure] * 4), FakeLLM(["Recovered."])
    adapter = llm.FallbackAdapter([primary, primary, backup], retry_interval=0, sticky=True)
    changes: list[tuple[object, bool]] = []

    @adapter.on("llm_availability_changed")
    def changed(event: llm.AvailabilityChangedEvent) -> None:
        changes.append((event.llm, event.available))

    try:
        async with adapter.chat(chat_ctx=llm.ChatContext.empty()) as stream:
            response = await stream.collect()
    finally:
        await adapter.aclose()

    assert response.text == "Recovered."
    assert changes == [(primary, False)]


async def test_stt_fallback_recovers_on_the_backup() -> None:
    """A failing STT is marked unavailable and the backup's transcript is returned."""
    failure = APIError("recognizer failed", retryable=True)
    primary, backup = FakeSTT([failure] * 4, streaming=False), FakeSTT([HEARD], streaming=False)
    adapter = stt.FallbackAdapter(
        [primary, backup], vad=FakeVAD(), max_retry_per_stt=0, retry_interval=0
    )
    unavailable: list[stt.STT] = []
    adapter.on(
        "stt_availability_changed", lambda ev: None if ev.available else unavailable.append(ev.stt)
    )
    try:
        result = await adapter.recognize([rtc.AudioFrame.create(16_000, 1, 160)])
    finally:
        await adapter.aclose()

    assert result.alternatives[0].text == HEARD
    assert unavailable == [primary]


async def test_tts_fallback_recovers_on_the_backup() -> None:
    """A failing TTS is marked unavailable and the backup synthesizes the text."""
    failure = APIError("synthesis failed", retryable=True)
    primary, backup = FakeTTS(streaming=False, error=failure), FakeTTS(streaming=False)
    adapter = tts.FallbackAdapter([primary, backup], max_retry_per_tts=0)
    changes: list[tuple[object, bool]] = []
    adapter.on("tts_availability_changed", lambda ev: changes.append((ev.tts, ev.available)))
    try:
        frame = await adapter.synthesize("hi").collect()
    finally:
        await adapter.aclose()

    assert frame.duration > 0
    assert changes == [(primary, False)]
    assert backup.texts == ["hi"]


async def test_realtime_fallback_answers_on_the_backup_when_the_primary_dies() -> None:
    """A realtime model whose session fails unrecoverably is marked unavailable once, and the
    backup speaks the reply asked after the swap."""
    primary, backup = FakeRealtime(dies="at_open"), FakeRealtime("Hello from the backup.")
    adapter = llm.RealtimeModelFallbackAdapter([primary, backup])
    changes: list[tuple[object, bool]] = []
    adapter.on(
        "realtime_availability_changed",
        lambda ev: changes.append((ev.realtime_model, ev.available)),
    )

    async with VoiceSession(llm=adapter, vad=None) as session:
        await session.start(Agent(instructions="You are realtime."))
        # the session swapped once it configures the backup; a reply asked earlier is lost (below)
        await until(
            lambda: backup.sessions and backup.sessions[-1].configured,
            "the session configured the backup",
        )
        await session.generate_reply(instructions="greet the user")

    assert changes == [(primary, False)]
    said = [i.text_content for i in session.history.items if i.type == "message"]
    assert said == ["Hello from the backup."]


async def test_a_reply_the_primary_dies_on_is_asked_again_on_the_backup() -> None:
    """A reply the primary dies on before answering is asked again on the backup, which speaks
    it (regenerate_on_swap)."""
    primary, backup = FakeRealtime(dies="at_reply"), FakeRealtime("Hello from the backup.")
    adapter = llm.RealtimeModelFallbackAdapter([primary, backup])

    async with VoiceSession(llm=adapter, vad=None) as session:
        await session.start(Agent(instructions="You are realtime."))
        await session.generate_reply(instructions="greet the user")
        await until(
            lambda: [i for i in session.history.items if i.type == "message"],
            "the backup spoke the reply the primary died on",
        )

    said = [i.text_content for i in session.history.items if i.type == "message"]
    assert said == ["Hello from the backup."]


async def test_a_custom_sentence_tokenizer_splits_the_adapters_text() -> None:
    """A StreamAdapter splits text with its TTS's sentence tokenizer, set after it was wrapped,
    and closes its stream."""
    provider, tokenizer = FakeTTS(streaming=False), FakeSentenceTokenizer()
    adapter = tts.StreamAdapter(tts=provider)
    # unpaced, so each sentence is its own request
    adapter.update_options(sentence_tokenizer=tokenizer, pacing=None)
    try:
        async with adapter.stream() as stream:
            for chunk in ("one two|three", " four|five"):
                stream.push_text(chunk)
            stream.end_input()
            frames = [audio async for audio in stream]
    finally:
        await adapter.aclose()

    assert provider.texts == ["one two", "three four", "five"]
    assert frames[-1].is_final and not any(f.is_final for f in frames[:-1])
    assert tokenizer.closed == 1


async def test_reported_usage_rides_the_final_frame_into_the_metrics() -> None:
    """A provider's reported usage rides the final frame and reaches the TTS metrics."""
    provider = FakeTTS(streaming=False, usage=TokenUsage(input_tokens=5, output_tokens=50))
    metrics: list[TTSMetrics] = []
    provider.on("metrics_collected", metrics.append)

    async with provider.synthesize("Hello there, world") as stream:
        frames = [audio async for audio in stream]

    assert [(f.usage, f.characters_count) for f in frames if not f.is_final] == [(None, None)] * (
        len(frames) - 1
    )
    assert frames[-1].is_final
    assert frames[-1].usage == TokenUsage(input_tokens=5, output_tokens=50)
    assert frames[-1].characters_count == 18
    [record] = metrics
    assert (record.tokens.input_tokens, record.tokens.output_tokens) == (5, 50)
    assert record.characters_count == 18


async def test_text_reaches_the_consumer_before_the_model_ends() -> None:
    """A streamed LLM reply is spoken by a SynthesizeStream before the LLM stream ends."""
    eof = Hang()
    text = "Your booking is ready. Let me check the next step."
    model = FakeLLM([[text, eof]])
    audio = FakeAudioOutput()
    async with VoiceSession(tts=FakeTTS(), vad=None) as session:
        session.output.audio = audio
        await session.start(Agent(instructions="Help", llm=model))
        handle = session.generate_reply(user_input="Booking")
        try:
            await audio.captured.wait()
            assert not eof.ended.is_set()
        finally:
            eof.release()
        await handle
        replies = [i.text_content for i in session.history.items if i.type == "message"]
    assert replies[-1] == text, replies
    assert len(model.requests) == 1


@dataclass
class _Answer:
    city: str


async def test_host_sdk_receives_and_core_executes_its_virtual_output_tool() -> None:
    """The OpenAI Responses plugin gets the core's output tool beside its own tools."""
    offered: list[dict[str, object]] = []  # the request's `tools`, as JSON
    call = ResponseFunctionToolCall(
        type="function_call", name="final_result", call_id="final", arguments='{"city":"Paris"}'
    )
    response = Response(
        id="response",
        created_at=0,
        model="gpt-4.1",
        object="response",
        output=[call],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
    )
    events = [
        ResponseCreatedEvent(type="response.created", sequence_number=0, response=response),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done", sequence_number=1, output_index=0, item=call
        ),
        ResponseCompletedEvent(type="response.completed", sequence_number=2, response=response),
    ]

    async def respond(request: web.Request) -> web.Response:
        offered.extend((await request.json())["tools"])
        return web.Response(
            content_type="text/event-stream",
            text="".join("data: " + event.model_dump_json() + "\n\n" for event in events),
        )

    @llm.function_tool
    async def lookup() -> str:
        return "Paris"

    app = web.Application()
    app.router.add_post("/responses", respond)
    async with TestServer(app) as server:
        model = ResponsesLLM(
            api_key="test", base_url=str(server.make_url("/")), use_websocket=False
        )
        agent = Agent(
            instructions="Find a city.", output_type=_Answer, llm=model, tools=[lookup, WebSearch()]
        )
        try:
            result = await agent.run("Find a city.")
        finally:
            await model.aclose()

    assert result.status == "completed"
    assert result.output == _Answer("Paris")
    assert {tool.get("name") for tool in offered if tool["type"] == "function"} == {
        "lookup",
        "final_result",
    }
    assert any(tool["type"] == "web_search" for tool in offered)
    [final] = [tool for tool in offered if tool.get("name") == "final_result"]
    parameters = final["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["properties"]["city"]["type"] == "string"


async def test_an_avatar_that_never_joins_times_out() -> None:
    """Waiting for an avatar that never joins the room ends at its timeout."""
    avatar = FakeAvatar()
    await avatar.start(VoiceSession(vad=None), rtc.Room())

    with pytest.raises(TimeoutError):
        await avatar.wait_for_join(timeout=0.05)
    await avatar.aclose()


async def test_closing_an_avatar_ends_a_wait_for_its_join() -> None:
    """An unbounded wait for the avatar's join ends when the avatar is closed, which closes once."""
    avatar = FakeAvatar()
    await avatar.start(VoiceSession(vad=None), rtc.Room())
    waiting = asyncio.ensure_future(avatar.wait_for_join(timeout=None))
    await asyncio.sleep(0)  # the wait starts

    await avatar.aclose()

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert avatar.closes == 1
