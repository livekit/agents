"""Smoke run of the copy: a session over ``_core.FakeRoom`` on virtual time, with a scripted LLM,
a scripted STT and a silent TTS written as plugins (so all three cross the core's provider views).
It starts, says something, replies, runs a tool, handles speech from the room,
explicitly switches to a second agent whose ``on_enter`` speaks, and closes.
Then a second session on a scripted realtime model replies through the core's realtime leg.

    python livekit-agents-next/scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys

from livekit import rtc
from livekit.agents import (
    Agent,
    APIConnectOptions,
    LiveSession,
    RunContext,
    _core,
    function_tool,
    llm,
    stt,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

SAMPLE_RATE = 24000

# (words in the user's message, the tool it calls, its arguments from the message)
TOOL_CALLS = [
    ("weather", "get_weather", lambda text: {"city": "Paris"}),
    ("billing", "to_billing", lambda text: {}),
]


class ScriptedLLM(llm.LLM):
    """Calls the tool a message asks for when the agent has it, reads a tool's output back,
    else echoes."""

    def chat(
        self, *, chat_ctx: llm.ChatContext, tools: list[llm.Tool] | None = None, **_: object
    ) -> llm.LLMStream:
        return _Stream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )


class _Stream(llm.LLMStream):
    async def _run(self) -> None:
        last = self._chat_ctx.items[-1]
        names = {t.info.name for t in self._tools if isinstance(t, llm.FunctionTool)}
        text = (last.text_content or "") if last.type == "message" else ""
        delta = llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
        if last.type == "function_call_output":
            delta = llm.ChoiceDelta(role="assistant", content=f"The tool said: {last.output}")
        for words, name, arguments in TOOL_CALLS:
            if words in text and name in names:
                call = llm.FunctionToolCall(
                    name=name, arguments=json.dumps(arguments(text)), call_id=shortuuid("call_")
                )
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
                break
        self._event_ch.send_nowait(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


class ScriptedSTT(stt.STT):
    """A streaming STT that hears "my name is Theo" in half a second of anything."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(self, **_: object) -> stt.RecognizeStream:
        return _Recognize(stt=self, conn_options=DEFAULT_API_CONNECT_OPTIONS)


class _Recognize(stt.RecognizeStream):
    async def _run(self) -> None:
        heard = 0.0
        async for frame in self._input_ch:
            if not isinstance(frame, rtc.AudioFrame) or heard < 0:
                continue
            heard += frame.duration
            if heard >= 0.5:
                heard = -1.0
                said = stt.SpeechData(language="en", text="my name is Theo", confidence=1.0)
                for kind, alternatives in (
                    (stt.SpeechEventType.START_OF_SPEECH, []),
                    (stt.SpeechEventType.FINAL_TRANSCRIPT, [said]),
                    (stt.SpeechEventType.END_OF_SPEECH, []),
                ):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=kind, alternatives=alternatives)
                    )


class SilentTTS(tts.TTS):
    """20 ms of silence per character, as one chunked synthesis."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _Chunked(tts=self, input_text=text, conn_options=conn_options)


class _Chunked(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid("req_"),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 50 * max(1, len(self.input_text))))
        output_emitter.flush()


class Billing(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You handle billing.")

    async def on_enter(self) -> None:
        self.session.say("Billing here, how can I help?")


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are the front desk.")

    async def on_enter(self) -> None:
        self.session.say("Hello, front desk.")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        print(f"on_user_turn_completed {new_message.text_content!r}")

    @function_tool
    async def get_weather(self, ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @function_tool
    async def to_billing(self, ctx: RunContext) -> None:
        """Transfer the caller to billing."""
        ctx.session.update_agent(Billing())


async def settle(room: _core.FakeRoom, seconds: float) -> None:
    """Run the virtual clock forward in 20 ms steps, letting the loop work between them."""
    for _ in range(int(seconds / 0.02)):
        room.advance(0.02)
        for _ in range(5):
            await asyncio.sleep(0)


class ScriptedRealtime(llm.RealtimeModel):
    """A realtime model that answers each reply request with one spoken message."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=True,
            )
        )

    def session(self, *, turn_detection_disabled: bool = False) -> llm.RealtimeSession:
        return _RealtimeSession(self)


class _RealtimeSession(llm.RealtimeSession):
    def __init__(self, model: ScriptedRealtime) -> None:
        super().__init__(model)
        self._chat_ctx = llm.ChatContext.empty()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def update_options(self, **_: object) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(self, **_: object) -> asyncio.Future[llm.GenerationCreatedEvent]:
        async def _one(value: object) -> object:
            yield value

        modalities: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        silence = rtc.AudioFrame(b"\x00\x00" * 4800, SAMPLE_RATE, 1, 4800)
        message = llm.MessageGeneration(
            message_id=shortuuid("msg_"),
            text_stream=_one("Hello from the realtime model."),
            audio_stream=_one(silence),
            modalities=modalities,
        )
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        fut.set_result(
            llm.GenerationCreatedEvent(
                message_stream=_one(message),
                function_stream=_empty(),
                user_initiated=True,
                response_id=shortuuid("resp_"),
            )
        )
        return fut

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        pass


async def _empty() -> object:
    return
    yield


async def realtime() -> None:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(llm=ScriptedRealtime(), vad=None)
    session.on(
        "conversation_item_added",
        lambda ev: print(f"conversation_item {ev.item.role}: {ev.item.text_content!r}"),
    )
    await session.start(Agent(instructions="You are realtime.", name="realtime"), room=room)
    handle = session.generate_reply(instructions="greet the user")
    await settle(room, 2)
    assert handle.done(), "the realtime reply did not finish"
    await session.aclose()
    print(f"played_bytes={room.played_bytes()}")


def agent_name(session: LiveSession) -> str:
    return type(session.current_agent).__name__


async def main() -> int:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(stt=ScriptedSTT(), llm=ScriptedLLM(), tts=SilentTTS(), vad=None)

    def on_tools(ev: object) -> None:
        calls = [(c.name, o.output) for c, o in ev.zipped()]  # type: ignore[attr-defined]
        print(f"function_tools_executed {calls}")

    session.on("agent_state_changed", lambda ev: print(f"agent_state {ev.new_state}"))
    session.on("user_input_transcribed", lambda ev: print(f"user_input {ev.transcript!r}"))
    session.on(
        "conversation_item_added",
        lambda ev: print(f"conversation_item {ev.item.role}: {ev.item.text_content!r}"),
    )
    session.on("function_tools_executed", on_tools)
    session.on("speech_created", lambda ev: print(f"speech_created {ev.source}"))
    session.on("close", lambda ev: print(f"close {ev.reason.value}"))

    print("== start")
    await session.start(FrontDesk(), room=room)
    await settle(room, 2)

    print("== say")
    handle = session.say("Let me say something.")
    await settle(room, 2)
    assert handle.done(), "say() did not finish"

    print("== generate_reply")
    session.generate_reply(user_input="hello there")
    await settle(room, 3)

    print("== tool")
    session.generate_reply(user_input="what's the weather like?")
    await settle(room, 4)

    print("== speech from the room")
    room.speak(frames=40)
    await settle(room, 6)

    print("== session.run(), asserted")
    result = session.run(user_input="what's the weather like?")
    await settle(room, 4)
    assert result.done(), "the run did not settle"
    result.expect.next_event().is_function_call(name="get_weather")
    result.expect.next_event().is_function_call_output(output="sunny in Paris")
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()
    print(f"run events: {[type(e).__name__ for e in result.events]}")

    print("== handoff")
    session.generate_reply(user_input="put me through to billing")
    await settle(room, 4)
    assert agent_name(session) == "Billing", agent_name(session)

    print("== close")
    await session.aclose()

    print(f"played_bytes={room.played_bytes()} text_streams={len(room.text_streams())}")
    print("history:")
    for item in session.history.items:
        detail = getattr(item, "text_content", None) or getattr(item, "name", "")
        print(f"  {item.type} {getattr(item, 'role', '')} {detail or ''}".rstrip())

    print("== agent.run(), the text path")
    text = await Agent(instructions="You answer by text.", name="text", llm=ScriptedLLM()).run(
        "hello by text"
    )
    print(f"output {text.output!r}")
    assert text.output == "You said: hello by text", text.output

    print("== realtime")
    await realtime()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
