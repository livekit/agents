"""Smoke run of the copy: a session over ``_core.FakeRoom`` on virtual time, with a scripted LLM
and a silent TTS written as plugins (so both cross the core's provider views). It starts, says
something, replies, runs a tool, hands off to a second agent whose ``on_enter`` speaks, and
closes.

    python livekit-agents-next/scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys

from livekit.agents import (
    Agent,
    APIConnectOptions,
    LiveSession,
    RunContext,
    _core,
    function_tool,
    llm,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

SAMPLE_RATE = 24000


class ScriptedLLM(llm.LLM):
    """Answers from the last item of the request: a tool call for a question it has a tool
    for, the tool's output read back, else an echo."""

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
        delta: llm.ChoiceDelta
        if last.type == "function_call_output":
            delta = llm.ChoiceDelta(role="assistant", content=f"The tool said: {last.output}")
        elif (
            last.type == "message"
            and "weather" in (last.text_content or "")
            and "get_weather" in names
        ):
            call = llm.FunctionToolCall(
                name="get_weather",
                arguments=json.dumps({"city": "Paris"}),
                call_id=shortuuid("call_"),
            )
            delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
        elif (
            last.type == "message"
            and "billing" in (last.text_content or "")
            and "to_billing" in names
        ):
            call = llm.FunctionToolCall(
                name="to_billing", arguments="{}", call_id=shortuuid("call_")
            )
            delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
        else:
            text = last.text_content if last.type == "message" else "ok"
            delta = llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
        self._event_ch.send_nowait(llm.ChatChunk(id=shortuuid("chunk_"), delta=delta))


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
        samples = SAMPLE_RATE // 50 * max(1, len(self.input_text))
        output_emitter.push(b"\x00\x00" * samples)
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

    @function_tool
    async def get_weather(self, ctx: RunContext, city: str) -> str:
        """The weather in a city."""
        return f"sunny in {city}"

    @function_tool
    async def to_billing(self, ctx: RunContext) -> Agent:
        """Transfer the caller to billing."""
        return Billing()


async def settle(room: _core.FakeRoom, seconds: float) -> None:
    """Run the virtual clock forward in 20 ms steps, letting the loop work between them."""
    for _ in range(int(seconds / 0.02)):
        room.advance(0.02)
        for _ in range(5):
            await asyncio.sleep(0)


async def main() -> int:
    room = _core.FakeRoom(sample_rate=48000)
    room.join("user")
    session = LiveSession(llm=ScriptedLLM(), tts=SilentTTS(), vad=None)
    log: list[str] = []
    session.on(
        "agent_state_changed",
        lambda ev: log.append(f"agent_state {ev.old_state} -> {ev.new_state}"),
    )
    session.on(
        "conversation_item_added",
        lambda ev: log.append(f"item {ev.item.role}: {ev.item.text_content!r}"),
    )
    session.on(
        "function_tools_executed",
        lambda ev: log.append(
            f"tools {[c.name for c in ev.function_calls]} -> {[o.output for o in ev.function_call_outputs]}"
        ),
    )
    session.on("speech_created", lambda ev: log.append(f"speech_created {ev.source}"))
    session.on("close", lambda ev: log.append(f"close {ev.reason.value}"))

    await session.start(FrontDesk(), room=room)
    await settle(room, 2)

    handle = session.say("Let me say something.")
    await settle(room, 2)
    assert handle.done(), "say() did not finish"

    session.generate_reply(user_input="hello there")
    await settle(room, 3)

    session.generate_reply(user_input="what's the weather like?")
    await settle(room, 4)

    session.generate_reply(user_input="put me through to billing")
    await settle(room, 4)
    assert type(session.current_agent).__name__ == "Billing", session.current_agent

    await session.aclose()

    print("\n".join(log))
    print(f"played_bytes={room.played_bytes()} text_streams={len(room.text_streams())}")
    print("history:")
    for item in session.history.items:
        print(
            " ",
            item.type,
            getattr(item, "role", ""),
            getattr(item, "text_content", None)
            or getattr(item, "name", "")
            or getattr(item, "new_agent_id", ""),
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
