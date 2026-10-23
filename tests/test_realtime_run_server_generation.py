"""A realtime generation the server starts on its own while ``session.run()`` is open belongs
to that run.

Auto-continuing realtime models (Gemini Live, gpt-live) answer a tool result without a
``generate_reply()``; after a handoff the continuation lands on the new activity, where no
tool-reply placeholder waits for it. The speech is still scheduled and recorded in the chat
history, so the run must record it too.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool, utils
from livekit.agents.llm import ChatContext, FunctionCall, GenerationCreatedEvent, MessageGeneration

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession

pytestmark = pytest.mark.unit

GREETING = "could you give me your ID number?"


def _generation(text: str | None, fnc: FunctionCall | None, *, rid: str) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    if text is not None:
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["text"])
        message_ch.send_nowait(
            MessageGeneration(
                message_id=f"msg-{rid}",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        text_ch.send_nowait(text)
        text_ch.close()
        audio_ch.close()
    message_ch.close()
    if fnc is not None:
        function_ch.send_nowait(fnc)
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=False,
        response_id=rid,
    )


class _AutoContinuingSession(FakeRealtimeSession):
    """Answers the first generate_reply() with a tool call and, like Gemini Live, continues on
    its own once the tool result is pushed. The protocol carries no response ids, so the
    plugin hands that continuation to whichever generate_reply() is pending (here the
    handed-off agent's), and the greeting itself arrives as the server's own generation."""

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        known = {item.id for item in self._chat_ctx.items}
        await super().update_chat_ctx(chat_ctx)
        if any(
            item.type == "function_call_output" and item.id not in known for item in chat_ctx.items
        ):
            asyncio.get_running_loop().call_later(0.2, self._continue)

    def _continue(self) -> None:
        pending = next((fut for fut in self._reply_futs if not fut.done()), None)
        continuation = _generation("", None, rid="continuation")
        if pending is not None:
            continuation.user_initiated = True
            pending.set_result(continuation)
        else:
            self.emit("generation_created", continuation)
        asyncio.get_running_loop().call_later(
            0.2, self.emit, "generation_created", _generation(GREETING, None, rid="greeting")
        )

    def generate_reply(self, **kwargs):  # type: ignore[override]
        fut = super().generate_reply(**kwargs)
        if self.generate_reply_calls == 1:
            ev = _generation(
                None, FunctionCall(call_id="c1", name="verify", arguments="{}"), rid="r1"
            )
            ev.user_initiated = True
            fut.set_result(ev)
        return fut


class _AutoContinuingModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> _AutoContinuingSession:
        sess = _AutoContinuingSession(self, turn_detection_disabled=turn_detection_disabled)
        self.created_sessions.append(sess)
        return sess


class IdAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="collect the id number")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask for the id number")
        # on_enter is still running when the server's continuation arrives
        await asyncio.sleep(1.0)


class Root(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="root")

    @function_tool
    async def verify(self, ctx: RunContext) -> Agent:
        """Called when the user wants to verify their identity."""
        return IdAgent()


async def test_server_generation_after_handoff_is_recorded_in_the_run() -> None:
    async with AgentSession(llm=_AutoContinuingModel()) as sess:
        await sess.start(Root())

        result = await asyncio.wait_for(sess.run(user_input="verify me"), timeout=5.0)

        result.expect.next_event().is_function_call(name="verify")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=IdAgent)
        result.expect.next_event().is_message(role="assistant")
        assert GREETING in [
            item.text_content for item in sess.history.items if item.type == "message"
        ]
