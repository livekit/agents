"""A question from a delegation in the background, on a call to a realtime model that cannot
``say()``: it is spoken at once, word for word, and a text that answers it during the call gets
the call's wording of the expert's result, with each fact recorded once."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import Agent, InputRequired, LiveSession, NotGivenOr, RunContext, utils
from livekit.agents.llm import (
    ChatContext,
    FunctionCall,
    GenerationCreatedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
)
from livekit.agents.llm.tool_context import ANSWERS_KEY, SAY_AS_WRITTEN, WAITING_PREFIX

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

QUESTION = "Please confirm: Tuesday at 16:30. Yes or no?"


class ScriptedSession(FakeRealtimeSession):
    """Answers every ``generate_reply`` at once: ``reply`` maps the instructions and the
    context the model holds then to what it says and the calls it makes."""

    def __init__(self, model: ScriptedModel) -> None:
        super().__init__(model)
        self.reply = model.reply
        self.seen: list[tuple[str, ChatContext]] = []

    def generate_reply(self, *, instructions: NotGivenOr[str] = "", **_: Any) -> Any:
        fut = super().generate_reply()
        text = instructions if isinstance(instructions, str) else ""
        self.seen.append((text, self.chat_ctx.copy()))
        said, calls = self.reply(text, self.chat_ctx)
        fut.set_result(_generation(said, calls))
        return fut


class ScriptedModel(FakeRealtimeModel):
    def __init__(self, reply: Any) -> None:
        # no voice of its own: a verbatim text is spoken by asking the model to say it
        super().__init__(
            capabilities=fake_capabilities(supports_say=False, auto_tool_reply_generation=False)
        )
        self.reply = reply

    def session(self, *, turn_detection_disabled: bool = False) -> ScriptedSession:
        sess = ScriptedSession(self)
        self.created_sessions.append(sess)
        return sess


def _generation(text: str, calls: list[FunctionCall]) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities: asyncio.Future[list[str]] = asyncio.Future()
    modalities.set_result(["audio", "text"])
    if text:
        message_ch.send_nowait(
            MessageGeneration(
                message_id=utils.shortuuid("msg_"),
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        text_ch.send_nowait(text)
        samples = 2400
        audio_ch.send_nowait(
            rtc.AudioFrame(
                data=b"\x00\x01" * samples,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=samples,
            )
        )
    text_ch.close()
    audio_ch.close()
    message_ch.close()
    for call in calls:
        function_ch.send_nowait(call)
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch,
        function_stream=function_ch,
        user_initiated=True,
        response_id=utils.shortuuid("resp_"),
    )


async def test_a_background_question_is_spoken_and_a_text_answer_gets_the_result() -> None:
    expert_done = asyncio.Event()
    asked: list[str] = []

    desk = Agent(instructions="front desk")

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str | InputRequired:
        """Ask the back office."""
        asked.append(request)
        if request == "book tuesday":
            await ctx.update("Checking the calendar.")
            await expert_done.wait()
            return InputRequired(QUESTION, verbatim=True)
        return "booked Tuesday at 16:30"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)

    delegated: list[bool] = []

    def reply(instructions: str, ctx: ChatContext) -> tuple[str, list[FunctionCall]]:
        if instructions.startswith(SAY_AS_WRITTEN.split("{")[0]):
            return instructions.split(": ", 1)[1], []
        if not delegated:
            # the person's spoken request, which the model heard itself
            delegated.append(True)
            return "One moment.", [
                FunctionCall(
                    call_id="c1", name="ask", arguments=json.dumps({"request": "book tuesday"})
                )
            ]
        outputs = [i for i in ctx.items if i.type == "function_call_output"]
        if outputs and outputs[-1].output == "booked Tuesday at 16:30":
            return "You're booked for Tuesday at 4:30.", []
        return "Sure, I'm checking.", []

    model = ScriptedModel(reply)
    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(desk)
        rt = model.active_session

        rt.emit(
            "input_audio_transcription_completed",
            InputTranscriptionCompleted(item_id="u1", transcript="book me tuesday", is_final=True),
        )
        await session.generate_reply()
        # an empty transcript is no turn: not recorded, and it answers nothing
        rt.emit(
            "input_audio_transcription_completed",
            InputTranscriptionCompleted(item_id="u2", transcript="  ", is_final=True),
        )

        expert_done.set()
        for _ in range(200):
            if any(i.startswith(SAY_AS_WRITTEN.split("{")[0]) for i, _ in rt.seen):
                break
            await asyncio.sleep(0.05)
        spoken = [i for i, _ in rt.seen if i.startswith(SAY_AS_WRITTEN.split("{")[0])]
        assert spoken == [SAY_AS_WRITTEN.format(text=QUESTION)], "spoken at once, word for word"

        # the text that answers it during the call gets the call's wording of the result
        row = SimpleNamespace(message={"parts": [{"kind": "text", "text": "yes"}]})
        answer = await session._answer_texts([row])  # type: ignore[list-item]
        assert answer == "You're booked for Tuesday at 4:30."
        # the model held the forward and its output before it worded the result
        _, held = rt.seen[-1]
        assert [i.output for i in held.items if i.type == "function_call_output"][-1] == (
            "booked Tuesday at 16:30"
        )
        items = session.history.items

    assert asked == ["book tuesday", "yes"], "the answer went to the tool, as said"
    users = [i.text_content for i in items if i.type == "message" and i.role == "user"]
    assert users == ["book me tuesday", "yes"], "each message once, no empty one"
    outputs = {i.call_id: i.output for i in items if i.type == "function_call_output"}
    assert outputs["c1_final"] == f"{WAITING_PREFIX}{QUESTION}", "the question stays the output"
    forward = [i for i in items if i.type == "function_call" and ANSWERS_KEY in i.extra]
    assert len(forward) == 1 and outputs[forward[0].call_id] == "booked Tuesday at 16:30"
    said = [i.text_content for i in items if i.type == "message" and i.role == "assistant"]
    assert said.count(QUESTION) == 1
