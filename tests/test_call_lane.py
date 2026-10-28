"""The call's lane (design §5, plan 3.3.7): a call commits only what it said and dropped, a text
between voice turns reaches the realtime model, an empty transcript is no turn, a tool's
``ModelRetry`` reads as in text, and under ``live`` a text during the call is a turn of the call.
Seen in what reaches the model, in the call's history and on the A2A wire."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from livekit.agents import Agent, LiveSession, RunContext, llm
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.extension import INTERRUPTED
from livekit.agents.llm import FunctionToolCall, ModelRetry
from livekit.agents.storage import AgentDBStorage, live
from livekit.agents.storage.resolve import Thread

from .a2a_driver import AGENT, Door, Reply, call, no_project_env, state_of  # noqa: F401
from .agentdb_fixture import new_session
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import FakeRealtimeModel, ScriptedRealtimeModel, fake_capabilities
from .test_call_progress import Booking, answering, soon, told
from .test_live_inbox import in_the_background
from .test_storage import a2a, run_turn

pytestmark = [pytest.mark.unit]

LANE = "desk"


def reply(text: str, content: str, *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=text, content=content, ttft=0, duration=0, tool_calls=list(calls))


async def by_text(storage: AgentDBStorage, thread_id: str, text: str, seen: list[str]) -> None:
    """One text turn on the lane; ``seen`` gets the history its handler read."""

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        seen[:] = [i.text_content or "" for i in ctx.message_history.items if i.type == "message"]
        return "texted back"

    await run_turn(
        storage,
        thread_id,
        LANE,
        a2a(text, f"m_{text}", channel="sms"),
        None,
        handler,
        debounce=timedelta(0),
    )


@pytest.mark.rule("R63", "B11")
async def test_a_voice_commit_keeps_what_a_text_turn_wrote(agentdb_storage: AgentDBStorage) -> None:
    sid = await new_session(agentdb_storage, LANE)
    seen: list[str] = []
    model = FakeLLM(fake_responses=[reply("hello", "hi"), reply("again", "yes")])
    agent = Agent(instructions="desk", name=LANE)
    async with LiveSession(llm=model, thread=Thread(agentdb_storage, sid, agent_name=LANE)) as s:
        await s.start(agent)
        await s.run(user_input="hello")
        await s.aclose()
        # a text while the call is closed; the call restarts from the history it kept
        await by_text(agentdb_storage, sid, "first text", seen)
        await s.start(agent)
        await s.run(user_input="again")
        await asyncio.sleep(0.05)  # the turn's commit is chained behind the speech

    await by_text(agentdb_storage, sid, "second text", seen)
    assert "first text" in seen and "again" in seen


@pytest.mark.rule("B10")
async def test_a_text_between_voice_turns_reaches_the_realtime_model(
    agentdb_storage: AgentDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model's context is updated only when the lane moved: what it had, then the text."""
    monkeypatch.setattr(live, "HOLD_LIMIT", 0.1)  # a turn the fake never answers lets go soon
    sid = await new_session(agentdb_storage, LANE)
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    thread = Thread(agentdb_storage, sid, agent_name=LANE)
    updates: list[llm.ChatContext] = []

    def turns(ctx: llm.ChatContext) -> list[tuple[str, str | None]]:
        return [(i.id, i.text_content) for i in ctx.items if i.type == "message"]

    def hear(item_id: str, transcript: str) -> None:
        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=True),
        )

    async with LiveSession(llm=model, thread=thread) as session:
        await session.start(Agent(instructions="desk", name=LANE))
        rt = model.active_session
        update = rt.update_chat_ctx

        async def record(chat_ctx: llm.ChatContext) -> None:
            updates.append(chat_ctx.copy())
            await update(chat_ctx)

        monkeypatch.setattr(rt, "update_chat_ctx", record)
        hear("u1", "hello")
        await by_text(agentdb_storage, sid, "a text", [])
        assert updates == []

        before = turns(session.history)
        hear("u2", "still there?")
        for _ in range(100):
            if updates:
                break
            await asyncio.sleep(0.05)

    assert len(updates) == 1
    got = turns(updates[0])
    assert got[: len(before)] == before
    assert [text for _, text in got[len(before) :]] == ["a text", "texted back", "still there?"]


@pytest.mark.rule("B67")
async def test_a_call_closing_while_its_turn_takes_the_lane_waits_for_it(
    agentdb_storage: AgentDBStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The take ends before the close returns, and the closing activity gets no history."""
    sid = await new_session(agentdb_storage, LANE)
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    session: LiveSession[Any] = LiveSession(
        llm=model, thread=Thread(agentdb_storage, sid, agent_name=LANE)
    )
    closing = asyncio.Event()
    late: list[llm.ChatContext] = []
    take = live.LiveLane._take

    async def slow_take(lane: live.LiveLane) -> Any:
        await closing.wait()
        return await take(lane)

    await session.start(Agent(instructions="desk", name=LANE))
    rt = model.active_session
    activity = session._activity
    assert activity is not None
    close = activity.aclose

    async def close_slowly() -> None:
        closing.set()
        await asyncio.sleep(0.2)  # the take settles while the activity is closing
        await close()

    update = rt.update_chat_ctx

    async def record(chat_ctx: llm.ChatContext) -> None:
        if closing.is_set():
            late.append(chat_ctx)
        await update(chat_ctx)

    monkeypatch.setattr(rt, "update_chat_ctx", record)
    monkeypatch.setattr(live.LiveLane, "_take", slow_take)
    monkeypatch.setattr(activity, "aclose", close_slowly)
    await by_text(agentdb_storage, sid, "a text", [])
    rt.emit(
        "input_audio_transcription_completed",
        llm.InputTranscriptionCompleted(item_id="u1", transcript="hello", is_final=True),
    )
    await session.aclose()

    assert session._lane_turn is not None and session._lane_turn.done()
    assert late == []
    assert "could not take its lane" not in caplog.text


@pytest.mark.rule("R62", "B9")
async def test_an_empty_transcript_is_no_turn() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="desk"))
        model.active_session.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id="u1", transcript=" ", is_final=True),
        )
        await asyncio.sleep(0)
        users = [i for i in session.history.items if i.type == "message" and i.role == "user"]

    assert users == []


@pytest.mark.rule("B4")
async def test_a_model_retry_on_a_call_reads_as_in_text() -> None:
    model = FakeLLM(
        fake_responses=[
            reply(
                "cancel my visit",
                "",
                FunctionToolCall(name="cancel", arguments="{}", call_id="c1"),
            ),
            reply("say which date", "Which date?"),
        ]
    )
    agent = Agent(instructions="desk")

    @agent.tool
    async def cancel(ctx: RunContext, date: str = "") -> str:
        """Cancel a visit.

        Args:
            date: the date
        """
        raise ModelRetry("say which date")

    async with LiveSession(llm=model) as session:
        await session.start(agent)
        result: Any = await session.run(user_input="cancel my visit")

    assert result.output == "Which date?"


@pytest.mark.rule("R59", "R60")
@pytest.mark.parametrize(
    "holds",
    [
        pytest.param(
            True,
            id="live",
            marks=pytest.mark.xfail(strict=True, reason="R59, R60: the call never holds its lane"),
        ),
        pytest.param(False, id="per-turn"),
    ],
)
async def test_a_text_during_a_call_is_a_turn_of_the_call_when_the_call_holds_its_lane(
    door: Callable[..., Door], holds: bool
) -> None:
    async def handler(ctx: A2ASessionContext[Any]) -> str:
        return "the door answered"

    desk = door(handler, inbox="live" if holds else "queue")
    thread = await desk.thread()
    model = ScriptedRealtimeModel(
        Reply("Your visit is on Monday.", when=answering("when is my visit?"))
    )
    lane = Thread(desk.storage, thread, agent_name=AGENT)
    lane._call_holds_lane = holds  # as the server sets it from the lane's text door
    async with LiveSession(llm=model, thread=lane) as session:
        await session.start(Agent(instructions="front desk"))
        sent = await asyncio.wait_for(desk.send("when is my visit?", context_id=thread), 20)

    assert sent.state == "completed"
    assert sent.answer == ("Your visit is on Monday." if holds else "the door answered")
    assert sent.metadata.get("livekit/answeredByVoice", False) is holds
    # under live the realtime model had the text in its context when it replied
    assert any(answering("when is my visit?")(r) for r in model.requests) == holds


def held(desk: Door, thread: str) -> Thread:
    """The call's thread on ``desk``'s lane, held for the whole call as under ``live``."""
    lane = Thread(desk.storage, thread, agent_name=AGENT)
    lane._call_holds_lane = True  # as the server sets it from the lane's text door
    return lane


@pytest.mark.rule("B64")
@pytest.mark.xfail(strict=True, reason="B64: the text ends with the acknowledgement")
async def test_a_text_whose_work_runs_on_ends_with_the_calls_reply_to_its_result(
    door: Callable[..., Door],
) -> None:
    booking = Booking()
    model = ScriptedRealtimeModel(
        Reply("", [call("book", "b1", day="monday")]),
        Reply("Booking it now."),
        Reply("Monday is booked.", when=told("b1_final")),
    )
    desk = door(inbox="live")
    thread = await desk.thread()
    async with LiveSession(llm=model, thread=held(desk, thread)) as session:
        await session.start(booking.desk)
        sending = asyncio.create_task(desk.send("book monday", context_id=thread))
        try:
            (waiting,) = await in_the_background(desk, thread)
        finally:
            booking.second.set()
            booking.done.set()
        sent = await asyncio.wait_for(sending, 20)

    assert waiting["status"]["message"]["parts"][0]["text"] == "Booking it now."
    assert [state_of(f) for f in sent.frames if state_of(f)] == [
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
    ]
    assert sent.answer == "Monday is booked."


@pytest.mark.rule("B65")
@pytest.mark.xfail(strict=True, reason="B65: the text ends completed")
@pytest.mark.parametrize("said", ["", "Let me look at"], ids=["nothing-said", "part-said"])
async def test_a_text_whose_reply_was_cut_off_ends_as_in_text(
    door: Callable[..., Door], said: str
) -> None:
    """A sound on the line cuts the text's reply off: the task ends as a text cut off ends, with
    what was said, and the provider's own reply that follows is not its answer."""
    under_way = asyncio.Event()
    model = ScriptedRealtimeModel(Reply(said, hold=under_way), Reply("The 11:00 or the 2:30?"))
    desk = door(inbox="live")
    thread = await desk.thread()
    async with LiveSession(llm=model, thread=held(desk, thread)) as session:
        await session.start(Agent(instructions="front desk"))
        sending = asyncio.create_task(desk.send("the first one", context_id=thread))
        await soon(lambda: model.requests, "the text's reply under way")
        model.active_session.noise()
        under_way.set()
        sent = await asyncio.wait_for(sending, 20)

    assert sent.state == "canceled" and sent.metadata[INTERRUPTED] is True
    assert sent.answer == (said or None)
