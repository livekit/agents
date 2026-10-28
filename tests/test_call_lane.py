"""The call's lane (design §5, plan 3.3.7): a call commits only what it said and dropped, a text
between voice turns reaches the realtime model, an empty transcript is no turn, and a tool's
``ModelRetry`` reads as in text. Seen in what reaches the model and in the call's history."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from livekit.agents import Agent, LiveSession, RunContext, llm
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import FunctionToolCall, ModelRetry
from livekit.agents.storage import AgentDBStorage, live
from livekit.agents.storage.resolve import Thread

from .agentdb_fixture import new_session
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_realtime import FakeRealtimeModel, fake_capabilities
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
@pytest.mark.xfail(strict=True, reason="R63: the voice commit deletes every stored item it lacks")
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
@pytest.mark.xfail(strict=True, reason="B10: the realtime model's context never gets the reload")
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
@pytest.mark.xfail(strict=True, reason="B67: a closing activity gets the take's history")
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
@pytest.mark.xfail(strict=True, reason="R62: an empty transcript is committed as a turn")
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
@pytest.mark.xfail(strict=True, reason="B4: the model reads 'An internal error occurred'")
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
