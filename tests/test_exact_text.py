"""Exact text in text: what happens during a task is a status message, the answer is the artifact.

``ctx.update(text)`` is a ``working`` status, ``ctx.say(text)`` and a verbatim result the model
goes on from are a ``working`` status whose text part is flagged ``livekit/verbatim``, sent at
once; the answer artifact is flagged only when the turn ends on a verbatim result, and a
question only when it is asked as written. Read on the wire and in what reaches the model.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.llm import ToolResult
from livekit.agents.llm.tool_context import DELIVERED

from .a2a_driver import (
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
)

pytestmark = pytest.mark.unit

NOTICE = "Calls are recorded for quality."
QUESTION = "What email address should the confirmation go to?"
VERBATIM = "livekit/verbatim"

RUST_ASK = (
    "rust ask HandlerOutput.messages/verbatim: the task row's history and flagged parts are "
    "written by the Rust commit"
)


def exact(part: dict[str, Any]) -> bool:
    return metadata_from_wire(part.get("metadata")).get(VERBATIM) is True


def working(frames: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Each ``working`` status's text parts, in order, with their flag."""
    out: list[tuple[str, bool]] = []
    for frame in frames:
        status = frame.get("statusUpdate", {}).get("status")
        if status and status["state"] == "TASK_STATE_WORKING":
            parts = status.get("message", {}).get("parts", [])
            out.extend((p.get("text", ""), exact(p)) for p in parts if "text" in p)
    return out


def answer(frames: list[dict[str, Any]]) -> tuple[str, bool]:
    """The artifact whose last chunk came last: its text, and whether every part is flagged."""
    chunks: dict[str, list[dict[str, Any]]] = {}
    last = ""
    for frame in frames:
        update = frame.get("artifactUpdate")
        if update is None:
            continue
        artifact = update["artifact"]
        parts = artifact.get("parts", [])
        if update.get("append"):
            chunks.setdefault(artifact["artifactId"], []).extend(parts)
        else:
            chunks[artifact["artifactId"]] = list(parts)
        if update.get("lastChunk"):
            last = artifact["artifactId"]
    parts = chunks.get(last, [])
    return "".join(p.get("text", "") for p in parts), bool(parts) and all(map(exact, parts))


def answer_parts(frames: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """The last chunk of the answer artifact: each text part, with its flag."""
    last = [f["artifactUpdate"] for f in frames if f.get("artifactUpdate", {}).get("lastChunk")]
    parts = last[-1]["artifact"].get("parts", []) if last else []
    return [(p.get("text", ""), exact(p)) for p in parts]


def first_index(frames: list[dict[str, Any]], key: str) -> int:
    return next(i for i, f in enumerate(frames) if key in f)


def serving(agent: Agent) -> Callable[[A2ASessionContext[Any]], Any]:
    async def handler(ctx: A2ASessionContext[Any]) -> object:
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required if result.input_required is not None else result.output

    return handler


@pytest.mark.rule("R22", "R23")
async def test_a_tool_says_exact_text_at_once_and_updates_in_words(
    door: Callable[..., Door],
) -> None:
    """``ctx.update`` is an unflagged ``working`` status, ``ctx.say`` a flagged one, both sent
    as they happen; the said text is the agent's message in the history the model reads next."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "c1")]),
        Reply("Booked for Tuesday.", when=output_of("c1")),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def book(ctx: RunContext) -> str:
        """Book."""
        await ctx.update("checking the calendar")
        await ctx.say(NOTICE)  # type: ignore[attr-defined]
        return "booked"

    sent = await door(serving(desk)).send("book me")

    assert sent.state == "completed"
    assert working(sent.frames) == [("checking the calendar", False), (NOTICE, True)]
    assert answer(sent.frames) == ("Booked for Tuesday.", False)
    assert VERBATIM not in sent.metadata
    said = [i for i in model.requests[-1].items if i.type == "message" and i.role == "assistant"]
    assert [m.text_content for m in said] == [NOTICE]


@pytest.mark.rule("R23", "B12")
async def test_a_verbatim_result_the_model_goes_on_from_is_sent_at_once(
    door: Callable[..., Door],
) -> None:
    """A verbatim result that asks for a reply is sent when the tool returns, flagged; the model
    reads that it was delivered, and its own answer is the artifact, unflagged."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("disclose", "d1")]),
        Reply("How can I help?", when=output_of("d1")),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True)

    sent = await door(serving(desk)).send("hi")

    assert working(sent.frames) == [(NOTICE, True)]
    assert answer(sent.frames) == ("How can I help?", False)
    assert first_index(sent.frames, "statusUpdate") < first_index(sent.frames, "artifactUpdate")
    (output,) = [i for i in model.requests[-1].items if i.type == "function_call_output"]
    assert output.output == DELIVERED.format(text=NOTICE)


@pytest.mark.rule("R23", "B12")
async def test_a_verbatim_result_that_ends_the_turn_is_the_flagged_answer(
    door: Callable[..., Door],
) -> None:
    """A verbatim result with no reply is the answer, its part flagged, and no model step
    follows it."""
    model = ScriptedLLM(Reply(tool_calls=[call("disclose", "d1")]))
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True, reply_required=False)

    sent = await door(serving(desk)).send("hi")

    assert sent.state == "completed"
    assert working(sent.frames) == []
    assert answer(sent.frames) == (NOTICE, True)
    assert VERBATIM not in sent.metadata
    assert len(model.requests) == 1


@pytest.mark.rule("R23")
async def test_a_question_asked_as_written_is_a_flagged_status_part(
    door: Callable[..., Door],
) -> None:
    """A verbatim question is the ``input-required`` status message with its text part flagged;
    the status metadata carries no second copy of it."""
    model = ScriptedLLM(Reply(tool_calls=[call("confirm", "q1")]))
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def confirm(ctx: RunContext) -> InputRequired:
        """Confirm."""
        return InputRequired(QUESTION, verbatim=True)

    sent = await door(serving(desk)).send("book me")

    final = sent.frames[-1]["statusUpdate"]
    assert final["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    parts = final["status"]["message"]["parts"]
    assert [(p["text"], exact(p)) for p in parts] == [(QUESTION, True)]
    assert VERBATIM not in metadata_from_wire(final.get("metadata"))


@pytest.mark.rule("R23")
@pytest.mark.xfail(strict=True, reason=RUST_ASK)
async def test_exact_text_said_during_a_task_is_kept_in_its_history(
    door: Callable[..., Door],
) -> None:
    """A client that reads only ``tasks/get`` sees what was said as written, flagged, in the
    task's history."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "c1")]),
        Reply("Booked.", when=output_of("c1")),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def book(ctx: RunContext) -> str:
        """Book."""
        await ctx.say(NOTICE)  # type: ignore[attr-defined]
        return "booked"

    sent = await door(serving(desk)).send("book me")

    said = [
        (p.get("text"), exact(p))
        for m in sent.task.get("history", [])
        if m.get("role") == "ROLE_AGENT"
        for p in m.get("parts", [])
    ]
    assert said == [(NOTICE, True)]


@pytest.mark.rule("R23")
@pytest.mark.xfail(strict=True, reason=RUST_ASK)
async def test_tasks_get_shows_an_exact_answer_flagged(door: Callable[..., Door]) -> None:
    """The task as ``tasks/get`` reads it carries the same flag as the stream."""
    model = ScriptedLLM(Reply(tool_calls=[call("disclose", "d1")]))
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True, reply_required=False)

    sent = await door(serving(desk)).send("hi")

    parts = sent.task["artifacts"][-1]["parts"]
    assert [(p["text"], exact(p)) for p in parts] == [(NOTICE, True)]


# ---------------------------------------------------------------------------------------------
# the calls of one step run together; its turn-ending results are one answer
# ---------------------------------------------------------------------------------------------


async def test_two_slow_tools_of_one_step_finish_in_the_time_of_one() -> None:
    """The calls of one step start together, so two slow tools take the time of the slower."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("slow", "c1"), call("slow", "c2")]),
        Reply("Both done.", when=output_of("c2")),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def slow(ctx: RunContext) -> str:
        """Slow."""
        await asyncio.sleep(0.4)
        return "done"

    started = time.monotonic()
    result = await desk.run("go")
    assert result.output == "Both done."
    assert time.monotonic() - started < 0.7


@pytest.mark.rule("R22", "R23")
async def test_says_from_two_tools_of_one_step_go_out_as_they_finish(
    door: Callable[..., Door],
) -> None:
    """Two tools of one step say exact text: each goes out when its tool says it, so the faster
    tool's text comes first even though it was called second."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("late", "c1"), call("early", "c2")]),
        Reply("Done."),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def late(ctx: RunContext) -> str:
        """Late."""
        await asyncio.sleep(0.3)
        await ctx.say("from the slow tool")
        return "ok"

    @desk.tool
    async def early(ctx: RunContext) -> str:
        """Early."""
        await ctx.say("from the fast tool")
        return "ok"

    sent = await door(serving(desk)).send("go")

    assert working(sent.frames) == [("from the fast tool", True), ("from the slow tool", True)]
    assert answer(sent.frames) == ("Done.", False)


@pytest.mark.rule("R23")
async def test_a_turn_ending_result_is_said_when_another_asks_for_a_reply(
    door: Callable[..., Door],
) -> None:
    """A verbatim result that would end the turn, with a result after it that asks for a
    reply: the first is said as written, and the model goes on."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("disclose", "c1"), call("look_up", "c2")]),
        Reply("Your order shipped.", when=output_of("c2")),
    )
    desk = Agent(instructions="desk", llm=model)

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True, reply_required=False)

    @desk.tool
    async def look_up(ctx: RunContext) -> str:
        """Look up."""
        return "shipped"

    sent = await door(serving(desk)).send("where is my order?")

    assert working(sent.frames) == [(NOTICE, True)]
    assert answer(sent.frames) == ("Your order shipped.", False)
    assert len(model.requests) == 2


async def test_a_cancel_during_a_step_stops_every_call() -> None:
    """The run is cancelled while its step's two calls run: both are cancelled."""
    model = ScriptedLLM(Reply(tool_calls=[call("wait", "c1"), call("wait", "c2")]))
    desk = Agent(instructions="desk", llm=model)
    started: list[str] = []
    stopped: list[str] = []

    @desk.tool
    async def wait(ctx: RunContext) -> str:
        """Wait."""
        started.append(ctx.function_call.call_id)
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopped.append(ctx.function_call.call_id)
            raise
        return "done"

    run = asyncio.ensure_future(desk.run("go"))
    for _ in range(100):
        if len(started) == 2:
            break
        await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert sorted(stopped) == ["c1", "c2"]


# ---------------------------------------------------------------------------------------------
# several exact results are one text; an exact answer keeps the agent's output type
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R23")
async def test_the_turn_ending_results_of_one_step_are_one_exact_text(
    door: Callable[..., Door],
) -> None:
    """Three verbatim results that end the turn: the answer is one text, the results in call
    order, as one flagged part, and nothing is sent before it."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("first", "c1"), call("second", "c2"), call("third", "c3")])
    )
    desk = Agent(instructions="desk", llm=model)

    def ending(text: str, delay: float) -> Any:
        async def tool(ctx: RunContext) -> ToolResult:
            await asyncio.sleep(delay)
            return ToolResult(text, verbatim=True, reply_required=False)

        return tool

    desk.tool(name="first", description="First.")(ending("one", 0.2))
    desk.tool(name="second", description="Second.")(ending("two", 0.0))
    desk.tool(name="third", description="Third.")(ending("three", 0.1))

    sent = await door(serving(desk)).send("go")

    assert working(sent.frames) == []
    assert answer_parts(sent.frames) == [("one\n\ntwo\n\nthree", True)]
    assert len(model.requests) == 1


@pytest.mark.rule("B46", "R23")
async def test_an_exact_result_on_a_typed_agent_is_said_and_the_output_keeps_its_type(
    door: Callable[..., Door],
) -> None:
    """On an agent whose output is a model, a verbatim result that would end the turn is said
    as written, and the model goes on to produce the typed output."""
    from pydantic import BaseModel

    class Booking(BaseModel):
        day: str

    model = ScriptedLLM(
        Reply(tool_calls=[call("disclose", "c1")]),
        Reply(tool_calls=[call("final_result", "c2", day="Tuesday")]),
    )
    desk = Agent(instructions="desk", llm=model, output_type=Booking)
    outputs: list[object] = []

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(NOTICE, verbatim=True, reply_required=False)

    async def handler(ctx: A2ASessionContext[Any]) -> object:
        result = await desk.run(ctx.message, message_history=ctx.message_history)
        outputs.append(result.output)
        return result.output.model_dump() if isinstance(result.output, Booking) else None

    sent = await door(handler).send("book me")

    assert sent.state == "completed"
    assert outputs == [Booking(day="Tuesday")]
    assert working(sent.frames) == [(NOTICE, True)]
