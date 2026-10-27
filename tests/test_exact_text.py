"""Exact text in text: what happens during a task is a status message, the answer is the artifact.

``ctx.update(text)`` is a ``working`` status, ``ctx.say(text)`` and a verbatim result the model
goes on from are a ``working`` status whose text part is flagged ``livekit/verbatim``, sent at
once; the answer artifact is flagged only when the turn ends on a verbatim result, and a
question only when it is asked as written. Read on the wire and in what reaches the model.
"""

from __future__ import annotations

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

NOT_YET = "exact text as status and artifact: not built yet"
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


def first_index(frames: list[dict[str, Any]], key: str) -> int:
    return next(i for i, f in enumerate(frames) if key in f)


def serving(agent: Agent) -> Callable[[A2ASessionContext[Any]], Any]:
    async def handler(ctx: A2ASessionContext[Any]) -> object:
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required if result.input_required is not None else result.output

    return handler


@pytest.mark.rule("R22", "R23")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
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
@pytest.mark.xfail(strict=True, reason=NOT_YET)
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
@pytest.mark.xfail(strict=True, reason=NOT_YET)
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
@pytest.mark.xfail(strict=True, reason=NOT_YET)
async def test_the_verbatim_results_of_one_step_go_out_in_call_order(
    door: Callable[..., Door],
) -> None:
    """Three verbatim results in one step, each ending the turn: every call runs, each result
    but the last is sent at once in call order, and the last is the answer."""
    model = ScriptedLLM(
        Reply(tool_calls=[call("first", "c1"), call("second", "c2"), call("third", "c3")])
    )
    desk = Agent(instructions="desk", llm=model)
    ran: list[str] = []

    @desk.tool
    async def first(ctx: RunContext) -> ToolResult:
        """First."""
        ran.append("first")
        return ToolResult("one", verbatim=True, reply_required=False)

    @desk.tool
    async def second(ctx: RunContext) -> ToolResult:
        """Second."""
        ran.append("second")
        return ToolResult("two", verbatim=True, reply_required=False)

    @desk.tool
    async def third(ctx: RunContext) -> ToolResult:
        """Third."""
        ran.append("third")
        return ToolResult("three", verbatim=True, reply_required=False)

    sent = await door(serving(desk)).send("go")

    assert ran == ["first", "second", "third"]
    assert working(sent.frames) == [("one", True), ("two", True)]
    assert answer(sent.frames) == ("three", True)
    assert len(model.requests) == 1


@pytest.mark.rule("R23")
@pytest.mark.xfail(strict=True, reason=NOT_YET)
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
