"""Gemini Live tools and session resumption against a local peer speaking the Live protocol."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pytest
from google.genai import types
from pydantic import BaseModel

from livekit.agents import llm
from livekit.plugins.google.realtime.realtime_api import RealtimeSession
from tests.fakes import until

if TYPE_CHECKING:
    from tests.plugins.conftest import LivePeer

Open = Callable[..., RealtimeSession]

TOOL_CALL = {"functionCalls": [{"id": "fc_1", "name": "lookup", "args": {}}]}


def answer(call_id: str = "fc_1", *, reply_required: bool = True) -> llm.ChatContext:
    """The chat context after the lookup tool ran."""
    ctx = llm.ChatContext.empty()
    ctx.items.append(llm.FunctionCall(call_id=call_id, name="lookup", arguments="{}"))
    ctx.items.append(
        llm.FunctionCallOutput(
            call_id=call_id,
            name="lookup",
            output="42",
            is_error=False,
            reply_required=reply_required,
        )
    )
    return ctx


async def lookup_called(gemini_live: LivePeer, session: RealtimeSession) -> None:
    """The server calls the lookup tool and the session hands the call on."""
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    await gemini_live.send(toolCall=TOOL_CALL)
    await until(lambda: generations, "the call reaches the session")


def texts(contents: list[dict[str, list[dict[str, list[dict[str, str]]]]]]) -> list[list[str]]:
    """The text of each client_content message, turn by turn."""
    return [
        [part["text"] for turn in c.get("turns", []) for part in turn["parts"]] for c in contents
    ]


# ---- tools


@pytest.mark.parametrize(
    "vertex, behavior, reply_required, scheduling",
    [
        (False, types.Behavior.NON_BLOCKING, False, "SILENT"),
        (False, types.Behavior.NON_BLOCKING, True, None),
        # Gemini ignores scheduling on BLOCKING tools: the result goes out regardless
        (False, types.Behavior.BLOCKING, False, None),
        # Vertex rejects scheduling; gemini-3.8-live on Vertex stalls on an id-less result
        (True, types.Behavior.NON_BLOCKING, False, None),
        (True, types.Behavior.BLOCKING, True, None),
    ],
)
async def test_tool_result_answers_its_call_with_the_scheduling_the_backend_takes(
    gemini_live: LivePeer,
    open_session: Open,
    vertex: bool,
    behavior: types.Behavior,
    reply_required: bool,
    scheduling: str | None,
) -> None:
    """A tool result carries its call id, and SILENT only where the backend honours it.

    https://github.com/livekit/agents/issues/6569 https://github.com/livekit/agents/issues/3853
    https://github.com/livekit/agents/issues/2883 https://github.com/livekit/agents/issues/4545
    https://github.com/livekit/agents/pull/5413 https://github.com/livekit/agents/pull/7591
    """
    session = open_session(vertexai=vertex, tool_behavior=behavior)
    await lookup_called(gemini_live, session)

    await session.update_chat_ctx(answer(reply_required=reply_required))
    [response] = await gemini_live.wait_sent("tool_response")

    [result] = response["functionResponses"]
    assert (result["id"], result["response"]) == ("fc_1", {"output": "42"})
    assert result.get("scheduling") == scheduling


async def test_a_call_sent_without_an_id_is_answered_without_one(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """The id the plugin made up for an id-less call never reaches the server.

    https://github.com/livekit/agents/pull/7610
    """
    session = open_session(vertexai=True, tool_behavior=types.Behavior.BLOCKING)
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    await gemini_live.send(toolCall={"functionCalls": [{"name": "lookup", "args": {}}]})
    await until(lambda: generations, "the call opens a generation")
    call = await anext(aiter(generations[0].function_stream))
    await session.update_chat_ctx(answer(call.call_id))
    [response] = await gemini_live.wait_sent("tool_response")

    assert call.call_id
    assert "id" not in response["functionResponses"][0]


async def test_tool_choice_none_rejects_calls_without_consuming_the_pending_reply(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """With tools suppressed a call is refused and the caller still gets the spoken reply.

    https://github.com/livekit/agents/issues/3265 https://github.com/livekit/agents/issues/6002
    """
    session = open_session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    session.update_options(tool_choice="none")
    pending = session.generate_reply()

    await gemini_live.send(toolCall=TOOL_CALL)
    [response] = await gemini_live.wait_sent("tool_response")
    assert not pending.done()
    assert generations == []
    await gemini_live.send(serverContent={**gemini_live.AUDIO, "turnComplete": True})

    assert await gemini_live.drain(await pending) == ("", 1, [])
    [result] = response["functionResponses"]
    assert result["id"] == "fc_1"
    assert "error" in result["response"]


async def test_cancelled_tool_calls_leave_the_session_usable(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """A tool call the server cancels does not break the session.

    https://github.com/livekit/agents/issues/1561 https://github.com/livekit/agents/issues/1354
    """
    session = open_session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)

    await gemini_live.send(toolCall=TOOL_CALL)
    await gemini_live.send(toolCallCancellation={"ids": ["fc_1"]})
    await until(lambda: generations, "the call opens a generation")
    pending = session.generate_reply()
    await gemini_live.send(
        serverContent={
            **gemini_live.AUDIO,
            "outputTranscription": {"text": "Still connected"},
            "turnComplete": True,
        }
    )

    assert await gemini_live.drain(await pending) == ("Still connected", 1, [])
    assert len(gemini_live.setups) == 1


@pytest.mark.parametrize(
    "combined",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/7195 reproduces: trailing audio opens a second generation",
            ),
        ),
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/issues/7151 reproduces: turn_complete closes the channel before tool delivery",
            ),
        ),
    ],
)
async def test_tool_boundaries_preserve_one_complete_generation(
    gemini_live: LivePeer, open_session: Open, combined: bool
) -> None:
    """Speech, a tool call and trailing audio form one generation.

    https://github.com/livekit/agents/issues/7195 https://github.com/livekit/agents/issues/7151
    """
    session = open_session()
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    speech = {
        **gemini_live.AUDIO,
        "outputTranscription": {"text": "Please wait while I look that up."},
    }

    if combined:
        await gemini_live.send(serverContent={**speech, "turnComplete": True}, toolCall=TOOL_CALL)
    else:
        await gemini_live.send(serverContent=speech)
        await gemini_live.send(toolCall=TOOL_CALL)
        await gemini_live.send(serverContent=gemini_live.AUDIO)
        await gemini_live.send(serverContent={"generationComplete": True, "turnComplete": True})
    await until(lambda: generations, "a generation starts")

    assert await gemini_live.drain(generations[0]) == (
        "Please wait while I look that up.",
        1,
        ["lookup"],
    )
    assert len(generations) == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/pull/7944 reproduces: changed tools reuse the old resumption handle",
)
async def test_tool_config_restart_is_fresh_and_delivers_the_pending_result(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """New tools start a fresh session that still receives the pending tool result.

    https://github.com/livekit/agents/issues/6479 https://github.com/livekit/agents/pull/7944
    """

    @llm.function_tool
    async def lookup() -> str:
        """Look something up."""
        return "42"

    session = open_session()
    await lookup_called(gemini_live, session)
    ctx = answer()
    output = ctx.items.pop()
    await session.update_chat_ctx(ctx)
    await gemini_live.ready(session, handle="old-config-handle")

    await session.update_tools([lookup])
    ctx.items.append(output)
    await session.update_chat_ctx(ctx)
    await until(lambda: len(gemini_live.setups) == 2, "new tools restart the session")

    assert "handle" not in gemini_live.setups[1].get("sessionResumption", {})
    [response] = await gemini_live.wait_sent("tool_response")
    assert [(r["id"], r["response"]) for r in response["functionResponses"]] == [
        ("fc_1", {"output": "42"})
    ]


async def test_async_tools_by_default_declare_non_blocking_and_answer_silently(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """On gemini-3.8-live tools are NON_BLOCKING, so a no-reply result goes out SILENT.

    https://github.com/livekit/agents/issues/7302 https://github.com/livekit/agents/pull/7329
    """

    @llm.function_tool
    async def lookup() -> str:
        """Look something up."""
        return "42"

    session = open_session(model="gemini-3.8-live")
    await gemini_live.ready(session)
    await session.update_tools([lookup])
    await until(lambda: len(gemini_live.setups) == 2, "new tools restart the session")

    await lookup_called(gemini_live, session)
    await session.update_chat_ctx(answer(reply_required=False))
    [response] = await gemini_live.wait_sent("tool_response")

    [declaration] = gemini_live.setups[1]["tools"][0]["functionDeclarations"]
    assert declaration["behavior"] == "NON_BLOCKING"
    assert response["functionResponses"][0]["scheduling"] == "SILENT"


class DecisionDetails(BaseModel):
    zeta: str
    alpha: str


@llm.function_tool
async def decide(reasoning: str, answer: Literal["a", "b"], details: DecisionDetails) -> None:
    """Reason first, then answer."""


async def test_live_tool_parameters_keep_their_declared_order(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Live tool parameters are declared in source order, nested models too.

    https://github.com/livekit/agents/pull/7387
    """
    session = open_session()

    await session.update_tools([decide])
    await until(lambda: gemini_live.setups, "the plugin connects")

    [declaration] = gemini_live.setups[0]["tools"][0]["functionDeclarations"]
    parameters = declaration["parameters"]
    assert parameters["property_ordering"] == ["reasoning", "answer", "details"]
    assert parameters["properties"]["details"]["property_ordering"] == ["zeta", "alpha"]


async def test_live_tool_schema_drops_keywords_gemini_does_not_know(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """An unknown keyword inside a $ref definition does not stop the session from connecting.

    https://github.com/livekit/agents/issues/7349
    """
    tool = llm.function_tool(
        lambda raw_arguments: None,
        raw_schema={
            "name": "create_draft",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "readOnly": True}},
                    }
                },
            },
        },
    )
    session = open_session()

    await session.update_tools([tool])
    await until(lambda: gemini_live.setups, "the plugin connects")

    [declaration] = gemini_live.setups[0]["tools"][0]["functionDeclarations"]
    assert declaration["name"] == "create_draft"
    assert "readOnly" not in json.dumps(declaration)


async def test_a_tool_schema_gemini_rejects_names_the_tool(
    open_session: Open,
) -> None:
    """A schema Gemini would reject fails the session with an error naming the tool.

    https://github.com/livekit/agents/pull/7371
    """
    bad = llm.function_tool(
        lambda raw_arguments: None,
        raw_schema={
            "name": "bad",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string", "minLength": "many"}},
            },
        },
    )
    session = open_session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    await session.update_tools([bad])
    await until(lambda: errors, "the session fails")

    assert "tool bad has a schema Gemini rejected" in str(errors[0].error)
    assert not errors[0].recoverable


# ---- session resumption


async def hello_then_handle(gemini_live: LivePeer, session: RealtimeSession) -> llm.ChatContext:
    """Sends "hello" over the first socket, then hands out the handle that covers it."""
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")
    await session.update_chat_ctx(ctx)
    await gemini_live.wait_sent("client_content")
    await gemini_live.ready(session, handle="resume-1")
    return ctx


async def test_resumed_session_does_not_replay_what_the_handle_holds(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """The server restored the conversation from the handle; replaying it would duplicate it.

    https://github.com/livekit/agents/issues/5985
    """
    session = open_session()
    await hello_then_handle(gemini_live, session)

    await gemini_live.drop()
    session.generate_reply()
    contents = await gemini_live.wait_sent("client_content")

    assert gemini_live.setups[1]["sessionResumption"]["handle"] == "resume-1"
    assert [c.get("turnComplete") for c in contents] == [True]
    assert [m.text_content for m in session.chat_ctx.messages()] == ["hello"]


async def test_resumed_session_sends_only_the_update_made_while_disconnected(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """Items added while reconnecting are new to the resumed session; the rest is not.

    https://github.com/livekit/agents/pull/6000
    """
    session = open_session()
    ctx = await hello_then_handle(gemini_live, session)
    ctx.add_message(role="user", content="one more thing")

    gemini_live.accepting.clear()
    await gemini_live.drop()
    await session.update_chat_ctx(ctx)
    gemini_live.accepting.set()

    assert texts(await gemini_live.wait_sent("client_content")) == [["one more thing"]]
    assert [m.text_content for m in session.chat_ctx.messages()] == ["hello", "one more thing"]


async def test_resumed_session_delivers_the_tool_result_made_while_disconnected(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """The resumed session still holds the call open, so the result made meanwhile answers it.

    https://github.com/livekit/agents/pull/6000
    """
    session = open_session()
    ctx = answer("call-1")
    output = ctx.items.pop()
    await session.update_chat_ctx(ctx)
    await gemini_live.ready(session, handle="resume-1")

    gemini_live.accepting.clear()
    await gemini_live.drop()
    ctx.items.append(output)
    await session.update_chat_ctx(ctx)
    gemini_live.accepting.set()

    [response] = await gemini_live.wait_sent("tool_response")
    assert [r["id"] for r in response["functionResponses"]] == ["call-1"]
    assert gemini_live.sent("client_content") == []


async def test_resumed_session_resends_what_the_handle_missed(
    gemini_live: LivePeer, open_session: Open
) -> None:
    """A message sent after the last handle is not in the server snapshot, so it is resent.

    https://github.com/livekit/agents/pull/6000
    """
    session = open_session()
    ctx = await hello_then_handle(gemini_live, session)
    ctx.add_message(role="user", content="sent before the socket dropped")
    await session.update_chat_ctx(ctx)
    await gemini_live.wait_sent("client_content", 2)

    await gemini_live.drop()

    assert texts(await gemini_live.wait_sent("client_content")) == [
        ["sent before the socket dropped"]
    ]
