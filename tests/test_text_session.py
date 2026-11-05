"""Local child conversations through the public API and generated Rust persistence codec."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import pytest
from pydantic import AfterValidator, BaseModel, Field

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    APIError,
    APIStatusError,
    RunContext,
    TextSession,
    TextSessionState,
    persisted,
)
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionToolCall,
    LLMStream,
)
from livekit.agents.storage.session import ContextStore

pytestmark = pytest.mark.unit


@dataclass
class Reply:
    text: str = ""
    calls: list[FunctionToolCall] = field(default_factory=list)
    error: bool = False
    blocked: asyncio.Event | None = None


def call(name: str, arguments: dict[str, object], id: str) -> Reply:
    return Reply(calls=[FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=id)])


class Script(LLM):
    def __init__(self, *replies: Reply) -> None:
        super().__init__()
        self.replies = deque(replies)
        self.requests: list[ChatContext] = []

    def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
        self.requests.append(chat_ctx.copy())
        if not self.replies:
            raise RuntimeError("unexpected model request")
        return ScriptStream(self, chat_ctx=chat_ctx, reply=self.replies.popleft())


class ScriptStream(LLMStream):
    def __init__(self, model: Script, *, chat_ctx: ChatContext, reply: Reply) -> None:
        super().__init__(
            model, chat_ctx=chat_ctx, tools=[], conn_options=DEFAULT_API_CONNECT_OPTIONS
        )
        self.reply = reply

    async def _run(self) -> None:
        if self.reply.error:
            raise APIError("scripted failure", retryable=False)
        if self.reply.blocked is not None:
            self.reply.blocked.set()
            await asyncio.Future()
        self._event_ch.send_nowait(
            ChatChunk(
                id="response",
                delta=ChoiceDelta(
                    role="assistant", content=self.reply.text, tool_calls=self.reply.calls
                ),
            )
        )


@dataclass
class Task:
    nights: Annotated[int, Field(ge=1)]
    label: Annotated[str, AfterValidator(str.upper)] = "READY"


class Booking(BaseModel):
    reference: str


@dataclass
class App:
    service: object
    seen: list[object] = field(default_factory=list)


@dataclass
class Conversation:
    session_state: TextSessionState | None = persisted(default=None)


def restore(state: TextSessionState) -> TextSessionState:
    return TextSessionState._from_json(state._to_json())


def history(frame: Any) -> list[Any]:
    return list(frame.history)


@pytest.mark.parametrize("mode", ["fresh", "fork"])
@pytest.mark.parametrize("nights", [2, 5])
async def test_text_session_reconstruction_resolves_the_original_parent_call_once(
    mode: Literal["fresh", "fork"],
    nights: int,
) -> None:
    parent_model = Script(
        call("book", {"nights": str(nights), "label": "new"}, "original"),
        Reply("Booked"),
        Reply("Welcome back"),
    )
    child_model = Script(
        Reply("Which date?"), call("final_result", {"reference": "R2"}, "accepted")
    )
    child = Agent(name="booking", instructions="Book", llm=child_model, output_type=Booking)
    app = App(object())

    @child.instructions
    def child_context(ctx: RunContext[App]) -> str:
        assert ctx.context.service is app.service
        ctx.context.seen.append(ctx.context.service)
        return "Use the supplied service"

    @child.output_validator
    def accepted(ctx: RunContext[App], value: Booking) -> Booking:
        return Booking(reference=value.reference + "-validated")

    parent = Agent(
        name="desk",
        instructions="Help",
        llm=parent_model,
        tools=[
            child.as_tool(
                tool_name="book", parameters=Task, chat_history=mode, merge_chat_history=True
            )
        ],
    )
    first = await TextSession(agent=parent).run("Book a room", context=app)
    assert first.result.status == "input_required" and first.result.question == "Which date?"
    saved = restore(first.state)
    root_id, active_id = saved.root_frame_id, saved.active_frame_id
    assert saved.ancestors == [root_id]
    child_frame = saved.frames[active_id]
    assert child_frame.parent.call_id == "original"
    assert json.loads(child_model.requests[0].items[-1].text_content) == {
        "nights": nights,
        "label": "NEW",
    }
    assert not any(
        i.type == "function_call" and i.call_id == "original" for i in child_model.requests[0].items
    )
    assert any(
        i.type == "message" and i.text_content == "Book a room"
        for i in child_model.requests[0].items
    ) == (mode == "fork")

    first_app = app
    app = App(object())
    second = await TextSession(agent=parent).run("Friday", state=saved, context=app)
    assert second.result.status == "completed" and second.result.output == "Booked"
    assert second.state.frames[active_id].completed and second.state.ancestors == []
    assert second.state.execution_id == saved.execution_id
    root = history(second.state.frames[root_id])
    outputs = [i for i in root if i.type == "function_call_output" and i.call_id == "original"]
    assert len(outputs) == 1 and json.loads(outputs[0].output) == {"reference": "R2-validated"}
    assert sum(i.type == "message" and i.text_content == "Which date?" for i in root) == 1
    assert not any(
        i.type == "message" and i.text_content == "Friday" for i in history(saved.frames[root_id])
    )
    assert first_app.seen == [first_app.service]
    assert app.seen == [app.service]

    third = await TextSession(agent=parent).run("Thanks", state=restore(second.state), context=app)
    assert third.result.output == "Welcome back"
    assert (
        sum(
            i.type == "message" and i.text_content == "Which date?"
            for i in history(third.state.frames[root_id])
        )
        == 1
    )


async def test_text_session_state_saves_and_loads_in_a_marked_context_field() -> None:
    agent = Agent(name="desk", instructions="Help", llm=Script(Reply("Hello"), Reply("Again")))
    conversation = Conversation()
    store = ContextStore()
    await store.load_context(conversation)
    first = await TextSession(agent=agent).run("Hi")
    conversation.session_state = first.state
    await store.save_context(conversation)
    restored = Conversation()
    store = ContextStore(scopes=[write.scope for write in store._scope_writes])
    await store.load_context(restored)
    assert restored.session_state is not None
    second = await TextSession(agent=agent).run("More", state=restored.session_state)
    assert second.result.output == "Again"
    assert second.state.session_id == first.state.session_id


async def test_rejected_child_approval_restores_without_starting_the_child() -> None:
    parent_model = Script(
        call("book", {"nights": 2}, "approval"), Reply("May I start?"), Reply("I will not start")
    )
    child_model = Script()
    child = Agent(name="booking", instructions="Book", llm=child_model, output_type=Booking)
    parent = Agent(
        name="desk",
        instructions="Help",
        llm=parent_model,
        tools=[child.as_tool(tool_name="book", parameters=Task, needs_approval=True)],
    )
    first = await TextSession(agent=parent).run("Book")
    assert first.result.status == "input_required"
    root = first.state.frames[first.state.root_frame_id]
    assert len(root.pending_calls) == 1
    second = await TextSession(agent=parent).run("no", state=restore(first.state))
    assert second.result.status == "completed" and not child_model.requests
    assert not second.state.frames[second.state.root_frame_id].pending_calls
    output = next(
        i
        for i in second.result.all_messages()
        if i.type == "function_call_output" and i.call_id == "approval"
    )
    assert output.output == "no"


async def test_invalid_child_parameters_never_start_a_frame() -> None:
    child_model = Script()
    child = Agent(name="booking", instructions="Book", llm=child_model, output_type=Booking)
    parent = Agent(
        name="desk",
        instructions="Help",
        llm=Script(call("book", {"nights": 0}, "bad"), Reply("Try a positive number")),
        tools=[child.as_tool(tool_name="book", parameters=Task)],
    )
    turn = await TextSession(agent=parent).run("Book")
    assert len(turn.state.frames) == 1 and not child_model.requests
    output = next(i for i in turn.result.new_messages() if i.type == "function_call_output")
    assert output.is_error and "nights" in output.output


async def test_failure_and_stream_cancellation_produce_no_replacement_state() -> None:
    agent = Agent(name="desk", instructions="Help", llm=Script(Reply("Ready"), Reply(error=True)))
    first = await TextSession(agent=agent).run("Hi")
    saved = first.state._to_json()
    with pytest.raises(Exception, match="scripted failure"):
        await TextSession(agent=agent).run("Fail", state=first.state)
    assert first.state._to_json() == saved
    started = asyncio.Event()
    blocked = Agent(name="desk", instructions="Help", llm=Script(Reply(blocked=started)))
    async with TextSession(agent=blocked).run_stream("Hi") as stream:
        await started.wait()
        stream.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stream.result()


def test_local_agent_tools_require_explicit_output_and_refuse_callable_history() -> None:
    with pytest.raises(ValueError, match="output_type"):
        Agent(name="child", instructions="Book").as_tool()
    child = Agent(name="child", instructions="Book", output_type=str)
    with pytest.raises(ValueError, match="chat_history"):
        child.as_tool(chat_history=lambda ctx: [])  # type: ignore[arg-type]


async def test_default_depth_runs_eight_frames_and_refuses_the_ninth() -> None:
    models = [Script() for _ in range(9)]
    child = Agent(name="a8", instructions="Descend", llm=models[8], output_type=str)
    for i in reversed(range(8)):
        child = Agent(
            name=f"a{i}",
            instructions="Descend",
            llm=models[i],
            output_type=str,
            tools=[child.as_tool(tool_name="descend")],
        )
        models[i].replies.extend([call("descend", {}, f"child-{i}"), Reply(f"level {i}")])
    turn = await TextSession(agent=child).run("Start")
    assert turn.result.status == "completed" and turn.result.output == "level 0"
    assert len(turn.state.frames) == 8 and not models[8].requests
    deepest = next(frame for frame in turn.state.frames.values() if frame.agent_name == "a7")
    refusal = next(item for item in history(deepest) if item.type == "function_call_output")
    assert refusal.is_error and "max_depth=8" in refusal.output


@pytest.mark.parametrize("view", ["events", "text", "result"])
async def test_stream_views_merge_only_delivered_child_dialogue(
    view: Literal["events", "text", "result"],
) -> None:
    child = Agent(
        name="booking",
        instructions="Book",
        llm=Script(Reply("Which date?"), call("final_result", {"reference": "R"}, "final")),
        output_type=Booking,
    )
    parent = Agent(
        name="desk",
        instructions="Help",
        llm=Script(call("book", {}, "book"), Reply("Done")),
        tools=[child.as_tool(tool_name="book", merge_chat_history=True)],
    )
    async with TextSession(agent=parent).run_stream("Book") as stream:
        if view == "events":
            events = [event async for event in stream.stream_events()]
            assert (
                "".join(event.delta for event in events if event.type == "text_delta")
                == "Which date?"
            )
        elif view == "text":
            assert "".join([text async for text in stream.stream_text(delta=True)]) == "Which date?"
        first = await stream.result()
    if view == "events":
        text = next(event for event in events if event.type == "text_delta")
        assert text.agent_name == "booking" and text.frame_id == first.state.active_frame_id.value
        assert text.invocation_id and all(
            a.sequence < b.sequence for a, b in zip(events, events[1:], strict=False)
        )
    second = await TextSession(agent=parent).run("Friday", state=restore(first.state))
    root = history(second.state.frames[second.state.root_frame_id])
    assert sum(item.type == "message" and item.text_content == "Which date?" for item in root) == (
        view != "result"
    )
    assert any(item.type == "message" and item.text_content == "Friday" for item in root)


async def test_nested_child_merges_text_delivered_before_its_own_child_started() -> None:
    leaf = Agent(
        name="leaf", instructions="Help", llm=Script(Reply("Leaf result")), output_type=str
    )
    preface = call("leaf", {}, "leaf-call")
    preface.text = "Checking with my helper"
    middle = Agent(
        name="middle",
        instructions="Help",
        llm=Script(preface, Reply("Middle result")),
        output_type=str,
        tools=[leaf.as_tool(tool_name="leaf")],
    )
    root = Agent(
        name="root",
        instructions="Help",
        llm=Script(call("middle", {}, "middle-call"), Reply("Done")),
        tools=[middle.as_tool(tool_name="middle", merge_chat_history=True)],
    )
    usage_before_request: list[int] = []
    accumulators = []

    @root.instructions
    def usage(ctx: RunContext) -> str:
        usage_before_request.append(ctx.usage.requests)
        accumulators.append(ctx.usage)
        return "Help"

    turn = await TextSession(agent=root).run("Start")
    assert usage_before_request == [0, 4]
    # a snapshot never changes underneath its reader; the turn counts every frame once
    assert accumulators[-1].requests == 4 and turn.result.usage.requests == 5
    root_history = history(turn.state.frames[turn.state.root_frame_id])
    assert any(
        item.type == "message" and item.text_content == "Checking with my helper"
        for item in root_history
    )
    assert not any(
        item.type == "message" and item.text_content == "Leaf result" for item in root_history
    )


async def test_a_childs_accepted_json_null_resolves_the_parent_call() -> None:
    child = Agent(
        name="child",
        instructions="Finish",
        llm=Script(call("final_result", {"result": None}, "final")),
        output_type=type(None),
    )
    parent = Agent(
        name="parent",
        instructions="Help",
        llm=Script(call("child", {}, "child-call"), Reply("Done")),
        tools=[child.as_tool(tool_name="child")],
    )
    turn = await TextSession(agent=parent).run("Start")
    output = next(
        item
        for item in turn.result.all_messages()
        if item.type == "function_call_output" and item.call_id == "child-call"
    )
    assert json.loads(output.output) is None
    assert turn.result.output == "Done"


async def test_session_preserves_the_original_provider_failure() -> None:
    class Denied(Script):
        def chat(self, *, chat_ctx: ChatContext, **kwargs: Any) -> LLMStream:
            raise APIStatusError("denied", status_code=401, retryable=False)

    agent = Agent(name="desk", instructions="Help", llm=Denied())
    with pytest.raises(APIStatusError) as failure:
        await TextSession(agent=agent).run("Start")
    assert failure.value.status_code == 401


async def test_managed_checkpoint_yields_before_starting_the_child_model() -> None:
    from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

    calls = 0

    async def checkpoint() -> bool:
        nonlocal calls
        calls += 1
        return calls == 2

    child_model = Script()
    child = Agent(name="child", instructions="Help", llm=child_model, output_type=str)
    root = Agent(
        name="root",
        instructions="Help",
        llm=Script(call("child", {}, "child")),
        tools=[child.as_tool(tool_name="child")],
    )
    hooks = RunHooks(checkpoint=checkpoint)
    token = set_run_hooks(hooks)
    try:
        turn = await TextSession(agent=root).run("Start")
        assert hooks.interrupted
        assert calls == 2 and not child_model.requests
        active = turn.state.frames[turn.state.active_frame_id]
        assert active.agent_name == "child" and not active.completed
        assert active.parent is not None
        parent = turn.state.frames[active.parent.frame_id]
        assert not any(
            item.type == "function_call_output" and item.call_id == active.parent.call_id
            for item in history(parent)
        )
    finally:
        reset_run_hooks(token)
