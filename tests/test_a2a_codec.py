"""The A2A codec: the two pairs are inverses, and the wire carries what the extension says."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import a2a
from livekit.agents.a2a.extension import (
    ANSWER_ARTIFACT_NAME,
    DIRECTIVE,
    KIND,
    KIND_CHAT_CTX,
    KIND_CHAT_ITEM,
    KIND_DELEGATION,
    VERBATIM,
    as_dict,
    pb,
)
from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput

pytestmark = pytest.mark.unit


def test_the_extension_names_itself_and_its_keys() -> None:
    """A rename is a wire change, so it is spelled out here rather than derived."""
    assert a2a.EXTENSION_URI == "https://livekit.io/a2a/ext/agent-session/v1"
    assert KIND == "https://livekit.io/a2a/ext/agent-session/v1/kind"
    assert VERBATIM == "https://livekit.io/a2a/ext/agent-session/v1/verbatim"
    assert DIRECTIVE == "https://livekit.io/a2a/ext/agent-session/v1/directive"
    assert a2a.REASON == "https://livekit.io/a2a/ext/agent-session/v1/reason"


def test_a_card_offers_the_extension_without_requiring_it() -> None:
    card = a2a.agent_card(url="http://localhost:8080/fare-desk", name="fare-desk", description="d")
    (extension,) = card.capabilities.extensions
    assert extension.uri == a2a.EXTENSION_URI
    assert extension.required is False
    assert card.capabilities.streaming is True
    assert [skill.id for skill in card.skills] == ["delegate"]


def _conversation() -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="change my Monday flight", id="m1")
    ctx.add_message(
        role="assistant", content="One moment.", id="m2", extra={"message_source": "say"}
    )
    ctx.insert(
        FunctionCall(
            id="i1",
            call_id="c3_update_0",
            name="check_availability",
            arguments="{}",
            update_of="c3",
        )
    )
    ctx.insert(
        FunctionCallOutput(
            id="i2",
            call_id="c3_update_0",
            name="check_availability",
            output="looking",
            is_error=False,
        )
    )
    return ctx


INPUTS: list[a2a.TaskInput] = [
    a2a.TaskInput(text="hello"),
    a2a.TaskInput(text="what is the fee?", chat_ctx=_conversation()),
    a2a.TaskInput(instruction="find the change fee", chat_ctx=_conversation()),
    # a duplex model delegates without saying anything
    a2a.TaskInput(instruction="", chat_ctx=_conversation()),
    a2a.TaskInput(instruction="look it up", metadata={"customer_id": "c-42", "tier": 2}),
    a2a.TaskInput(text="multi\nline\nturn"),
]


@pytest.mark.parametrize("task_input", INPUTS, ids=lambda i: f"{i.is_delegation}-{i.body[:12]!r}")
def test_request_round_trip(task_input: a2a.TaskInput) -> None:
    back = a2a.from_a2a_request(a2a.to_a2a_request(task_input, context_id="sess-1"))

    assert back.text == task_input.text
    assert back.instruction == task_input.instruction
    assert back.is_delegation == task_input.is_delegation
    assert back.metadata == task_input.metadata
    assert back.chat_ctx.to_dict(exclude_timestamp=False) == task_input.chat_ctx.to_dict(
        exclude_timestamp=False
    )


def test_request_carries_the_extension_keys() -> None:
    request = a2a.to_a2a_request(
        a2a.TaskInput(instruction="find it", chat_ctx=_conversation()),
        context_id="sess-1",
        reference_task_ids=["task-open"],
    )

    assert request.message.context_id == "sess-1"
    assert request.message.message_id
    assert as_dict(request.message.metadata)[KIND] == KIND_DELEGATION
    assert list(request.message.reference_task_ids) == ["task-open"]

    text_parts = [p for p in request.message.parts if p.WhichOneof("content") == "text"]
    data_parts = [p for p in request.message.parts if p.WhichOneof("content") == "data"]
    assert [p.text for p in text_parts] == ["find it"]
    assert [as_dict(p.metadata)[KIND] for p in data_parts] == [KIND_CHAT_CTX]


def test_a_persons_turn_is_not_tagged_a_delegation() -> None:
    request = a2a.to_a2a_request(a2a.TaskInput(text="hello"), context_id="sess-1")
    assert KIND not in as_dict(request.message.metadata)
    # nothing to share, so no conversation part rides along
    assert all(p.WhichOneof("content") == "text" for p in request.message.parts)


def test_a_vanilla_request_reads_as_a_persons_turn() -> None:
    """What a stock A2A client sends: one text part and no metadata at all."""
    request = pb.SendMessageRequest(
        message=pb.Message(
            message_id="msg-1", role=pb.Role.ROLE_USER, parts=[pb.Part(text="what is the fee?")]
        )
    )
    task_input = a2a.from_a2a_request(request)

    assert task_input.text == "what is the fee?"
    assert task_input.instruction is None
    assert task_input.chat_ctx.items == []


def test_exactly_one_of_text_and_instruction() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        a2a.TaskInput()
    with pytest.raises(ValueError, match="exactly one"):
        a2a.TaskInput(text="a", instruction="b")


UPDATES: list[a2a.TaskUpdate] = [
    a2a.TaskUpdate(text="Checking Tuesday."),
    a2a.TaskUpdate(item=_conversation().items[2]),
    a2a.TaskUpdate(text="Checking Tuesday.", item=_conversation().items[2]),
    a2a.TaskUpdate(text="Your confirmation code is AB12.", verbatim=True),
    a2a.TaskUpdate(state="completed", text="The change fee is $75."),
    a2a.TaskUpdate(state="completed", text="Read this back exactly.", verbatim=True),
    a2a.TaskUpdate(state="completed", text=""),
    a2a.TaskUpdate(
        state="completed",
        text="Anything else?",
        directive=a2a.Directive(kind="end_session", reason="user_request"),
    ),
    a2a.TaskUpdate(state="failed", text="the fare service is unreachable"),
    a2a.TaskUpdate(state="canceled", text="the hold was released"),
    a2a.TaskUpdate(state="input-required", text="Which Tuesday flight?"),
    # the answer is an item as well as text, and the caller stores what it renders
    a2a.TaskUpdate(state="completed", text="The change fee is $75.", item=_conversation().items[1]),
    a2a.TaskUpdate(state="input-required", text="Which one?", item=_conversation().items[1]),
    a2a.TaskUpdate(state="canceled", text="the hold was released", item=_conversation().items[2]),
]


async def _emit(events: list[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


@pytest.mark.parametrize("update", UPDATES, ids=lambda u: f"{u.state}-{u.text[:14]!r}")
async def test_event_round_trip(update: a2a.TaskUpdate) -> None:
    events = a2a.to_a2a_events(update, task_id="task-1", context_id="sess-1")
    # a stream has to end somewhere: a working update is followed by the status that ends it
    if update.state == "working":
        events += a2a.to_a2a_events(
            a2a.TaskUpdate(state="completed"), task_id="task-1", context_id="sess-1"
        )

    out = [u async for u in a2a.from_a2a_events(_emit(events))]
    back = out[0]

    assert back.state == update.state
    assert back.text == update.text
    if update.item is not None:
        assert back.item is not None
        assert back.item.model_dump() == update.item.model_dump()
    else:
        assert back.item is None
    # nothing was said, so nothing is marked as said-as-written
    assert back.verbatim == (update.verbatim and bool(update.text))
    assert back.directive == update.directive


def test_the_answer_rides_an_artifact_and_the_reason_rides_the_status() -> None:
    completed = a2a.to_a2a_events(
        a2a.TaskUpdate(state="completed", text="The change fee is $75."),
        task_id="task-1",
        context_id="sess-1",
    )
    artifact, status = completed
    assert artifact.artifact.name == ANSWER_ARTIFACT_NAME
    assert artifact.artifact.parts[0].text == "The change fee is $75."
    assert artifact.last_chunk is True
    assert status.status.state == pb.TaskState.TASK_STATE_COMPLETED

    (failed,) = a2a.to_a2a_events(
        a2a.TaskUpdate(state="failed", text="unreachable"), task_id="task-1", context_id="sess-1"
    )
    assert failed.status.message.parts[0].text == "unreachable"


def test_a_working_event_carries_the_item_and_the_relayed_text() -> None:
    call = _conversation().items[2]
    (status,) = a2a.to_a2a_events(
        a2a.TaskUpdate(text="Checking Tuesday.", item=call),
        task_id="task-1",
        context_id="sess-1",
    )

    assert status.status.state == pb.TaskState.TASK_STATE_WORKING
    parts = status.status.message.parts
    assert parts[0].text == "Checking Tuesday."
    assert as_dict(parts[1].metadata)[KIND] == KIND_CHAT_ITEM
    # the report names the call it reports for, so a reader need not parse the id
    assert as_dict(parts[1].data)["update_of"] == "c3"


def test_verbatim_is_marked_on_both_carriers() -> None:
    (status,) = a2a.to_a2a_events(
        a2a.TaskUpdate(text="AB12", verbatim=True), task_id="t", context_id="s"
    )
    assert as_dict(status.status.message.metadata)[VERBATIM] is True

    artifact, _ = a2a.to_a2a_events(
        a2a.TaskUpdate(state="completed", text="AB12", verbatim=True), task_id="t", context_id="s"
    )
    assert as_dict(artifact.artifact.metadata)[VERBATIM] is True


def test_a_directive_rides_only_a_completed_task() -> None:
    directive = a2a.Directive(kind="escalate", reason="policy_exception")
    _, status = a2a.to_a2a_events(
        a2a.TaskUpdate(state="completed", text="done", directive=directive),
        task_id="t",
        context_id="s",
    )
    assert as_dict(status.metadata)[DIRECTIVE] == {
        "kind": "escalate",
        "reason": "policy_exception",
    }

    (failed,) = a2a.to_a2a_events(
        a2a.TaskUpdate(state="failed", text="no", directive=directive), task_id="t", context_id="s"
    )
    assert DIRECTIVE not in as_dict(failed.metadata)


async def test_a_progress_report_reaches_the_caller_before_the_answer() -> None:
    events = [
        *a2a.to_a2a_events(a2a.TaskUpdate(text="Checking."), task_id="t", context_id="s"),
        *a2a.to_a2a_events(
            a2a.TaskUpdate(state="completed", text="The fee is $75."), task_id="t", context_id="s"
        ),
    ]
    out = [u async for u in a2a.from_a2a_events(_emit(events))]
    assert [(u.state, u.text) for u in out] == [
        ("working", "Checking."),
        ("completed", "The fee is $75."),
    ]


async def test_nothing_is_read_after_the_terminal_status() -> None:
    events = [
        *a2a.to_a2a_events(
            a2a.TaskUpdate(state="completed", text="done"), task_id="t", context_id="s"
        ),
        *a2a.to_a2a_events(a2a.TaskUpdate(text="too late"), task_id="t", context_id="s"),
    ]
    out = [u async for u in a2a.from_a2a_events(_emit(events))]
    assert [u.state for u in out] == ["completed"]


async def test_the_submitted_acknowledgment_says_nothing_of_its_own() -> None:
    """The first event carries the server's task id, not an update to relay."""
    events = [
        pb.Task(
            id="task-9",
            context_id="s",
            status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED),
        ),
        *a2a.to_a2a_events(
            a2a.TaskUpdate(state="completed", text="done"), task_id="task-9", context_id="s"
        ),
    ]
    out = [u async for u in a2a.from_a2a_events(_emit(events))]
    assert [(u.state, u.text) for u in out] == [("completed", "done")]


async def test_a_reply_that_opens_no_task_is_the_answer() -> None:
    """A vanilla A2A server may answer with a Message and never open a Task."""
    message = pb.Message(
        message_id="msg-1", role=pb.Role.ROLE_AGENT, parts=[pb.Part(text="the fee is $75")]
    )
    out = [u async for u in a2a.from_a2a_events(_emit([message]))]
    assert [(u.state, u.text) for u in out] == [("completed", "the fee is $75")]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (pb.TaskState.TASK_STATE_REJECTED, "failed"),
        (pb.TaskState.TASK_STATE_AUTH_REQUIRED, "input-required"),
    ],
)
async def test_foreign_states_map_to_what_the_caller_can_act_on(state: Any, expected: str) -> None:
    event = pb.TaskStatusUpdateEvent(
        task_id="t",
        context_id="s",
        status=pb.TaskStatus(
            state=state,
            message=pb.Message(
                message_id="m", role=pb.Role.ROLE_AGENT, parts=[pb.Part(text="no can do")]
            ),
        ),
    )
    out = [u async for u in a2a.from_a2a_events(_emit([event]))]
    assert [(u.state, u.text) for u in out] == [(expected, "no can do")]


async def test_streamed_artifact_chunks_join_per_artifact() -> None:
    def chunk(artifact_id: str, text: str, *, append: bool, name: str = "") -> Any:
        return pb.TaskArtifactUpdateEvent(
            task_id="t",
            context_id="s",
            artifact=pb.Artifact(artifact_id=artifact_id, name=name, parts=[pb.Part(text=text)]),
            append=append,
        )

    events = [
        chunk("a1", "The fee ", append=False, name=ANSWER_ARTIFACT_NAME),
        chunk("a2", "internal note", append=False, name="notes"),
        chunk("a1", "is $75.", append=True),
        *a2a.to_a2a_events(a2a.TaskUpdate(state="completed"), task_id="t", context_id="s"),
    ]
    out = [u async for u in a2a.from_a2a_events(_emit(events))]
    # only the artifact named `answer` is the answer
    assert [(u.state, u.text) for u in out] == [("completed", "The fee is $75.")]
