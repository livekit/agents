"""Between the in-memory model and A2A, in both directions.

:func:`to_a2a_request` / :func:`from_a2a_request` are inverses, and so are
:func:`to_a2a_events` / :func:`from_a2a_events`. Nothing outside this package names an A2A
type, and nothing here invents an item format: a chat item travels as the JSON
``ChatContext`` already serializes itself to.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Any

from ..llm.chat_context import ChatContext, ChatItem
from ..utils import shortuuid
from ..voice.served_request import Directive
from .extension import (
    ANSWER_ARTIFACT_NAME,
    CALLER,
    CONVERSATION,
    DIRECTIVE,
    KIND,
    KIND_CHAT_CTX,
    KIND_CHAT_ITEM,
    KIND_CLOSE,
    KIND_DELEGATION,
    VERBATIM,
    as_dict,
    pb,
    struct,
    text_of,
    value,
)
from .types import TaskInput, TaskState, TaskUpdate

_STATE_FROM_A2A: dict[Any, TaskState] = {
    pb.TaskState.TASK_STATE_SUBMITTED: "working",
    pb.TaskState.TASK_STATE_WORKING: "working",
    pb.TaskState.TASK_STATE_COMPLETED: "completed",
    pb.TaskState.TASK_STATE_CANCELED: "canceled",
    pb.TaskState.TASK_STATE_FAILED: "failed",
    # a server that declines, or one that wants the caller to authenticate first, could not
    # answer this caller: nothing resumes a task
    pb.TaskState.TASK_STATE_REJECTED: "failed",
    pb.TaskState.TASK_STATE_AUTH_REQUIRED: "failed",
    pb.TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
}

_STATE_TO_A2A: dict[TaskState, Any] = {
    "working": pb.TaskState.TASK_STATE_WORKING,
    "completed": pb.TaskState.TASK_STATE_COMPLETED,
    "canceled": pb.TaskState.TASK_STATE_CANCELED,
    "failed": pb.TaskState.TASK_STATE_FAILED,
    "input-required": pb.TaskState.TASK_STATE_INPUT_REQUIRED,
}

# where the answer rides: an artifact for what the task produced, the terminal status message
# for what it has to say about ending instead
_ANSWER_IN_ARTIFACT: frozenset[TaskState] = frozenset({"completed", "input-required"})


def encode_ctx(chat_ctx: ChatContext) -> dict[str, Any]:
    """The chat history as JSON, with timestamps so a receiver renders history in its own
    order, and without images or audio, which a history carries by the megabyte."""
    return chat_ctx.to_dict(exclude_timestamp=False)


def encode_item(item: ChatItem) -> dict[str, Any]:
    ctx = ChatContext.empty()
    ctx.insert(item)
    return encode_ctx(ctx)["items"][0]  # type: ignore[no-any-return]


def decode_item(data: dict[str, Any]) -> ChatItem | None:
    items = ChatContext.from_dict({"items": [data]}).items
    return items[0] if items else None


def to_a2a_request(
    task_input: TaskInput,
    *,
    context_id: str,
    reference_task_ids: Sequence[str] = (),
) -> pb.SendMessageRequest:
    """One message on a context: the text, the history, and what it may be answering."""
    parts = [pb.Part(text=task_input.body)]
    if task_input.chat_ctx.items:
        parts.append(
            pb.Part(
                data=value(encode_ctx(task_input.chat_ctx)),
                metadata=struct({KIND: KIND_CHAT_CTX}),
            )
        )

    message = pb.Message(
        message_id=shortuuid("msg-"),
        context_id=context_id,
        role=pb.Role.ROLE_USER,
        parts=parts,
        reference_task_ids=list(reference_task_ids),
    )
    metadata: dict[str, Any] = {}
    if task_input.closing:
        metadata[KIND] = KIND_CLOSE
    elif task_input.is_delegation:
        metadata[KIND] = KIND_DELEGATION
    if task_input.conversation_id:
        metadata[CONVERSATION] = task_input.conversation_id
    if task_input.caller_session_id:
        metadata[CALLER] = task_input.caller_session_id
    if metadata:
        message.metadata.CopyFrom(struct(metadata))

    return pb.SendMessageRequest(
        message=message,
        metadata=struct(dict(task_input.metadata)),
        configuration=pb.SendMessageConfiguration(accepted_output_modes=["text/plain"]),
    )


def from_a2a_request(request: pb.SendMessageRequest) -> TaskInput:
    """The input an incoming request carries — the inverse of :func:`to_a2a_request`.

    A client that sends text and nothing else is a person's turn with an empty history,
    which is what makes a plain A2A client usable against a LiveKit endpoint.
    """
    message = request.message
    chat_ctx = ChatContext.empty()
    for part in message.parts:
        if part.WhichOneof("content") != "data":
            continue
        if as_dict(part.metadata).get(KIND) != KIND_CHAT_CTX:
            continue
        chat_ctx = ChatContext.from_dict(as_dict(part.data))

    body = text_of(message.parts)
    message_metadata = as_dict(message.metadata)
    kind = message_metadata.get(KIND)
    delegation = kind == KIND_DELEGATION
    return TaskInput(
        text=None if delegation else body,
        instruction=body if delegation else None,
        chat_ctx=chat_ctx,
        metadata=as_dict(request.metadata),
        closing=kind == KIND_CLOSE,
        conversation_id=message_metadata.get(CONVERSATION),
        caller_session_id=message_metadata.get(CALLER),
    )


def to_a2a_events(update: TaskUpdate, *, task_id: str, context_id: str) -> list[Any]:
    """One update as the A2A events that carry it.

    A terminal state with something to hand over is two events, because the protocol keeps
    the deliverable and the lifecycle apart: the artifact, then the status that ends the task.
    """
    status = pb.TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=pb.TaskStatus(state=_STATE_TO_A2A[update.state]),
    )

    if update.state == "completed" and update.directive is not None:
        status.metadata.CopyFrom(
            struct({DIRECTIVE: {"kind": update.directive.kind, "reason": update.directive.reason}})
        )

    # the answer rides an artifact; working reports as it goes, and failed and canceled say
    # why in the status message
    answer = update.text if update.state in _ANSWER_IN_ARTIFACT else ""

    parts: list[pb.Part] = []
    if update.text and not answer:
        parts.append(pb.Part(text=update.text))
    if update.item is not None:
        parts.append(
            pb.Part(
                data=value(encode_item(update.item)),
                metadata=struct({KIND: KIND_CHAT_ITEM}),
            )
        )
    if parts:
        message = pb.Message(
            message_id=shortuuid("msg-"),
            task_id=task_id,
            context_id=context_id,
            role=pb.Role.ROLE_AGENT,
            parts=parts,
        )
        if update.verbatim and not answer:
            message.metadata.CopyFrom(struct({VERBATIM: True}))
        status.status.message.CopyFrom(message)

    if not answer:
        # a task may conclude with nothing to add, the way a tool may return None
        return [status]

    artifact = pb.TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=pb.Artifact(
            artifact_id=shortuuid("art-"),
            name=ANSWER_ARTIFACT_NAME,
            parts=[pb.Part(text=answer)],
        ),
        last_chunk=True,
    )
    if update.verbatim:
        artifact.artifact.metadata.CopyFrom(struct({VERBATIM: True}))
    return [artifact, status]


async def from_a2a_events(events: AsyncIterable[Any]) -> AsyncIterator[TaskUpdate]:
    """A task's event stream as updates — the inverse of :func:`to_a2a_events`.

    Artifacts accumulate until the terminal status, so the deliverable and the lifecycle
    reach the caller as one update; a stream that ends without one ends this iterator early.
    """
    # keyed per artifact id: a server may stream several at once and `append` names the one it
    # extends, so keying on nothing would concatenate one artifact's chunks onto another's
    bodies: dict[str, str] = {}
    names: dict[str, str] = {}
    verbatim_answer = False

    async for event in events:
        if isinstance(event, pb.StreamResponse):
            event = getattr(event, event.WhichOneof("payload"))

        if isinstance(event, pb.Message):
            # a server that replies without opening a task has one output, so it is the answer
            yield TaskUpdate(state="completed", text=text_of(event.parts))
            return

        if isinstance(event, pb.TaskArtifactUpdateEvent):
            artifact = event.artifact
            chunk = text_of(artifact.parts)
            bodies[artifact.artifact_id] = (
                bodies.get(artifact.artifact_id, "") + chunk if event.append else chunk
            )
            if artifact.name:
                names[artifact.artifact_id] = artifact.name
            if as_dict(artifact.metadata).get(VERBATIM):
                verbatim_answer = True
            continue

        if not isinstance(event, (pb.Task, pb.TaskStatusUpdateEvent)):
            continue

        message = event.status.message
        state = _STATE_FROM_A2A.get(event.status.state, "working")
        text = text_of(message.parts)
        item = next(
            (
                decode_item(as_dict(part.data))
                for part in message.parts
                if part.WhichOneof("content") == "data"
                and as_dict(part.metadata).get(KIND) == KIND_CHAT_ITEM
            ),
            None,
        )

        if state == "working":
            if text or item is not None:
                yield TaskUpdate(
                    text=text, item=item, verbatim=bool(as_dict(message.metadata).get(VERBATIM))
                )
            continue

        # where a server produced more than one artifact, the one named `answer` is the
        # answer, and otherwise they join in arrival order
        named = [body for a_id, body in bodies.items() if names.get(a_id) == ANSWER_ARTIFACT_NAME]
        answer = "\n".join(named or bodies.values())
        raw_directive = as_dict(event.metadata).get(DIRECTIVE) if state == "completed" else None
        yield TaskUpdate(
            state=state,
            text=answer or text,
            item=item,
            verbatim=verbatim_answer if answer else bool(as_dict(message.metadata).get(VERBATIM)),
            directive=Directive(kind=raw_directive["kind"], reason=raw_directive.get("reason", ""))
            if raw_directive
            else None,
        )
        return
