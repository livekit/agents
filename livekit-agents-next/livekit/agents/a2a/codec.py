"""Between the storage layer's JSON and the A2A wire, in both directions.

Both sides speak A2A v1. A stored message, task history entry or artifact is its v1 protobuf
JSON with the extension's metadata keys short (``livekit/``); on the wire the keys are the
extension's own, written per request when the echoed ``A2A-Extensions`` activated them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from google.protobuf import json_format, timestamp_pb2
from pydantic import JsonValue

from .._generated.storage_run import TaskRow
from ..llm.chat_context import ChatContext
from ..utils import shortuuid
from .extension import (
    ANSWER_ARTIFACT_NAME,
    KIND,
    KIND_CHAT_CTX,
    Extensions,
    Metadata,
    as_dict,
    pb,
    shorten,
    widen,
)

StoredMessage = dict[str, Any]
"""A message as the storage layer holds it: v1 protobuf JSON, metadata under short keys."""

StoredPart = dict[str, Any]
"""One part of a stored message, v1 protobuf JSON."""

StreamEvent = pb.Task | pb.Message | pb.TaskStatusUpdateEvent | pb.TaskArtifactUpdateEvent
"""What a server yields on ``message/stream``, before it is wrapped in a ``StreamResponse``."""

_WithMetadata = pb.Message | pb.Artifact | pb.Task | pb.TaskStatusUpdateEvent


_STATE_TO_PB: dict[str, pb.TaskState] = {
    "submitted": pb.TaskState.TASK_STATE_SUBMITTED,
    "working": pb.TaskState.TASK_STATE_WORKING,
    "input-required": pb.TaskState.TASK_STATE_INPUT_REQUIRED,
    "completed": pb.TaskState.TASK_STATE_COMPLETED,
    "failed": pb.TaskState.TASK_STATE_FAILED,
    "canceled": pb.TaskState.TASK_STATE_CANCELED,
    "rejected": pb.TaskState.TASK_STATE_REJECTED,
}
"""The framework's task states, as A2A's."""
_STATE_FROM_PB: dict[pb.TaskState, str] = {v: k for k, v in _STATE_TO_PB.items()}
_STATE_FROM_PB[pb.TaskState.TASK_STATE_AUTH_REQUIRED] = "auth-required"


def state_to_pb(status: str) -> pb.TaskState:
    return _STATE_TO_PB.get(status, pb.TaskState.TASK_STATE_WORKING)


def state_from_pb(state: pb.TaskState) -> str:
    return _STATE_FROM_PB.get(state, "working")


# -- metadata --


def metadata_from_wire(raw: object) -> Metadata:
    """Wire keys to short keys; unknown keys kept as they are."""
    if not isinstance(raw, Mapping):
        return {}
    return {shorten(str(k)): v for k, v in raw.items()}


def metadata_to_wire(short: Mapping[str, JsonValue] | None) -> Metadata:
    """Short keys to wire keys. Every key travels either way: ``metadata`` is core A2A."""
    return {widen(key): item for key, item in (short or {}).items()}


def _active(exts: Extensions | None) -> bool:
    """Whether the extension's own keys travel: yes when no profile was negotiated."""
    return exts is None or exts.livekit


def _meta(message: _WithMetadata) -> Metadata:
    """The ``metadata`` of a proto message, short keys."""
    return metadata_from_wire(as_dict(message.metadata))


def _rekey(obj: Mapping[str, Any], keys: Any) -> dict[str, Any]:
    """``obj`` with its ``metadata`` and its parts' ``metadata`` keys mapped by ``keys``."""
    out = dict(obj)
    if isinstance(out.get("metadata"), Mapping):
        out["metadata"] = keys(out["metadata"])
    if isinstance(out.get("parts"), list):
        out["parts"] = [_rekey(p, keys) if isinstance(p, Mapping) else p for p in out["parts"]]
    return out


# -- parts and messages --


def message_from_wire(raw: Mapping[str, Any]) -> StoredMessage:
    """An A2A v1 ``Message`` from the wire as the storage layer's dict."""
    try:
        msg = json_format.ParseDict(dict(raw), pb.Message(), ignore_unknown_fields=True)
    except json_format.ParseError as e:
        raise ValueError(f"an A2A v1 message: {e}") from e
    return _rekey(json_format.MessageToDict(msg), metadata_from_wire)


def message_to_pb(
    msg: StoredMessage,
    *,
    exts: Extensions | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
) -> pb.Message:
    wire = _rekey(msg, metadata_to_wire)
    if not _active(exts):
        wire["parts"] = [p for p in wire.get("parts") or [] if not _tagged(p, KIND_CHAT_CTX)]
    out = json_format.ParseDict(wire, pb.Message(), ignore_unknown_fields=True)
    out.message_id = out.message_id or shortuuid("msg_")
    out.context_id = context_id or out.context_id
    out.task_id = task_id or out.task_id
    return out


def text_of(msg: StoredMessage | None) -> str:
    """The text parts of a stored message, joined."""
    if not msg:
        return ""
    return "\n".join(str(p["text"]) for p in msg.get("parts") or [] if "text" in p)


def _text(parts: Iterable[pb.Part]) -> str:
    """The text parts of a proto message or artifact, joined."""
    return "\n".join(part.text for part in parts if part.WhichOneof("content") == "text")


# -- the conversation as a data part --


def chat_ctx_part(chat_ctx: ChatContext) -> StoredPart:
    """The caller's conversation as a ``chat_ctx`` data part (a delegation outside the session)."""
    return {"data": chat_ctx.to_dict(exclude_timestamp=False), "metadata": {KIND: KIND_CHAT_CTX}}


def _tagged(part: Mapping[str, Any], kind: str) -> bool:
    """Whether the part's metadata carries ``livekit/kind: <kind>`` (short or wire key)."""
    metadata = part.get("metadata") or {}
    return kind in (metadata.get(KIND), metadata.get(widen(KIND)))


# -- the task object and the events the server sends --


def _timestamp(ms: int | None) -> timestamp_pb2.Timestamp:
    """Unix milliseconds, now when ``None``."""
    stamp = timestamp_pb2.Timestamp()
    stamp.FromMilliseconds(ms if ms else int(time.time() * 1000))
    return stamp


def _artifact_to_pb(artifact: Mapping[str, Any], exts: Extensions | None) -> pb.Artifact:
    wire = _rekey(artifact, metadata_to_wire)
    if not _active(exts):
        wire["parts"] = [p for p in wire.get("parts") or [] if not _tagged(p, KIND_CHAT_CTX)]
    out = json_format.ParseDict(wire, pb.Artifact(), ignore_unknown_fields=True)
    out.artifact_id = out.artifact_id or shortuuid("art_")
    out.name = out.name or ANSWER_ARTIFACT_NAME
    return out


def _status_message_to_pb(
    record: TaskRow, context_id: str, exts: Extensions | None
) -> pb.Message | None:
    if record.data.status_message is None:
        return None
    return message_to_pb(
        record.data.status_message, exts=exts, context_id=context_id, task_id=record.task_id
    )


def _history_window(record: TaskRow, history_length: int | None) -> list[StoredMessage]:
    """The last ``history_length`` messages, all of them when it is ``None``."""
    history = record.data.history
    if history_length is None or history_length < 0:
        return history
    if history_length == 0:
        return []
    return history[len(history) - history_length :]


def task_to_pb(
    record: TaskRow,
    context_id: str,
    *,
    exts: Extensions | None = None,
    history_length: int | None = None,
    include_artifacts: bool = True,
) -> pb.Task:
    """The task row as the A2A ``Task`` object: id, the thread as context, the status from its
    columns, then history, artifacts and metadata from ``data``."""
    task = pb.Task(
        id=record.task_id,
        context_id=context_id,
        status=pb.TaskStatus(
            state=state_to_pb(record.status), timestamp=_timestamp(record.updated_at)
        ),
    )

    status_message = _status_message_to_pb(record, context_id, exts)
    if status_message is not None:
        task.status.message.CopyFrom(status_message)

    for item in _history_window(record, history_length):
        task.history.append(
            message_to_pb(item, exts=exts, context_id=context_id, task_id=record.task_id)
        )

    if include_artifacts:
        task.artifacts.extend(_artifact_to_pb(a, exts) for a in record.data.artifacts or [])

    metadata = metadata_to_wire(record.data.metadata)
    if metadata:
        task.metadata.update(metadata)
    return task


def submitted_task(task_id: str, context_id: str) -> pb.Task:
    """The acknowledgment, first on every stream: the ids a client reads everything else by."""
    return pb.Task(
        id=task_id,
        context_id=context_id,
        status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_SUBMITTED, timestamp=_timestamp(None)),
    )


def working_event(
    task_id: str,
    context_id: str,
    text: str,
    *,
    exts: Extensions,
) -> pb.TaskStatusUpdateEvent:
    """``ctx.update(text)``: a ``working`` status with the progress as its message."""
    msg: StoredMessage = {"messageId": shortuuid("msg_"), "role": "ROLE_AGENT", "parts": []}
    if text:
        msg["parts"].append({"text": text})

    return pb.TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=pb.TaskStatus(
            state=pb.TaskState.TASK_STATE_WORKING,
            message=message_to_pb(msg, exts=exts, context_id=context_id, task_id=task_id),
            timestamp=_timestamp(None),
        ),
    )


def artifact_event(
    task_id: str,
    context_id: str,
    *,
    artifact_id: str,
    text: str,
    append: bool,
    last_chunk: bool,
    exts: Extensions,
) -> pb.TaskArtifactUpdateEvent:
    """One chunk of the answer artifact; the final chunk sets ``lastChunk``."""
    artifact = pb.Artifact(artifact_id=artifact_id, name=ANSWER_ARTIFACT_NAME)
    if text or not append:
        artifact.parts.append(pb.Part(text=text))

    return pb.TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=append,
        last_chunk=last_chunk,
    )


def final_event(record: TaskRow, context_id: str, *, exts: Extensions) -> pb.TaskStatusUpdateEvent:
    """The terminal status event, built from the committed row so a replay says the same."""
    event = pb.TaskStatusUpdateEvent(
        task_id=record.task_id,
        context_id=context_id,
        status=pb.TaskStatus(
            state=state_to_pb(record.status), timestamp=_timestamp(record.updated_at)
        ),
    )

    # the answer rides the artifact; the status message carries a question, an error, or a
    # merged task's note
    carries_message = record.status != "completed" or not record.data.artifacts
    status_message = _status_message_to_pb(record, context_id, exts) if carries_message else None
    if status_message is not None:
        event.status.message.CopyFrom(status_message)

    metadata = metadata_to_wire(record.data.metadata)
    if metadata:
        event.metadata.update(metadata)
    return event


def to_stream_response(event: StreamEvent | pb.StreamResponse) -> pb.StreamResponse:
    if isinstance(event, pb.StreamResponse):
        return event

    response = pb.StreamResponse()
    if isinstance(event, pb.Task):
        response.task.CopyFrom(event)
    elif isinstance(event, pb.Message):
        response.message.CopyFrom(event)
    elif isinstance(event, pb.TaskStatusUpdateEvent):
        response.status_update.CopyFrom(event)
    elif isinstance(event, pb.TaskArtifactUpdateEvent):
        response.artifact_update.CopyFrom(event)
    return response


def _unwrap(raw: StreamEvent | pb.StreamResponse) -> StreamEvent | None:
    """The event inside a ``StreamResponse``; a bare event as it is."""
    if not isinstance(raw, pb.StreamResponse):
        return raw

    if raw.HasField("task"):
        return raw.task
    if raw.HasField("message"):
        return raw.message
    if raw.HasField("status_update"):
        return raw.status_update
    if raw.HasField("artifact_update"):
        return raw.artifact_update
    return None


# -- reading a task's events, as a client: both profiles --


@dataclass
class TaskUpdate:
    """One event of a task as a caller reads it, whether or not the server wrote the extension."""

    state: str = "working"
    text: str = ""
    task_id: str = ""
    context_id: str = ""
    metadata: Metadata = field(default_factory=dict)


@dataclass
class _Answer:
    """The artifact chunks of one task, accumulated until its terminal status."""

    bodies: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)

    def collect(self, artifact: pb.Artifact, append: bool) -> None:
        chunk = _text(artifact.parts)
        if append:
            chunk = self.bodies.get(artifact.artifact_id, "") + chunk
        self.bodies[artifact.artifact_id] = chunk

        if artifact.name:
            self.names[artifact.artifact_id] = artifact.name

    def text(self) -> str:
        """The ``answer`` artifact; every artifact joined when none carries that name."""
        named = [
            body
            for artifact_id, body in self.bodies.items()
            if self.names.get(artifact_id) == ANSWER_ARTIFACT_NAME
        ]
        return "\n".join(named or self.bodies.values())


@dataclass
class _Ids:
    """The task and context ids, from the latest event that named them."""

    task_id: str = ""
    context_id: str = ""

    def note(self, event: pb.Task | pb.TaskStatusUpdateEvent | pb.TaskArtifactUpdateEvent) -> None:
        own = event.id if isinstance(event, pb.Task) else event.task_id
        self.task_id = own or self.task_id
        self.context_id = event.context_id or self.context_id


def _reply_update(message: pb.Message) -> TaskUpdate:
    """A bare ``Message`` reply: the answer of a server that opened no task."""
    return TaskUpdate(
        state="completed",
        text=_text(message.parts),
        task_id=message.task_id,
        context_id=message.context_id,
    )


def _status_update(
    event: pb.Task | pb.TaskStatusUpdateEvent, ids: _Ids, answer: _Answer
) -> TaskUpdate | None:
    """The update a status carries: the acknowledgment, a progress message, or the end."""
    state = state_from_pb(event.status.state)
    if state == "submitted" and isinstance(event, pb.Task):
        return TaskUpdate(state="submitted", task_id=ids.task_id, context_id=ids.context_id)

    metadata = _meta(event)
    text = _text(event.status.message.parts)

    if state in ("submitted", "working"):
        if not text:
            return None
        return TaskUpdate(
            state="working",
            text=text,
            task_id=ids.task_id,
            context_id=ids.context_id,
            metadata=metadata,
        )

    answer_text = answer.text()
    return TaskUpdate(
        state=state,
        text=answer_text or text,
        task_id=ids.task_id,
        context_id=ids.context_id,
        metadata=metadata,
    )


async def updates_from_events(
    events: AsyncIterable[StreamEvent | pb.StreamResponse],
) -> AsyncIterator[TaskUpdate]:
    """A task's event stream as updates, extension or no extension.

    Artifact chunks accumulate until the terminal status, so the answer and the lifecycle reach
    the caller as one terminal update; every ``working`` status with text is a progress update;
    a bare ``Message`` reply is the answer of a server that opened no task.
    """
    answer = _Answer()
    ids = _Ids()

    async for raw in events:
        event = _unwrap(raw)
        if event is None:
            continue

        if isinstance(event, pb.Message):
            yield _reply_update(event)
            return

        ids.note(event)
        if isinstance(event, pb.TaskArtifactUpdateEvent):
            answer.collect(event.artifact, event.append)
            continue

        if isinstance(event, pb.Task):
            for artifact in event.artifacts:
                answer.collect(artifact, append=False)

        update = _status_update(event, ids, answer)
        if update is None:
            continue

        yield update
        if update.state not in ("submitted", "working"):
            return


__all__ = [
    "StoredMessage",
    "StoredPart",
    "StreamEvent",
    "TaskUpdate",
    "artifact_event",
    "chat_ctx_part",
    "final_event",
    "message_from_wire",
    "message_to_pb",
    "metadata_from_wire",
    "metadata_to_wire",
    "state_from_pb",
    "state_to_pb",
    "submitted_task",
    "task_to_pb",
    "text_of",
    "to_stream_response",
    "updates_from_events",
    "working_event",
]
