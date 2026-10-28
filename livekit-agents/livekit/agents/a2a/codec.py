"""Between the storage layer's JSON and the A2A wire, in both directions.

Inside the worker a message is the dict the storage layer takes (``messageId``, ``role``,
``parts`` tagged with ``kind``, ``metadata`` under short keys). On the wire it is the A2A 1.0
JSON of the protobuf types; :func:`message_from_wire` accepts the 1.0 shape and the 0.3 shape
alike. The server writes the extension's keys (``livekit/``) per request, when the echoed
``A2A-Extensions`` activated them; the client side reads them when a server wrote them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from google.protobuf import timestamp_pb2
from pydantic import JsonValue

from .._generated.storage_run import TaskRecord
from ..llm.chat_context import ChatContext
from ..utils import shortuuid
from .extension import (
    ANSWER_ARTIFACT_NAME,
    KIND,
    KIND_CHAT_CTX,
    VERBATIM,
    Extensions,
    Metadata,
    as_dict,
    pb,
    shorten,
    struct,
    value,
    widen,
)

StoredMessage = dict[str, Any]  # typed by the _core stub: TaskRecord.status_message, history items
"""A message as the storage layer holds it; the codec reads and builds it, the stub owns it."""

StoredPart = dict[str, Any]  # typed by the _core stub: one entry of a stored message's parts
"""One part of a stored message, tagged with ``kind``."""

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
_STATE_FROM_PB: dict[pb.TaskState, str] = {v: k for k, v in _STATE_TO_PB.items()}
_STATE_FROM_PB[pb.TaskState.TASK_STATE_AUTH_REQUIRED] = "auth-required"
_STATE_FROM_PB[pb.TaskState.TASK_STATE_UNSPECIFIED] = "working"


def state_to_pb(state: str) -> pb.TaskState:
    return _STATE_TO_PB.get(state, pb.TaskState.TASK_STATE_WORKING)


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


# -- parts and messages --


def _part_from_wire(raw: Mapping[str, Any]) -> StoredPart:
    """One part, 1.0 or 0.3 shaped, to the internal ``kind``-tagged dict."""
    kind = raw.get("kind")
    out: StoredPart
    if "text" in raw and kind in (None, "text"):
        out = {"kind": "text", "text": str(raw.get("text") or "")}
    elif "data" in raw and kind in (None, "data"):
        out = {"kind": "data", "data": raw.get("data")}
    elif kind == "file" or "url" in raw or "raw" in raw or "file" in raw:
        out = {"kind": "file", "file": _file_from_wire(raw)}
    else:
        out = {"kind": "text", "text": str(raw.get("text") or "")}

    metadata = metadata_from_wire(raw.get("metadata"))
    if metadata:
        out["metadata"] = metadata
    return out


def _file_from_wire(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The 1.0 flat file fields over the 0.3 ``file`` object, the 0.3 names kept inside."""
    file_info: dict[str, Any] = dict(raw.get("file") or {})

    renamed = (("url", "uri"), ("raw", "bytes"), ("mediaType", "mimeType"), ("filename", "name"))
    for wire_name, stored_name in renamed:
        if raw.get(wire_name) is not None:
            file_info[stored_name] = raw[wire_name]
    return file_info


def _part_to_pb(part: StoredPart, exts: Extensions | None) -> pb.Part:
    kind = part.get("kind", "text")
    out: pb.Part
    if kind == "data":
        out = pb.Part(data=value(part.get("data")))
    elif kind == "file":
        out = _file_to_pb(part.get("file") or {})
    else:
        out = pb.Part(text=str(part.get("text") or ""))

    metadata = metadata_to_wire(part.get("metadata"))
    if metadata:
        out.metadata.CopyFrom(struct(metadata))
    return out


def _file_to_pb(file_info: Mapping[str, Any]) -> pb.Part:
    out = pb.Part()
    if file_info.get("uri"):
        out.url = str(file_info["uri"])
    elif file_info.get("bytes"):
        raw = file_info["bytes"]
        out.raw = raw if isinstance(raw, bytes) else str(raw).encode()

    if file_info.get("mimeType"):
        out.media_type = str(file_info["mimeType"])
    if file_info.get("name"):
        out.filename = str(file_info["name"])
    return out


def _parts_to_pb(parts: Iterable[StoredPart], exts: Extensions | None) -> list[pb.Part]:
    """The internal parts to the wire; typed items ride only where the extension is active."""
    out: list[pb.Part] = []
    for part in parts:
        if _tagged(part, KIND_CHAT_CTX) and not _active(exts):
            continue
        out.append(_part_to_pb(part, exts))
    return out


def message_from_wire(raw: Mapping[str, Any]) -> StoredMessage:
    """An A2A ``Message`` (1.0 or 0.3 JSON) as the storage layer's dict."""
    agent = raw.get("role") in ("ROLE_AGENT", "agent", "assistant", pb.Role.ROLE_AGENT)
    parts = raw.get("parts") or []
    out: StoredMessage = {
        "messageId": str(raw.get("messageId") or raw.get("message_id") or ""),
        "role": "agent" if agent else "user",
        "parts": [_part_from_wire(part) for part in parts if isinstance(part, Mapping)],
    }

    metadata = metadata_from_wire(raw.get("metadata"))
    if metadata:
        out["metadata"] = metadata

    context_id = raw.get("context_id") or raw.get("contextId")
    if context_id:
        out["contextId"] = str(context_id)
    task_id = raw.get("task_id") or raw.get("taskId")
    if task_id:
        out["taskId"] = str(task_id)

    refs = raw.get("referenceTaskIds") or raw.get("reference_task_ids")
    if refs:
        out["referenceTaskIds"] = [str(ref) for ref in refs]
    return out


def message_to_pb(
    msg: StoredMessage,
    *,
    exts: Extensions | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
) -> pb.Message:
    agent = str(msg.get("role") or "user") in ("agent", "assistant", "ROLE_AGENT")
    out = pb.Message(
        message_id=str(msg.get("messageId") or shortuuid("msg_")),
        role=pb.Role.ROLE_AGENT if agent else pb.Role.ROLE_USER,
        parts=_parts_to_pb(list(msg.get("parts") or []), exts),
        context_id=str(context_id or msg.get("contextId") or ""),
        task_id=str(task_id or msg.get("taskId") or ""),
    )

    if msg.get("referenceTaskIds"):
        out.reference_task_ids.extend(str(ref) for ref in msg["referenceTaskIds"])

    metadata = metadata_to_wire(msg.get("metadata"))
    if metadata:
        out.metadata.CopyFrom(struct(metadata))
    return out


def text_of(msg: StoredMessage | None) -> str:
    """The text parts of a storage-shaped message, joined."""
    if not msg:
        return ""

    parts = msg.get("parts") or []
    texts = [str(part.get("text") or "") for part in parts if part.get("kind", "text") == "text"]
    return "\n".join(texts)


def _text(parts: Iterable[pb.Part]) -> str:
    """The text parts of a proto message or artifact, joined."""
    return "\n".join(part.text for part in parts if part.WhichOneof("content") == "text")


def _mark_verbatim(part: pb.Part) -> None:
    """Flags ``part`` to reach the person exactly as written."""
    part.metadata.CopyFrom(struct(metadata_to_wire({VERBATIM: True})))


def _all_verbatim(parts: Iterable[pb.Part]) -> bool:
    """Whether the parts hold text and every text part is flagged ``livekit/verbatim``."""
    texts = [part for part in parts if part.WhichOneof("content") == "text"]
    return bool(texts) and all(
        metadata_from_wire(as_dict(part.metadata)).get(VERBATIM) is True for part in texts
    )


# -- the conversation as a data part --


def chat_ctx_part(chat_ctx: ChatContext) -> StoredPart:
    """The caller's conversation as a ``chat_ctx`` data part (a delegation outside the session)."""
    return {
        "kind": "data",
        "data": chat_ctx.to_dict(exclude_timestamp=False),
        "metadata": {KIND: KIND_CHAT_CTX},
    }


def _tagged(part: StoredPart, kind: str) -> bool:
    """Whether the part's metadata carries ``livekit/kind: <kind>``."""
    return (part.get("metadata") or {}).get(KIND) == kind


# -- the task object and the events the server sends --


def _timestamp(seconds: int | float | None) -> timestamp_pb2.Timestamp:
    stamp = timestamp_pb2.Timestamp()
    stamp.FromMilliseconds(int((seconds if seconds else time.time()) * 1000))
    return stamp


def _artifact_to_pb(artifact: Mapping[str, Any], exts: Extensions | None) -> pb.Artifact:
    out = pb.Artifact(
        artifact_id=str(artifact.get("artifactId") or shortuuid("art_")),
        name=str(artifact.get("name") or ANSWER_ARTIFACT_NAME),
        parts=_parts_to_pb(list(artifact.get("parts") or []), exts),
    )

    metadata = metadata_to_wire(artifact.get("metadata"))
    if metadata:
        out.metadata.CopyFrom(struct(metadata))
    return out


def _status_message_to_pb(
    record: TaskRecord, context_id: str, exts: Extensions | None
) -> pb.Message | None:
    if not isinstance(record.status_message, dict):
        return None
    return message_to_pb(record.status_message, exts=exts, context_id=context_id, task_id=record.id)


def _history_window(record: TaskRecord, history_length: int | None) -> list[StoredMessage]:
    """The last ``history_length`` messages, all of them when it is ``None``."""
    history = [item for item in record.data.history if isinstance(item, dict)]
    if history_length is None or history_length < 0:
        return history
    if history_length == 0:
        return []
    return history[len(history) - history_length :]


def task_to_pb(
    record: TaskRecord,
    context_id: str,
    *,
    exts: Extensions | None = None,
    history_length: int | None = None,
    include_artifacts: bool = True,
) -> pb.Task:
    """The task row as the A2A ``Task`` object: id, the thread as context, the status from its
    columns, then history, artifacts and metadata from ``data``."""
    task = pb.Task(
        id=record.id,
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
            message_to_pb(item, exts=exts, context_id=context_id, task_id=record.id)
        )

    if include_artifacts:
        artifacts = [a for a in record.data.artifacts or [] if isinstance(a, Mapping)]
        task.artifacts.extend(_artifact_to_pb(a, exts) for a in artifacts)

    metadata = metadata_to_wire(record.data.metadata)
    if metadata:
        task.metadata.CopyFrom(struct(metadata))
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
    verbatim: bool = False,
) -> pb.TaskStatusUpdateEvent:
    """``ctx.update(text)``: a ``working`` status with the progress as its message; with
    ``verbatim``, ``ctx.say(text)``, its text part flagged."""
    msg: StoredMessage = {"messageId": shortuuid("msg_"), "role": "agent", "parts": []}
    if text:
        msg["parts"].append({"kind": "text", "text": text})

    message = message_to_pb(msg, exts=exts, context_id=context_id, task_id=task_id)
    if verbatim:
        for part in message.parts:
            _mark_verbatim(part)
    return pb.TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=pb.TaskStatus(
            state=pb.TaskState.TASK_STATE_WORKING, message=message, timestamp=_timestamp(None)
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
    verbatim: bool = False,
) -> pb.TaskArtifactUpdateEvent:
    """One chunk of the answer artifact; the final chunk sets ``lastChunk``. A ``verbatim``
    answer's part is flagged."""
    artifact = pb.Artifact(artifact_id=artifact_id, name=ANSWER_ARTIFACT_NAME)
    if text or not append:
        part = pb.Part(text=text)
        if verbatim:
            _mark_verbatim(part)
        artifact.parts.append(part)

    return pb.TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=append,
        last_chunk=last_chunk,
    )


def final_event(
    record: TaskRecord, context_id: str, *, exts: Extensions, verbatim: bool = False
) -> pb.TaskStatusUpdateEvent:
    """The terminal status event, built from the committed row so a replay says the same; a
    question asked ``verbatim`` has its text parts flagged."""
    event = pb.TaskStatusUpdateEvent(
        task_id=record.id,
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
        if verbatim and record.status == "input-required":
            for part in status_message.parts:
                if part.WhichOneof("content") == "text":
                    _mark_verbatim(part)
        event.status.message.CopyFrom(status_message)

    metadata = metadata_to_wire(record.data.metadata)
    if metadata:
        event.metadata.CopyFrom(struct(metadata))
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
    verbatim: bool = False
    """The text reaches the person exactly as written: every text part of it is flagged."""


@dataclass
class _Answer:
    """The artifact chunks of one task, accumulated until its terminal status."""

    bodies: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    all_verbatim: dict[str, bool] = field(default_factory=dict)
    last: str | None = None
    """The artifact whose last chunk arrived last: the whole answer, not a partial."""

    def collect(self, artifact: pb.Artifact, append: bool, last_chunk: bool = False) -> None:
        chunk = _text(artifact.parts)
        verbatim = _all_verbatim(artifact.parts)
        if append:
            chunk = self.bodies.get(artifact.artifact_id, "") + chunk
        if append and artifact.artifact_id in self.all_verbatim:
            # an empty closing chunk keeps what the chunks before it were
            verbatim = self.all_verbatim[artifact.artifact_id] and (verbatim or not artifact.parts)
        self.bodies[artifact.artifact_id] = chunk
        self.all_verbatim[artifact.artifact_id] = verbatim

        if artifact.name:
            self.names[artifact.artifact_id] = artifact.name
        if last_chunk:
            self.last = artifact.artifact_id

    def _answer_ids(self) -> list[str]:
        """The ``answer`` artifact that ended last; every ``answer`` artifact when none ended,
        every artifact when none carries that name."""
        named = [i for i in self.bodies if self.names.get(i) == ANSWER_ARTIFACT_NAME]
        if self.last is not None and (not named or self.last in named):
            return [self.last]
        return named or list(self.bodies)

    def text(self) -> str:
        """The answer's text."""
        return "\n".join(self.bodies[i] for i in self._answer_ids())

    def verbatim(self) -> bool:
        """Whether every part of the answer is flagged to reach the person as written."""
        ids = self._answer_ids()
        return bool(ids) and all(self.all_verbatim.get(i, False) for i in ids)


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
            verbatim=_all_verbatim(event.status.message.parts),
        )

    if state == "input-required" and text:
        # a question is the status message, whatever text streamed before it
        return TaskUpdate(
            state=state,
            text=text,
            task_id=ids.task_id,
            context_id=ids.context_id,
            metadata=metadata,
            verbatim=_all_verbatim(event.status.message.parts),
        )

    answer_text = answer.text()
    return TaskUpdate(
        state=state,
        text=answer_text or text,
        task_id=ids.task_id,
        context_id=ids.context_id,
        metadata=metadata,
        verbatim=answer.verbatim() if answer_text else _all_verbatim(event.status.message.parts),
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
            answer.collect(event.artifact, event.append, event.last_chunk)
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
