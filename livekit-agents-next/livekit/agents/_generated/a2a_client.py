# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/client.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class MessageMetadata:
    kind: str | None = None
    channel: str | None = None
    verbatim: str | None = None
    answer_tool: str | None = None
    merged_into: str | None = None
    reason: str | None = None
    interrupted: bool | None = None
    interruption_reason: str | None = None
    application: dict[str, object] | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class TextPart:
    text: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class DataPart:
    data: object
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class FilePart:
    url: str | None = None
    data: list[int] | None = None
    filename: str | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None


MessagePart: TypeAlias = TextPart | DataPart | FilePart


@dataclass(frozen=True, kw_only=True, slots=True)
class Message:
    parts: list[MessagePart]
    metadata: MessageMetadata | None = None
    message_id: str | None = None
    task_id: str | None = None
    context_id: str | None = None
    role: str = "user"
    channel: str | None = None
    received_at: float | None = None
    raw: object | None = None

    @property
    def text(self) -> str:
        """The text parts, joined by newlines."""
        from .. import _core

        return _core._core_a2a_message_text(self)

    @property
    def data(self) -> dict[str, object] | None:
        """The first object data part, if any."""
        from .. import _core

        return _core._core_a2a_message_data(self)


MessageInput: TypeAlias = str | Message
"""A string is shorthand for a generated message with one text part."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Artifact:
    artifact_id: str
    name: str | None = None
    description: str | None = None
    parts: list[MessagePart]
    metadata: dict[str, object]


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskUpdate:
    state: str
    text: str
    task_id: str
    context_id: str
    metadata: dict[str, object]
    artifacts: list[Artifact]


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentCard:
    name: str
    description: str
    extensions: list[str]
    input_schema: dict[str, object]
    structured_input: bool
