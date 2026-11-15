# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/mcp.rs

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import JsonValue
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolFlag

    from .tool_context import DuplicateMode, DuplicateScope


# Record declarations:


class ResolvedToolOptions(TypedDict):
    """Resolved per-tool behavior passed to the host's function-tool decorator."""

    flags: ToolFlag
    """Passed through to the `@function_tool` decorator, using `ToolFlag`."""
    on_duplicate: DuplicateMode
    duplicate_scope: DuplicateScope
    report_progress: bool
    """Forward the tool's MCP progress notifications to `ctx.update()`. Also selects the
    non-blocking call path (`_tool_called_nonblocking`, which takes a `RunContext`).
    """
    retries: int | None
    """The tool's `retries=`; `None` reads the run's tool budget."""
    timeout: float | None
    """This tool's request timeout in seconds, in place of the server's `request_timeout`."""


class MCPToolOptions(TypedDict, total=False):
    """Per-tool behavior for MCP tools exposed by an MCPToolset."""

    flags: ToolFlag
    """Flags passed to the @function_tool decorator, using ToolFlag."""
    on_duplicate: DuplicateMode
    """Behavior when a tool is called multiple times in the same context."""
    duplicate_scope: DuplicateScope
    """What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too."""
    report_progress: bool
    """Whether to forward the tool's progress notifications to ctx.update()."""
    retries: int
    """The tool's retry budget, in place of the agent's ``retries['tools']``."""
    timeout: float
    """Seconds one call may wait without a reply or progress, in place of the server's
    ``request_timeout``.
    """


@dataclass(frozen=True)
class MCPTool:
    """A tool an MCP server lists."""

    name: str
    description: str | None
    """What the tool does, which MCP leaves optional."""
    input_schema: dict[str, JsonValue]
    """The JSON Schema of its arguments."""
    meta: dict[str, JsonValue] | None
    """The server's `_meta` for the tool."""


@dataclass(frozen=True)
class CallToolResult:
    """A tool call's result as the server sent it: its content blocks, and whether the tool failed."""

    content: list[ContentBlock]
    is_error: bool


@dataclass(frozen=True)
class TextContent:
    text: str
    meta: dict[str, JsonValue] | None
    annotations: Annotations | None
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageContent:
    data: str
    """Base64."""
    mime_type: str
    meta: dict[str, JsonValue] | None
    annotations: Annotations | None
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class AudioContent:
    data: str
    """Base64."""
    mime_type: str
    meta: dict[str, JsonValue] | None
    annotations: Annotations | None
    type: Literal["audio"] = "audio"


@dataclass(frozen=True)
class EmbeddedResource:
    """A resource's contents, inline in the result."""

    resource: ResourceContents
    meta: dict[str, JsonValue] | None
    annotations: Annotations | None
    type: Literal["resource"] = "resource"


@dataclass(frozen=True)
class TextResourceContents:
    uri: str
    mime_type: str | None
    text: str
    meta: dict[str, JsonValue] | None


@dataclass(frozen=True)
class BlobResourceContents:
    uri: str
    mime_type: str | None
    blob: str
    """Base64."""
    meta: dict[str, JsonValue] | None


@dataclass(frozen=True)
class ResourceLink:
    """A link to a resource the client may read."""

    uri: str
    name: str
    title: str | None
    description: str | None
    mime_type: str | None
    size: int | None
    """Bytes, before encoding."""
    icons: list[Icon] | None
    meta: dict[str, JsonValue] | None
    annotations: Annotations | None
    type: Literal["resource_link"] = "resource_link"


@dataclass(frozen=True)
class Icon:
    src: str
    mime_type: str | None
    sizes: list[str] | None
    """Each `WxH`, or `any`."""
    theme: IconTheme | None


IconTheme: TypeAlias = Literal["light", "dark"]
"""The background an icon is drawn for."""


@dataclass(frozen=True)
class Annotations:
    """How a client may use a block: who it is for, how much it matters, and when it last changed."""

    audience: list[Role] | None
    priority: float | None
    """From 0, least important, to 1."""
    last_modified: str | None
    """An ISO 8601 timestamp."""


Role: TypeAlias = Literal["user", "assistant"]


ContentBlock: TypeAlias = (
    TextContent | ImageContent | AudioContent | EmbeddedResource | ResourceLink
)
"""One block of a tool's result, in MCP's wire form."""


ResourceContents: TypeAlias = TextResourceContents | BlobResourceContents
"""A resource's contents: text, or a base64 blob."""


MCPToolOptions.__module__ = "livekit.agents.llm.mcp"
MCPTool.__annotations__ = get_annotations(MCPTool, eval_str=True)
MCPTool.__module__ = "livekit.agents.llm.mcp"


MCP_IDLE_CLOSE_SECONDS: Final = 60.0
"""Seconds a connection stays open after its toolset's last run ends, for the next run to reuse:
`MCPToolset(idle_timeout=)`'s default.
"""
