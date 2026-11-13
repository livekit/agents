"""Ultravox data messages (https://docs.ultravox.ai/datamessages), as the plugin reads and sends
them. Output audio arrives as binary frames; input audio is sent as binary frames."""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import Field, JsonValue

# server messages: only the fields the plugin reads


class CallStarted(TypedDict):
    type: Literal["call_started"]


class State(TypedDict):
    type: Literal["state"]
    state: str


class Transcript(TypedDict):
    type: Literal["transcript"]
    role: str
    final: bool
    ordinal: int
    text: NotRequired[str | None]
    delta: NotRequired[str | None]


class ClientToolInvocation(TypedDict):
    type: Literal["client_tool_invocation"]
    toolName: str
    invocationId: str
    parameters: dict[str, JsonValue]


class PlaybackClearBuffer(TypedDict):
    type: Literal["playback_clear_buffer"]


ServerMessage = Annotated[
    CallStarted | State | Transcript | ClientToolInvocation | PlaybackClearBuffer | bytes,
    Field(discriminator="type"),
]

# client messages


class UserTextMessage(TypedDict):
    type: Literal["user_text_message"]
    text: str
    deferResponse: bool
    urgency: NotRequired[Literal["immediate", "soon", "later"]]


class ForcedAgentMessage(TypedDict):
    type: Literal["forced_agent_message"]
    content: str
    uninterruptible: bool


class SetOutputMedium(TypedDict):
    type: Literal["set_output_medium"]
    medium: Literal["voice", "text"]


class ClientToolResult(TypedDict):
    type: Literal["client_tool_result"]
    invocationId: str
    agentReaction: Literal["speaks", "listens", "speaks-once"]
    responseType: str
    result: NotRequired[str]
    errorType: NotRequired[Literal["undefined", "implementation-error"]]
    errorMessage: NotRequired[str]


ClientMessage = UserTextMessage | ForcedAgentMessage | SetOutputMedium | ClientToolResult
