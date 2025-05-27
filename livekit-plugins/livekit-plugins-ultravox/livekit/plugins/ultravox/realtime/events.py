"""Ultravox WebSocket event definitions based on the Ultravox data message protocol.

This module defines all the event types that can be sent and received through
Ultravox's WebSocket API for real-time voice AI communication.

Reference: https://docs.ultravox.ai/datamessages
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class UltravoxEvent(BaseModel):
    """Base class for all Ultravox WebSocket events."""

    type: str = Field(..., description="Event type identifier")


# Client-to-Server Events


class PingEvent(UltravoxEvent):
    """Client message to measure round-trip data message latency."""

    type: Literal["ping"] = "ping"
    timestamp: float = Field(..., description="Client timestamp for latency measurement")


class InputTextMessageEvent(UltravoxEvent):
    """User message sent via text."""

    type: Literal["input_text_message"] = "input_text_message"
    text: str = Field(..., description="The content of the user message")
    defer_response: Optional[bool] = Field(
        None,
        alias="deferResponse",
        description="If true, allows adding text without inducing immediate response",
    )


class SetOutputMediumEvent(UltravoxEvent):
    """Message to set the server's output medium."""

    type: Literal["set_output_medium"] = "set_output_medium"
    medium: Literal["voice", "text"] = Field(..., description="Output medium type")


class ClientToolResultEvent(UltravoxEvent):
    """Contains the result of a client-implemented tool invocation."""

    type: Literal["client_tool_result"] = "client_tool_result"
    invocation_id: str = Field(
        ..., alias="invocationId", description="Matches corresponding invocation"
    )
    result: Optional[str] = Field(None, description="Tool execution result, often JSON string")
    agent_reaction: Optional[Literal["speaks", "listens", "speaks-once"]] = Field(
        "speaks", alias="agentReaction", description="How the agent should react to the tool result"
    )
    response_type: str = Field(
        "tool-response", alias="responseType", description="Type of response"
    )
    error_type: Optional[Literal["undefined", "implementation-error"]] = Field(
        None, alias="errorType", description="Error type if tool execution failed"
    )
    error_message: Optional[str] = Field(
        None, alias="errorMessage", description="Error details if failed"
    )


# Server-to-Client Events


class PongEvent(UltravoxEvent):
    """Server reply to a ping message."""

    type: Literal["pong"] = "pong"
    timestamp: float = Field(..., description="Matching ping timestamp")


class StateEvent(UltravoxEvent):
    """Server message indicating its current state."""

    type: Literal["state"] = "state"
    state: str = Field(..., description="Current session state")


class TranscriptEvent(UltravoxEvent):
    """Message containing text transcripts of user and agent utterances."""

    type: Literal["transcript"] = "transcript"
    role: Literal["user", "agent"] = Field(..., description="Who emitted the utterance")
    medium: Literal["text", "voice"] = Field(
        ..., description="Medium through which utterance was emitted"
    )
    text: Optional[str] = Field(None, description="Full transcript text (exclusive with delta)")
    delta: Optional[str] = Field(
        None, description="Incremental transcript update (exclusive with text)"
    )
    final: bool = Field(..., description="Whether more updates are expected for this utterance")
    ordinal: int = Field(..., description="Used for ordering transcripts within a call")


class ClientToolInvocationEvent(UltravoxEvent):
    """Server request for client to invoke a client-implemented tool."""

    type: Literal["client_tool_invocation"] = "client_tool_invocation"
    tool_name: str = Field(..., alias="toolName", description="Tool to invoke")
    invocation_id: str = Field(..., alias="invocationId", description="Unique invocation ID")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")


class DebugEvent(UltravoxEvent):
    """Server message for debugging information."""

    type: Literal["debug"] = "debug"
    message: str = Field(..., description="Debug information")


class PlaybackClearBufferEvent(UltravoxEvent):
    """Server message to clear buffered output audio (WebSocket only)."""

    type: Literal["playback_clear_buffer"] = "playback_clear_buffer"


# Union type for all possible events
UltravoxEventType = Union[
    # Client events
    PingEvent,
    InputTextMessageEvent,
    SetOutputMediumEvent,
    ClientToolResultEvent,
    # Server events
    PongEvent,
    StateEvent,
    TranscriptEvent,
    ClientToolInvocationEvent,
    DebugEvent,
    PlaybackClearBufferEvent,
]


def parse_ultravox_event(data: Dict[str, Any]) -> UltravoxEventType:
    """Parse a raw WebSocket message into an Ultravox event object.

    Parameters
    ----------
    data : Dict[str, Any]
        Raw JSON data from WebSocket message

    Returns
    -------
    UltravoxEventType
        Parsed event object

    Raises
    ------
    ValueError
        If the event type is unknown or data is invalid
    """
    event_type = data.get("type")

    if event_type == "ping":
        return PingEvent(**data)
    elif event_type == "input_text_message":
        return InputTextMessageEvent(**data)
    elif event_type == "set_output_medium":
        return SetOutputMediumEvent(**data)
    elif event_type == "client_tool_result":
        return ClientToolResultEvent(**data)
    elif event_type == "pong":
        return PongEvent(**data)
    elif event_type == "state":
        return StateEvent(**data)
    elif event_type == "transcript":
        return TranscriptEvent(**data)
    elif event_type == "client_tool_invocation":
        return ClientToolInvocationEvent(**data)
    elif event_type == "debug":
        return DebugEvent(**data)
    elif event_type == "playback_clear_buffer":
        return PlaybackClearBufferEvent(**data)
    else:
        raise ValueError(f"Unknown Ultravox event type: {event_type}")


def serialize_ultravox_event(event: UltravoxEventType) -> Dict[str, Any]:
    """Serialize an Ultravox event object to JSON-compatible dict.

    Parameters
    ----------
    event : UltravoxEventType
        Event object to serialize

    Returns
    -------
    Dict[str, Any]
        JSON-compatible dictionary
    """
    return event.model_dump(by_alias=True, exclude_none=True)
