"""Mapping between the SDK's own types and the agent_session wire format.

A leaf: nothing here imports ``llm``, ``voice`` or ``telemetry`` at module scope, so
every path that puts a ChatItem on the wire can share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from livekit.protocol.agent_pb import agent_session as _pb

from .metrics import (
    AgentSessionUsage,
    EOTModelUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    STTModelUsage,
    TTSModelUsage,
)

if TYPE_CHECKING:
    from .llm import ChatItem

_ProtoT = TypeVar("_ProtoT", bound=Message)

ROLES = {
    "developer": _pb.DEVELOPER,
    "system": _pb.SYSTEM,
    "user": _pb.USER,
    "assistant": _pb.ASSISTANT,
}

# which ModelUsage variant each usage type fills, and the proto carrying it
USAGE_VARIANTS: tuple[tuple[type, str, type[Message]], ...] = (
    (LLMModelUsage, "llm", _pb.LLMModelUsage),
    (TTSModelUsage, "tts", _pb.TTSModelUsage),
    (STTModelUsage, "stt", _pb.STTModelUsage),
    (InterruptionModelUsage, "interruption", _pb.InterruptionModelUsage),
    (EOTModelUsage, "eot", _pb.EotModelUsage),
)


# The fields each message is filled from, by name. Listed rather than read off the installed
# descriptor: a livekit-protocol release that adds a field must not change what an
# already-released SDK puts in it.
FIELDS: dict[type[Message], tuple[str, ...]] = {
    _pb.ChatMessage: ("id", "interrupted", "transcript_confidence", "extra", "created_at"),
    _pb.FunctionCall: ("id", "call_id", "arguments", "name", "created_at", "extra"),
    _pb.FunctionCallOutput: ("id", "name", "call_id", "output", "is_error", "created_at"),
    _pb.AgentHandoff: ("id", "old_agent_id", "new_agent_id", "created_at"),
    _pb.AgentConfigUpdate: ("id", "instructions", "tools_added", "tools_removed", "created_at"),
    _pb.MetricsReport: (
        "started_speaking_at",
        "stopped_speaking_at",
        "transcription_delay",
        "end_of_turn_delay",
        "on_user_turn_completed_delay",
        "llm_node_ttft",
        "tts_node_ttfb",
        "e2e_latency",
        "llm_node_tps",
        "llm_node_ttfs",
    ),
    _pb.LLMModelUsage: (
        "provider",
        "model",
        "input_tokens",
        "input_cached_tokens",
        "input_audio_tokens",
        "input_cached_audio_tokens",
        "input_text_tokens",
        "input_cached_text_tokens",
        "input_image_tokens",
        "input_cached_image_tokens",
        "output_tokens",
        "output_audio_tokens",
        "output_text_tokens",
        "session_duration",
    ),
    _pb.TTSModelUsage: (
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "characters_count",
        "audio_duration",
    ),
    _pb.STTModelUsage: ("provider", "model", "input_tokens", "output_tokens", "audio_duration"),
    _pb.InterruptionModelUsage: ("provider", "model", "total_requests"),
    _pb.EotModelUsage: ("provider", "model", "total_requests"),
}


def encode_timestamp(created_at: float) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(int(created_at * 1e9))
    return ts


def encode_by_name(msg_type: type[_ProtoT], src: Any, **overrides: Any) -> _ProtoT:
    """Build ``msg_type`` by copying same-named entries off ``src``, a mapping or an object.

    Only the fields ``FIELDS`` lists for ``msg_type`` are read, and a field ``src`` has
    nothing for is left unset. Seconds-since-epoch floats fill Timestamp fields, and map values are
    stringified: every map these protos declare is ``map<string, string>``.
    """

    def lookup(name: str) -> Any:
        return src.get(name) if isinstance(src, Mapping) else getattr(src, name, None)

    kwargs: dict[str, Any] = dict(overrides)
    for name in FIELDS[msg_type]:
        if name in kwargs:
            continue
        fd = msg_type.DESCRIPTOR.fields_by_name[name]
        value = lookup(name)
        if value is None:
            continue
        if fd.message_type is not None and fd.message_type.full_name == "google.protobuf.Timestamp":
            value = encode_timestamp(value)
        elif fd.message_type is not None and fd.message_type.GetOptions().map_entry:
            value = {k: str(v) for k, v in value.items()}
        kwargs[fd.name] = value
    return msg_type(**kwargs)


def encode_metrics(metrics: Mapping[str, Any] | None) -> _pb.MetricsReport:
    return encode_by_name(_pb.MetricsReport, metrics or {})


def encode_chat_item(item: ChatItem) -> _pb.ChatContext.ChatItem:
    if item.type == "message":
        from .llm.chat_context import Instructions

        content = [
            _pb.ChatMessage.ChatContent(text=str(c))
            for c in item.content
            if isinstance(c, (str, Instructions))
        ]
        return _pb.ChatContext.ChatItem(
            message=encode_by_name(
                _pb.ChatMessage,
                item,
                role=ROLES[item.role],
                content=content,
                metrics=encode_metrics(item.metrics),
            )
        )
    elif item.type == "function_call":
        return _pb.ChatContext.ChatItem(function_call=encode_by_name(_pb.FunctionCall, item))
    elif item.type == "function_call_output":
        return _pb.ChatContext.ChatItem(
            function_call_output=encode_by_name(_pb.FunctionCallOutput, item)
        )
    elif item.type == "agent_handoff":
        return _pb.ChatContext.ChatItem(agent_handoff=encode_by_name(_pb.AgentHandoff, item))
    elif item.type == "agent_config_update":
        return _pb.ChatContext.ChatItem(
            agent_config_update=encode_by_name(
                _pb.AgentConfigUpdate,
                item,
                **({"instructions": str(item.instructions)} if item.instructions else {}),
            )
        )
    return _pb.ChatContext.ChatItem()


def encode_session_usage(usage: AgentSessionUsage) -> _pb.AgentSessionUsage:
    model_usages: list[_pb.ModelUsage] = []
    for mu in usage.model_usage:
        for src_type, variant, msg_type in USAGE_VARIANTS:
            if isinstance(mu, src_type):
                pb_usage = _pb.ModelUsage()
                getattr(pb_usage, variant).CopyFrom(encode_by_name(msg_type, mu))
                model_usages.append(pb_usage)
                break
    return _pb.AgentSessionUsage(model_usage=model_usages)
