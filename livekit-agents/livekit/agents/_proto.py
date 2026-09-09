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


def encode_timestamp(created_at: float) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(int(created_at * 1e9))
    return ts


def encode_by_name(msg_type: type[_ProtoT], src: Any, **overrides: Any) -> _ProtoT:
    """Build ``msg_type`` by copying same-named entries off ``src``, a mapping or an object.

    Only fields the proto declares are read, and a field ``src`` has nothing for is
    left unset. Seconds-since-epoch floats fill Timestamp fields.
    """

    def lookup(name: str) -> Any:
        return src.get(name) if isinstance(src, Mapping) else getattr(src, name, None)

    kwargs: dict[str, Any] = dict(overrides)
    for fd in msg_type.DESCRIPTOR.fields:
        if fd.name in kwargs:
            continue
        value = lookup(fd.name)
        if value is None:
            continue
        if fd.message_type is not None and fd.message_type.full_name == "google.protobuf.Timestamp":
            value = encode_timestamp(value)
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
                extra={k: str(v) for k, v in item.extra.items()},
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
