"""Mapping between the SDK's own types and the agent_session wire format.

A leaf: nothing here imports ``llm``, ``voice`` or ``telemetry`` at module scope, so
every path that puts a ChatItem on the wire can share one implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import JsonValue, TypeAdapter

from livekit.protocol.agent_pb import agent_session as _pb

from ._generated.chat_items import MetricsReport
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
    from .llm.chat_context import ChatRole

_ProtoT = TypeVar("_ProtoT", bound=Message)

_METRICS: TypeAdapter[MetricsReport] = TypeAdapter(MetricsReport)


class _Timestamped(Protocol):
    @property
    def created_at(self) -> Timestamp: ...
    def HasField(self, field_name: Literal["created_at"], /) -> bool: ...


ROLES: dict[ChatRole, int] = {
    "developer": _pb.DEVELOPER,
    "system": _pb.SYSTEM,
    "user": _pb.USER,
    "assistant": _pb.ASSISTANT,
}

ROLE_NAMES = {v: k for k, v in ROLES.items()}

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


def encode_by_name(msg_type: type[_ProtoT], src: object, **overrides: object) -> _ProtoT:
    """Build ``msg_type`` by copying same-named entries off ``src``, a mapping or an object.

    Only fields the proto declares are read, and a field ``src`` has nothing for is
    left unset. Seconds-since-epoch floats fill Timestamp fields.
    """

    def lookup(name: str) -> object:
        return src.get(name) if isinstance(src, Mapping) else getattr(src, name, None)

    kwargs: dict[str, object] = dict(overrides)
    for fd in msg_type.DESCRIPTOR.fields:
        if fd.name in kwargs:
            continue
        value = lookup(fd.name)
        if value is None:
            continue
        if fd.message_type is not None and fd.message_type.full_name == "google.protobuf.Timestamp":
            if not isinstance(value, (int, float)):
                raise TypeError(f"{fd.name} must be seconds since the epoch")
            value = encode_timestamp(value)
        kwargs[fd.name] = value
    return msg_type(**kwargs)


def encode_metrics(metrics: Mapping[str, object] | None) -> _pb.MetricsReport:
    return encode_by_name(_pb.MetricsReport, metrics or {})


def encode_chat_item(item: ChatItem) -> _pb.ChatContext.ChatItem:
    if item.type == "message":
        content = [_pb.ChatMessage.ChatContent(text=c) for c in item.content if isinstance(c, str)]
        return _pb.ChatContext.ChatItem(
            message=encode_by_name(
                _pb.ChatMessage,
                item,
                role=ROLES[item.role],
                content=content,
                metrics=encode_metrics(item.metrics),
                extra=_encode_extra(item.extra),
            )
        )
    elif item.type == "function_call":
        return _pb.ChatContext.ChatItem(
            function_call=encode_by_name(_pb.FunctionCall, item, extra=_encode_extra(item.extra))
        )
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


def decode_chat_item(pb: _pb.ChatContext.ChatItem) -> ChatItem:
    from .llm.chat_context import (
        AgentConfigUpdate,
        AgentHandoff,
        ChatMessage,
        FunctionCall,
        FunctionCallOutput,
    )

    kind = pb.WhichOneof("item")
    if kind == "message":
        m = pb.message
        return ChatMessage(
            id=m.id,
            role=ROLE_NAMES[m.role],
            content=[c.text for c in m.content if c.HasField("text")],
            interrupted=m.interrupted,
            transcript_confidence=(
                m.transcript_confidence if m.HasField("transcript_confidence") else None
            ),
            extra=_decode_extra(m.extra),
            metrics=_METRICS.validate_python(_decode_fields(m.metrics)),
            **_created_at(m),
        )
    if kind == "function_call":
        f = pb.function_call
        return FunctionCall(
            id=f.id,
            call_id=f.call_id,
            arguments=f.arguments,
            name=f.name,
            extra=_decode_extra(f.extra),
            **_created_at(f),
        )
    if kind == "function_call_output":
        o = pb.function_call_output
        return FunctionCallOutput(
            id=o.id,
            name=o.name,
            call_id=o.call_id,
            output=o.output,
            is_error=o.is_error,
            **_created_at(o),
        )
    if kind == "agent_handoff":
        h = pb.agent_handoff
        return AgentHandoff(
            id=h.id,
            old_agent_id=h.old_agent_id if h.HasField("old_agent_id") else None,
            new_agent_id=h.new_agent_id,
            **_created_at(h),
        )
    if kind == "agent_config_update":
        c = pb.agent_config_update
        return AgentConfigUpdate(
            id=c.id,
            instructions=c.instructions if c.HasField("instructions") else None,
            tools_added=list(c.tools_added) or None,
            tools_removed=list(c.tools_removed) or None,
            **_created_at(c),
        )
    raise ValueError("a chat item without an item")


def _encode_extra(extra: Mapping[str, object]) -> dict[str, str]:
    """The proto's ``extra`` holds strings: any other value as compact JSON, as the core writes it."""
    return {
        k: v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
        for k, v in extra.items()
    }


def _decode_extra(extra: Mapping[str, str]) -> dict[str, JsonValue]:
    """``extra`` as the core reads it: JSON values decoded, any other string kept."""
    decoded: dict[str, JsonValue] = {}
    for k, v in extra.items():
        try:
            decoded[k] = json.loads(v)
        except ValueError:
            decoded[k] = v
    return decoded


def _created_at(msg: _Timestamped) -> dict[str, float]:
    return (
        {"created_at": msg.created_at.ToNanoseconds() / 1e9} if msg.HasField("created_at") else {}
    )


def _decode_fields(msg: Message) -> dict[str, object]:
    """The fields ``msg`` has set, Timestamps as seconds since the epoch."""
    return {
        fd.name: v.ToNanoseconds() / 1e9 if isinstance(v, Timestamp) else v
        for fd, v in msg.ListFields()
    }


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
