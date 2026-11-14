from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from livekit.agents import llm
from livekit.agents._proto import encode_chat_item, encode_metrics, encode_session_usage
from livekit.agents.llm import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
)
from livekit.agents.metrics import (
    AgentSessionUsage,
    EOTModelUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    STTModelUsage,
    TTSModelUsage,
)
from livekit.protocol.agent_pb import agent_session as agent_pb

pytestmark = pytest.mark.unit

# One serializer walks the ChatItem -> ChatContext.ChatItem mapping; the RemoteSession
# and telemetry paths share it. It must reach every field the proto declares, or that
# field is silently dropped on the wire with nothing at the type level to catch it.
SERIALIZERS: dict[str, Callable[[llm.ChatItem], agent_pb.ChatContext.ChatItem]] = {
    "telemetry_traces": encode_chat_item,
}

# sub-millisecond, so a serializer that truncates timestamps diverges from one that doesn't
_TS = 1_700_000_000.0005

# Every field carries a non-default value: proto3 scalars have no presence, so a field
# left at its zero value is indistinguishable from one the serializer never assigned.
SATURATED_ITEMS: list[llm.ChatItem] = [
    ChatMessage(
        id="item_msg",
        role="user",
        # multi-part: `content` is repeated, and a serializer that joins the parts into
        # one entry still fills the field, so only asserting on the shape catches it
        content=["hello", "bye"],
        interrupted=True,
        transcript_confidence=0.75,
        extra={"key": "value"},
        created_at=_TS,
        metrics={
            "started_speaking_at": _TS,
            "stopped_speaking_at": _TS,
            "transcription_delay": 1.0,
            "end_of_turn_delay": 1.0,
            "on_user_turn_completed_delay": 1.0,
            "llm_node_ttft": 1.0,
            "tts_node_ttfb": 1.0,
            "e2e_latency": 1.0,
            "llm_node_tps": 1.0,
            "llm_node_ttfs": 1.0,
        },
    ),
    FunctionCall(
        id="item_fc",
        call_id="call-1",
        name="fn",
        arguments="{}",
        extra={"livekit": {"inference_deployment": "d"}},
        created_at=_TS,
    ),
    FunctionCallOutput(
        id="item_fco",
        call_id="call-1",
        name="fn",
        output="ok",
        is_error=True,
        created_at=_TS,
    ),
    AgentHandoff(id="item_ah", old_agent_id="a", new_agent_id="b", created_at=_TS),
    AgentConfigUpdate(
        id="item_acu",
        instructions="be brief",
        tools_added=["x"],
        tools_removed=["y"],
        created_at=_TS,
    ),
]


def _repeated(fd: FieldDescriptor) -> bool:
    # FieldDescriptor.is_repeated landed in protobuf 6.31; the package floor is 3
    repeated = getattr(fd, "is_repeated", None)
    if repeated is not None:
        return bool(repeated)
    return fd.label == fd.LABEL_REPEATED


@pytest.mark.parametrize("repeated", [False, True])
def test_repeated_supports_descriptors_without_label(repeated: bool) -> None:
    from types import SimpleNamespace

    assert _repeated(SimpleNamespace(is_repeated=repeated)) is repeated


@pytest.mark.parametrize("repeated", [False, True])
def test_repeated_supports_legacy_descriptors(repeated: bool) -> None:
    from types import SimpleNamespace

    descriptor = SimpleNamespace(label=3 if repeated else 1, LABEL_REPEATED=3)
    assert _repeated(descriptor) is repeated


def _unset_fields(msg: Message, prefix: str = "") -> list[str]:
    present = {fd.name for fd, _ in msg.ListFields()}
    missing: list[str] = []
    for name, fd in msg.DESCRIPTOR.fields_by_name.items():
        if name not in present:
            missing.append(prefix + name)
            continue
        # well-known types are leaves; their own fields are legitimately zero
        nested = fd.message_type is not None and not _repeated(fd)
        if nested and not fd.message_type.full_name.startswith("google.protobuf."):
            missing.extend(_unset_fields(getattr(msg, name), prefix + name + "."))
    return missing


def _oneof_payload(pb_item: agent_pb.ChatContext.ChatItem) -> tuple[str, Message]:
    which = pb_item.WhichOneof("item")
    assert which is not None
    return which, getattr(pb_item, which)


@pytest.mark.parametrize("serializer", SERIALIZERS.keys())
@pytest.mark.parametrize("item", SATURATED_ITEMS, ids=lambda i: i.type)
def test_every_proto_field_is_populated(serializer: str, item: llm.ChatItem) -> None:
    which, payload = _oneof_payload(SERIALIZERS[serializer](item))
    dropped = _unset_fields(payload)
    assert not dropped, f"{serializer} drops {which}: {', '.join(dropped)}"


def test_message_content_is_one_entry_per_part() -> None:
    msg = encode_chat_item(SATURATED_ITEMS[0]).message
    assert [c.text for c in msg.content] == ["hello", "bye"]


def test_map_values_are_stringified() -> None:
    # extra dicts hold provider payloads and flags, the wire maps only strings
    extra = {"livekit": {"inference_deployment": "d"}, "flag": True}
    msg = encode_chat_item(ChatMessage(role="user", content=["hi"], extra=extra)).message
    assert dict(msg.extra) == {k: str(v) for k, v in extra.items()}


def test_function_call_extra_never_reaches_the_wire() -> None:
    # livekit-protocol 1.1.28 added FunctionCall.extra; agents 1.7.0-1.8.8 filled it from the
    # SDK's nested extra and every session report with a tool call failed to encode
    call = FunctionCall(
        call_id="call-1",
        name="fn",
        arguments="{}",
        extra={"google": {"thought_signature": b"sig"}, "__livekit_agents_tool_non_blocking": True},
    )
    encode_chat_item(call).SerializeToString()


USAGE_SOURCES: list[tuple[Any, type[Message], str]] = [
    (LLMModelUsage, agent_pb.LLMModelUsage, "llm"),
    (TTSModelUsage, agent_pb.TTSModelUsage, "tts"),
    (STTModelUsage, agent_pb.STTModelUsage, "stt"),
    (InterruptionModelUsage, agent_pb.InterruptionModelUsage, "interruption"),
    (EOTModelUsage, agent_pb.EotModelUsage, "eot"),
]


def _saturated(src_type: Any, msg_type: type[Message]) -> Any:
    kwargs: dict[str, Any] = {}
    for fd in msg_type.DESCRIPTOR.fields:
        if fd.name not in src_type.model_fields:
            continue
        kwargs[fd.name] = "x" if fd.cpp_type == fd.CPPTYPE_STRING else 7
    return src_type(**kwargs)


@pytest.mark.parametrize(
    ("src_type", "msg_type", "variant"),
    USAGE_SOURCES,
    ids=[src.__name__ for src, _, _ in USAGE_SOURCES],
)
def test_model_usage_reaches_every_proto_field(
    src_type: Any, msg_type: type[Message], variant: str
) -> None:
    pb_usage = encode_session_usage(AgentSessionUsage(model_usage=[_saturated(src_type, msg_type)]))
    dropped = _unset_fields(getattr(pb_usage.model_usage[0], variant))
    assert not dropped, f"{msg_type.DESCRIPTOR.name} drops {', '.join(dropped)}"


def test_session_usage_carries_every_variant() -> None:
    sources = [_saturated(src, msg) for src, msg, _ in USAGE_SOURCES]
    pb_usage = encode_session_usage(AgentSessionUsage(model_usage=sources))
    assert [mu.WhichOneof("usage") for mu in pb_usage.model_usage] == [
        variant for _, _, variant in USAGE_SOURCES
    ]


def test_encode_metrics_reaches_every_proto_field() -> None:
    metrics = {fd.name: 1.5 for fd in agent_pb.MetricsReport.DESCRIPTOR.fields}
    assert _unset_fields(encode_metrics(metrics)) == []


def test_every_metrics_field_has_a_source_key() -> None:
    declared = {fd.name for fd in agent_pb.MetricsReport.DESCRIPTOR.fields}
    assert declared <= set(llm.chat_context.MetricsReport.__annotations__)


def test_guard_detects_a_dropped_field() -> None:
    fco = agent_pb.FunctionCallOutput(call_id="call-1", name="fn", output="ok", is_error=True)
    assert _unset_fields(fco) == ["id", "created_at"]


def test_saturated_items_cover_every_item_type() -> None:
    covered = {_oneof_payload(encode_chat_item(i))[0] for i in SATURATED_ITEMS}
    declared = {
        fd.name for fd in agent_pb.ChatContext.ChatItem.DESCRIPTOR.oneofs_by_name["item"].fields
    }
    assert covered == declared
