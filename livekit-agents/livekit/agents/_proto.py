"""Mapping between the SDK's own types and the agent_session wire format.

A leaf: nothing here imports ``llm``, ``voice`` or ``telemetry`` at module scope, so
every path that puts a ChatItem on the wire can share one implementation.

Every field is set by hand. A protocol release that adds a field must not change what an
already-released SDK puts on the wire.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
    from .llm.chat_context import MetricsReport

ROLES = {
    "developer": _pb.DEVELOPER,
    "system": _pb.SYSTEM,
    "user": _pb.USER,
    "assistant": _pb.ASSISTANT,
}


def encode_timestamp(created_at: float) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(int(created_at * 1e9))
    return ts


def _optional_timestamp(value: float | None) -> Timestamp | None:
    return encode_timestamp(value) if value is not None else None


def _string_map(extra: Mapping[str, Any]) -> dict[str, str]:
    # every map these protos declare is map<string, string>
    return {k: str(v) for k, v in extra.items()}


def encode_metrics(metrics: MetricsReport | Mapping[str, Any] | None) -> _pb.MetricsReport:
    m: Mapping[str, Any] = metrics or {}
    return _pb.MetricsReport(
        started_speaking_at=_optional_timestamp(m.get("started_speaking_at")),
        stopped_speaking_at=_optional_timestamp(m.get("stopped_speaking_at")),
        transcription_delay=m.get("transcription_delay"),
        end_of_turn_delay=m.get("end_of_turn_delay"),
        on_user_turn_completed_delay=m.get("on_user_turn_completed_delay"),
        llm_node_ttft=m.get("llm_node_ttft"),
        tts_node_ttfb=m.get("tts_node_ttfb"),
        e2e_latency=m.get("e2e_latency"),
        llm_node_tps=m.get("llm_node_tps"),
        llm_node_ttfs=m.get("llm_node_ttfs"),
    )


def encode_chat_item(item: ChatItem) -> _pb.ChatContext.ChatItem:
    if item.type == "message":
        from .llm.chat_context import Instructions

        return _pb.ChatContext.ChatItem(
            message=_pb.ChatMessage(
                id=item.id,
                role=ROLES[item.role],
                content=[
                    _pb.ChatMessage.ChatContent(text=str(c))
                    for c in item.content
                    if isinstance(c, (str, Instructions))
                ],
                interrupted=item.interrupted,
                transcript_confidence=item.transcript_confidence,
                extra=_string_map(item.extra),
                metrics=encode_metrics(item.metrics),
                created_at=encode_timestamp(item.created_at),
            )
        )
    elif item.type == "function_call":
        return _pb.ChatContext.ChatItem(
            function_call=_pb.FunctionCall(
                id=item.id,
                call_id=item.call_id,
                arguments=item.arguments,
                name=item.name,
                created_at=encode_timestamp(item.created_at),
            )
        )
    elif item.type == "function_call_output":
        return _pb.ChatContext.ChatItem(
            function_call_output=_pb.FunctionCallOutput(
                id=item.id,
                name=item.name,
                call_id=item.call_id,
                output=item.output,
                is_error=item.is_error,
                created_at=encode_timestamp(item.created_at),
            )
        )
    elif item.type == "agent_handoff":
        return _pb.ChatContext.ChatItem(
            agent_handoff=_pb.AgentHandoff(
                id=item.id,
                old_agent_id=item.old_agent_id,
                new_agent_id=item.new_agent_id,
                created_at=encode_timestamp(item.created_at),
            )
        )
    elif item.type == "agent_config_update":
        return _pb.ChatContext.ChatItem(
            agent_config_update=_pb.AgentConfigUpdate(
                id=item.id,
                instructions=str(item.instructions) if item.instructions else None,
                tools_added=item.tools_added,
                tools_removed=item.tools_removed,
                created_at=encode_timestamp(item.created_at),
            )
        )
    return _pb.ChatContext.ChatItem()


def encode_session_usage(usage: AgentSessionUsage) -> _pb.AgentSessionUsage:
    model_usages: list[_pb.ModelUsage] = []
    for mu in usage.model_usage:
        if isinstance(mu, LLMModelUsage):
            pb_usage = _pb.ModelUsage(
                llm=_pb.LLMModelUsage(
                    provider=mu.provider,
                    model=mu.model,
                    input_tokens=mu.input_tokens,
                    input_cached_tokens=mu.input_cached_tokens,
                    input_audio_tokens=mu.input_audio_tokens,
                    input_cached_audio_tokens=mu.input_cached_audio_tokens,
                    input_text_tokens=mu.input_text_tokens,
                    input_cached_text_tokens=mu.input_cached_text_tokens,
                    input_image_tokens=mu.input_image_tokens,
                    input_cached_image_tokens=mu.input_cached_image_tokens,
                    output_tokens=mu.output_tokens,
                    output_audio_tokens=mu.output_audio_tokens,
                    output_text_tokens=mu.output_text_tokens,
                    session_duration=mu.session_duration,
                )
            )
        elif isinstance(mu, TTSModelUsage):
            pb_usage = _pb.ModelUsage(
                tts=_pb.TTSModelUsage(
                    provider=mu.provider,
                    model=mu.model,
                    input_tokens=mu.input_tokens,
                    output_tokens=mu.output_tokens,
                    characters_count=mu.characters_count,
                    audio_duration=mu.audio_duration,
                )
            )
        elif isinstance(mu, STTModelUsage):
            pb_usage = _pb.ModelUsage(
                stt=_pb.STTModelUsage(
                    provider=mu.provider,
                    model=mu.model,
                    input_tokens=mu.input_tokens,
                    output_tokens=mu.output_tokens,
                    audio_duration=mu.audio_duration,
                )
            )
        elif isinstance(mu, InterruptionModelUsage):
            pb_usage = _pb.ModelUsage(
                interruption=_pb.InterruptionModelUsage(
                    provider=mu.provider, model=mu.model, total_requests=mu.total_requests
                )
            )
        elif isinstance(mu, EOTModelUsage):
            pb_usage = _pb.ModelUsage(
                eot=_pb.EotModelUsage(
                    provider=mu.provider, model=mu.model, total_requests=mu.total_requests
                )
            )
        else:
            continue
        model_usages.append(pb_usage)
    return _pb.AgentSessionUsage(model_usage=model_usages)
