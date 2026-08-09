"""Mapping between the SDK's own types and the agent_session wire format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from livekit.protocol.agent_pb import agent_session as agent_pb

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


_METRICS_FIELDS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
    "llm_node_tps",
    "llm_node_ttfs",
)


def _metrics_to_proto(metrics: Mapping[str, Any] | None) -> agent_pb.MetricsReport:
    if not metrics:
        return agent_pb.MetricsReport()
    kwargs = {k: metrics[k] for k in _METRICS_FIELDS if k in metrics}
    return agent_pb.MetricsReport(**kwargs)


def _chat_item_to_proto(item: ChatItem) -> agent_pb.ChatContext.ChatItem:
    if item.type == "message":
        role_map = {
            "developer": agent_pb.DEVELOPER,
            "system": agent_pb.SYSTEM,
            "user": agent_pb.USER,
            "assistant": agent_pb.ASSISTANT,
        }
        pb_role = role_map.get(item.role, agent_pb.ASSISTANT)
        content = []
        if item.raw_text_content:
            content.append(agent_pb.ChatMessage.ChatContent(text=item.raw_text_content))
        pb_msg = agent_pb.ChatMessage(
            id=item.id,
            role=pb_role,
            content=content,
            interrupted=item.interrupted,
            metrics=_metrics_to_proto(item.metrics),
        )
        return agent_pb.ChatContext.ChatItem(message=pb_msg)
    elif item.type == "function_call":
        return agent_pb.ChatContext.ChatItem(
            function_call=agent_pb.FunctionCall(
                id=item.id,
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
        )
    elif item.type == "function_call_output":
        return agent_pb.ChatContext.ChatItem(
            function_call_output=agent_pb.FunctionCallOutput(
                call_id=item.call_id,
                name=item.name,
                output=item.output,
                is_error=item.is_error,
            )
        )
    elif item.type == "agent_handoff":
        return agent_pb.ChatContext.ChatItem(
            agent_handoff=agent_pb.AgentHandoff(
                id=item.id,
                old_agent_id=item.old_agent_id,
                new_agent_id=item.new_agent_id,
            )
        )
    elif item.type == "agent_config_update":
        return agent_pb.ChatContext.ChatItem(
            agent_config_update=agent_pb.AgentConfigUpdate(
                id=item.id,
                instructions=str(item.instructions) if item.instructions is not None else None,
                tools_added=item.tools_added or [],
                tools_removed=item.tools_removed or [],
            )
        )
    return agent_pb.ChatContext.ChatItem()


def _build_proto_chat_item(
    item: ChatItem,
) -> agent_pb.ChatContext.ChatItem:
    item_pb = agent_pb.ChatContext.ChatItem()

    if item.type == "message":
        msg = item_pb.message
        msg.id = item.id

        role_map = {
            "developer": agent_pb.DEVELOPER,
            "system": agent_pb.SYSTEM,
            "user": agent_pb.USER,
            "assistant": agent_pb.ASSISTANT,
        }
        msg.role = role_map[item.role]

        from .llm.chat_context import Instructions

        for content in item.content:
            if isinstance(content, (str, Instructions)):
                content_pb = msg.content.add()
                content_pb.text = str(content)

        msg.interrupted = item.interrupted

        if item.transcript_confidence is not None:
            msg.transcript_confidence = item.transcript_confidence

        for key, value in item.extra.items():
            msg.extra[key] = str(value)

        metrics = item.metrics
        if "started_speaking_at" in metrics:
            msg.metrics.started_speaking_at.FromMilliseconds(
                int(metrics["started_speaking_at"] * 1000)
            )
        if "stopped_speaking_at" in metrics:
            msg.metrics.stopped_speaking_at.FromMilliseconds(
                int(metrics["stopped_speaking_at"] * 1000)
            )
        if "transcription_delay" in metrics:
            msg.metrics.transcription_delay = metrics["transcription_delay"]
        if "end_of_turn_delay" in metrics:
            msg.metrics.end_of_turn_delay = metrics["end_of_turn_delay"]
        if "on_user_turn_completed_delay" in metrics:
            msg.metrics.on_user_turn_completed_delay = metrics["on_user_turn_completed_delay"]
        if "llm_node_ttft" in metrics:
            msg.metrics.llm_node_ttft = metrics["llm_node_ttft"]
        if "tts_node_ttfb" in metrics:
            msg.metrics.tts_node_ttfb = metrics["tts_node_ttfb"]
        if "e2e_latency" in metrics:
            msg.metrics.e2e_latency = metrics["e2e_latency"]
        msg.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "function_call":
        fc = item_pb.function_call
        fc.id = item.id
        fc.call_id = item.call_id
        fc.arguments = item.arguments
        fc.name = item.name
        fc.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "function_call_output":
        fco = item_pb.function_call_output
        fco.id = item.id
        fco.name = item.name
        fco.call_id = item.call_id
        fco.output = item.output
        fco.is_error = item.is_error
        fco.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "agent_handoff":
        ah = item_pb.agent_handoff
        ah.id = item.id
        if item.old_agent_id is not None:
            ah.old_agent_id = item.old_agent_id
        ah.new_agent_id = item.new_agent_id
        ah.created_at.FromMilliseconds(int(item.created_at * 1000))

    elif item.type == "agent_config_update":
        acu = item_pb.agent_config_update
        acu.id = item.id
        if item.instructions is not None:
            acu.instructions = item.instructions
        if item.tools_added:
            acu.tools_added.extend(item.tools_added)
        if item.tools_removed:
            acu.tools_removed.extend(item.tools_removed)
        acu.created_at.FromMilliseconds(int(item.created_at * 1000))

    return item_pb


def _session_usage_to_proto(usage: AgentSessionUsage) -> agent_pb.AgentSessionUsage:
    model_usages: list[agent_pb.ModelUsage] = []
    for mu in usage.model_usage:
        if isinstance(mu, LLMModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    llm=agent_pb.LLMModelUsage(
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
            )
        elif isinstance(mu, TTSModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    tts=agent_pb.TTSModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        input_tokens=mu.input_tokens,
                        output_tokens=mu.output_tokens,
                        characters_count=mu.characters_count,
                        audio_duration=mu.audio_duration,
                    )
                )
            )
        elif isinstance(mu, STTModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    stt=agent_pb.STTModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        input_tokens=mu.input_tokens,
                        output_tokens=mu.output_tokens,
                        audio_duration=mu.audio_duration,
                    )
                )
            )
        elif isinstance(mu, InterruptionModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    interruption=agent_pb.InterruptionModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        total_requests=mu.total_requests,
                    )
                )
            )
        elif isinstance(mu, EOTModelUsage):
            model_usages.append(
                agent_pb.ModelUsage(
                    eot=agent_pb.EotModelUsage(
                        provider=mu.provider,
                        model=mu.model,
                        total_requests=mu.total_requests,
                    )
                )
            )
    return agent_pb.AgentSessionUsage(model_usage=model_usages)
