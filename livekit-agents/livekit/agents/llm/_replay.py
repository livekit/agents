"""Serves LLM requests from a recorded session instead of the provider.

With ``LIVEKIT_LLM_REPLAY_URL`` set (``lk agent simulate replay`` sets it), every
provider request is posted there as the GenAI attributes its ``llm_request`` span
records, and the response is the output that request got in the recorded session.
The server owns the comparison: a request that differs from the recorded one, or
arrives out of order, is refused, and the stream raises instead of falling back to
the provider.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import aiohttp

from .._exceptions import APIStatusError
from ..telemetry import gen_ai as gen_ai_telemetry, trace_types

if TYPE_CHECKING:
    from .llm import LLMStream

ENV_LLM_REPLAY_URL = "LIVEKIT_LLM_REPLAY_URL"


def replay_url() -> str | None:
    return os.environ.get(ENV_LLM_REPLAY_URL) or None


def _request(stream: LLMStream) -> dict[str, Any]:
    # the same serialization the llm_request span records, so the server compares
    # like with like
    return {
        "model": stream._llm.model,
        "provider": trace_types.gen_ai_provider_name(stream._llm.provider),
        "system_instructions": gen_ai_telemetry.to_system_instructions(stream.chat_ctx),
        "input_messages": gen_ai_telemetry.to_input_messages(stream.chat_ctx),
        "tool_definitions": gen_ai_telemetry.to_tool_definitions(stream.tools),
    }


async def serve(stream: LLMStream, url: str) -> None:
    from .llm import ChatChunk, ChoiceDelta, FunctionToolCall

    async with aiohttp.ClientSession() as http:
        async with http.post(
            url,
            data=gen_ai_telemetry._json(_request(stream)),
            headers={"Content-Type": "application/json"},
        ) as resp:
            body = await resp.json(content_type=None)
            if resp.status != 200:
                raise APIStatusError(
                    f"replay: {body.get('error', resp.reason)}",
                    status_code=resp.status,
                    retryable=False,
                )

    for message in body.get("output_messages") or []:
        text = ""
        tool_calls: list[FunctionToolCall] = []
        for part in message.get("parts") or []:
            if part.get("type") == "text":
                text += part.get("content", "")
            elif part.get("type") == "tool_call":
                arguments = part.get("arguments", "")
                tool_calls.append(
                    FunctionToolCall(
                        name=part["name"],
                        call_id=part["id"],
                        arguments=arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    )
                )
        stream._event_ch.send_nowait(
            ChatChunk(
                id=body.get("response_id") or "",
                delta=ChoiceDelta(role="assistant", content=text or None, tool_calls=tool_calls),
            )
        )
