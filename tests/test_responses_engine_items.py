from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx2 as httpx
import openai
import pytest
from openai.types.responses import ResponseOutputItemAddedEvent, ResponseOutputMessage

from livekit.agents import APIConnectOptions, llm as agents_llm
from livekit.plugins.openai.responses.llm import LLM as ResponsesLLM, LLMStream

pytestmark = pytest.mark.plugin("openai")

_RESPONSE = {
    "id": "resp_1",
    "created_at": 0,
    "model": "gpt-4.1",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_RESPONSE_CREATED = {"type": "response.created", "sequence_number": 0, "response": _RESPONSE}
_USAGE = {
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _added(item_id: str, item_type: str, phase: str | None = None) -> dict:
    if item_type == "message":
        item: dict = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        if phase is not None:
            item["phase"] = phase
    else:
        item = {
            "id": item_id,
            "type": item_type,
            "status": "in_progress",
            "call_id": item_id,
            "name": item_id,
            "arguments": "",
        }
    return {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": item,
    }


def _delta(item_id: str, text: str) -> dict:
    return {
        "type": "response.output_text.delta",
        "sequence_number": 0,
        "output_index": 0,
        "content_index": 0,
        "item_id": item_id,
        "delta": text,
        "logprobs": [],
    }


def _msg_done(item_id: str, text: str, phase: str | None = None) -> dict:
    content = [{"type": "output_text", "text": text, "annotations": []}] if text else []
    item: dict = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": content,
    }
    if phase is not None:
        item["phase"] = phase
    return {
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": 0,
        "item": item,
    }


def _fc_done(call_id: str, name: str) -> dict:
    return {
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "id": call_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "arguments": "{}",
        },
    }


def _web_search_call(item_id: str) -> list[dict]:
    added = {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"id": item_id, "type": "web_search_call", "status": "in_progress"},
    }
    done = {
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "weather"},
        },
    }
    return [added, done]


def _completed(usage: dict | None = None) -> dict:
    response = {**_RESPONSE, "status": "completed"}
    if usage is not None:
        response["usage"] = usage
    return {"type": "response.completed", "sequence_number": 0, "response": response}


def _message(item_id: str, text: str, phase: str | None = None) -> list[dict]:
    frames = [_added(item_id, "message", phase)]
    if text:
        frames.append(_delta(item_id, text))
    frames.append(_msg_done(item_id, text, phase))
    return frames


def _function_call(call_id: str) -> list[dict]:
    return [_added(call_id, "function_call"), _fc_done(call_id, call_id)]


def _stream(*items: list[dict], usage: dict | None = None) -> list[dict]:
    frames = [_RESPONSE_CREATED]
    for item in items:
        frames.extend(item)
    frames.append(_completed(usage))
    return frames


class _ReplayResponsesWS:
    _base_url = "https://api.openai.com/v1"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames

    def generate_response(self, payload: dict):  # noqa: ANN201
        frames = self._frames

        async def _replay():  # noqa: ANN202
            for frame in frames:
                yield frame

        return _replay()

    async def aclose(self) -> None:
        pass


class _SSETransport(httpx.AsyncBaseTransport):
    def __init__(self, frames: list[dict]) -> None:
        self._body = "".join(
            f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n" for frame in frames
        ).encode()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=self._body,
            request=request,
        )


@dataclass
class _Reply:
    text: str = ""
    calls: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    usage: agents_llm.CompletionUsage | None = None
    prev_resp_id: str = ""


async def _collect(llm_model: ResponsesLLM) -> _Reply:
    chat_ctx = agents_llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hi")

    reply = _Reply()
    try:
        async with llm_model.chat(
            chat_ctx=chat_ctx, conn_options=APIConnectOptions(timeout=5.0)
        ) as stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    reply.usage = chunk.usage
                if chunk.delta is None:
                    continue
                reply.text += chunk.delta.content or ""
                reply.calls.extend(tc.call_id for tc in chunk.delta.tool_calls)
                if chunk.delta.extra:
                    reply.phases.append(chunk.delta.extra["openai"]["phase"])
        reply.prev_resp_id = llm_model._prev_resp_id
    finally:
        await llm_model.aclose()
    return reply


async def _reply_over_ws(frames: list[dict]) -> _Reply:
    llm_model = ResponsesLLM(model="gpt-4.1", api_key="test-key")
    llm_model._ws = _ReplayResponsesWS(frames)  # type: ignore[assignment]
    return await _collect(llm_model)


async def test_single_message_is_unchanged() -> None:
    reply = await _reply_over_ws(_stream(_message("m1", "Hello.")))

    assert reply.text == "Hello."
    assert reply.calls == []


async def test_second_message_text_is_dropped() -> None:
    reply = await _reply_over_ws(_stream(_message("m1", "Hello."), _message("m2", "Hi again.")))

    assert reply.text == "Hello."


async def test_empty_first_message_yields_to_second() -> None:
    reply = await _reply_over_ws(_stream(_message("m1", ""), _message("m2", "Hello.")))

    assert reply.text == "Hello."


async def test_tool_calls_after_first_message_are_kept() -> None:
    reply = await _reply_over_ws(
        _stream(_message("m1", "Checking."), _function_call("fc1"), _function_call("fc2"))
    )

    assert reply.text == "Checking."
    assert reply.calls == ["fc1", "fc2"]


async def test_tool_call_between_messages_is_kept() -> None:
    reply = await _reply_over_ws(
        _stream(_message("m1", "Checking."), _function_call("fc1"), _message("m2", "Again."))
    )

    assert reply.text == "Checking."
    assert reply.calls == ["fc1"]


async def test_tool_call_after_second_message_is_dropped() -> None:
    reply = await _reply_over_ws(
        _stream(_message("m1", "Checking."), _message("m2", "Again."), _function_call("fc1"))
    )

    assert reply.text == "Checking."
    assert reply.calls == []


async def test_textless_second_message_keeps_later_tool_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(_message("m1", "A"), _message("m2", ""), _function_call("fc1"))
    )

    assert reply.text == "A"
    assert reply.calls == ["fc1"]
    assert not [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]


async def test_message_then_empty_then_text_message_drops_third(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(_message("m1", "A"), _message("m2", ""), _message("m3", "B"))
    )

    assert reply.text == "A"
    records = [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]
    assert len(records) == 1
    assert records[0].__dict__["dropped_item_ids"] == ["m3"]


async def test_commentary_owner_keeps_next_message_with_space() -> None:
    reply = await _reply_over_ws(
        _stream(
            _message("m1", "Let me check.", phase="commentary"),
            _message("m2", "It's sunny.", phase="final_answer"),
        )
    )

    assert reply.text == "Let me check. It's sunny."
    assert reply.phases == ["commentary", "final_answer"]


async def test_commentary_phase_only_on_done_still_joins() -> None:
    commentary = [
        _added("m1", "message"),
        _delta("m1", "Let me check."),
        _msg_done("m1", "Let me check.", phase="commentary"),
    ]
    reply = await _reply_over_ws(
        _stream(commentary, _message("m2", "It's sunny.", phase="final_answer"))
    )

    assert reply.text == "Let me check. It's sunny."


async def test_commentary_chain_joins_until_non_commentary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(
            _message("m1", "Let me check.", phase="commentary"),
            _message("m2", "Looking it up.", phase="commentary"),
            _message("m3", "It's sunny.", phase="final_answer"),
        )
    )

    assert reply.text == "Let me check. Looking it up. It's sunny."
    assert reply.phases == ["commentary", "commentary", "final_answer"]
    assert not [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]


async def test_final_answer_owner_after_commentary_drops_next(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(
            _message("m1", "Let me check.", phase="commentary"),
            _message("m2", "It's sunny.", phase="final_answer"),
            _message("m3", "It is sunny."),
        )
    )

    assert reply.text == "Let me check. It's sunny."
    records = [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]
    assert len(records) == 1
    assert records[0].__dict__["dropped_item_ids"] == ["m3"]


async def test_message_after_hosted_tool_call_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(
            _message("m1", "Let me look that up."),
            _web_search_call("ws_1"),
            _message("m2", "It's sunny."),
        )
    )

    assert reply.text == "Let me look that up. It's sunny."
    assert reply.prev_resp_id == "resp_1"
    assert not [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]


async def test_second_message_without_hosted_call_still_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    reply = await _reply_over_ws(
        _stream(_message("m1", "Let me look that up."), _message("m2", "It's sunny."))
    )

    assert reply.text == "Let me look that up."
    records = [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]
    assert len(records) == 1
    assert records[0].__dict__["dropped_item_ids"] == ["m2"]


async def test_dropped_message_phase_is_not_emitted() -> None:
    reply = await _reply_over_ws(
        _stream(
            _message("m1", "It's sunny.", phase="final_answer"),
            _message("m2", "It's warm.", phase="final_answer"),
        )
    )

    assert reply.text == "It's sunny."
    assert reply.phases == ["final_answer"]


async def test_usage_reported_after_drop() -> None:
    reply = await _reply_over_ws(
        _stream(_message("m1", "Hello."), _message("m2", "Hi again."), usage=_USAGE)
    )

    assert reply.usage is not None
    assert reply.usage.prompt_tokens == 120
    assert reply.usage.completion_tokens == 30
    assert reply.usage.total_tokens == 150
    assert reply.usage.prompt_cached_tokens == 64


async def test_drop_logs_one_warning_with_item_ids(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    await _reply_over_ws(_stream(_message("m1", "Hello."), _message("m2", "Hi again.")))

    records = [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]
    assert len(records) == 1
    assert records[0].__dict__["dropped_item_ids"] == ["m2"]
    assert records[0].__dict__["response_id"] == "resp_1"
    assert records[0].__dict__["model"] == "gpt-4.1"


async def test_single_message_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")

    await _reply_over_ws(_stream(_message("m1", "Hello.")))

    assert not [
        r for r in caplog.records if r.getMessage() == "dropped extra Responses output items"
    ]


async def test_drop_skips_previous_response_chaining() -> None:
    reply = await _reply_over_ws(
        _stream(_message("m1", "Checking."), _message("m2", "Again."), _function_call("fc1"))
    )

    assert reply.prev_resp_id == ""


async def test_kept_reply_chains_previous_response() -> None:
    reply = await _reply_over_ws(_stream(_message("m1", "Hello.")))

    assert reply.prev_resp_id == "resp_1"


def test_output_item_added_ws_frame_parses() -> None:
    parsed = LLMStream._parse_ws_event(  # type: ignore[arg-type]
        object(), _added("m1", "message", phase="commentary")
    )

    assert isinstance(parsed, ResponseOutputItemAddedEvent)
    assert isinstance(parsed.item, ResponseOutputMessage)
    assert parsed.item.id == "m1"
    assert parsed.item.phase == "commentary"


def test_output_item_added_tool_frame_is_not_parsed() -> None:
    # an in-progress web_search_call has no `action` yet, which the done event requires
    frame = {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
    }

    assert LLMStream._parse_ws_event(object(), frame) is None  # type: ignore[arg-type]


async def test_rest_stream_drops_second_message() -> None:
    frames = _stream(_message("m1", "Hello."), _message("m2", "Hi again."))
    client = openai.AsyncClient(
        api_key="k",
        base_url="https://api.test/v1",
        http_client=httpx.AsyncClient(transport=_SSETransport(frames)),
    )
    llm_model = ResponsesLLM(model="gpt-4.1", client=client, use_websocket=False)

    try:
        reply = await _collect(llm_model)
    finally:
        await client.close()

    assert reply.text == "Hello."
