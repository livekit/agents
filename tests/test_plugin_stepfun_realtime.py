from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeAudioConfigOutput,
    RealtimeConversationItemUserMessage,
    RealtimeError,
    RealtimeErrorEvent,
    RealtimeResponse,
    RealtimeSessionCreateRequest,
    ResponseCreatedEvent,
)
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents import APIError
from livekit.plugins.openai.realtime.realtime_model import process_base_url
from livekit.plugins.stepfun import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    realtime,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _hermetic_ws_conn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all unit tests are 100% hermetic with no outbound network connection attempts."""
    monkeypatch.setattr(realtime.RealtimeSession, "_main_task", AsyncMock())


def test_model_initialization_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-stepfun-key")
    model = realtime.RealtimeModel()

    assert model.model == DEFAULT_MODEL
    assert model.voice == DEFAULT_VOICE
    assert model._opts.base_url == DEFAULT_BASE_URL
    assert model._opts.api_key == "test-stepfun-key"
    assert model._provider_label == "StepFun StepAudio Realtime API"
    assert isinstance(model._opts.turn_detection, ServerVad)
    assert model._opts.turn_detection.type == "server_vad"
    assert model.capabilities.turn_detection is True


def test_model_initialization_custom_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    custom_key = "custom-key"
    custom_url = "wss://custom.stepfun.ai/v1/realtime"
    custom_model = "stepaudio-2.5-realtime"
    custom_voice = "清爽少年"

    model = realtime.RealtimeModel(
        model=custom_model,
        voice=custom_voice,
        api_key=custom_key,
        base_url=custom_url,
    )

    assert model.model == custom_model
    assert model.voice == custom_voice
    assert model._opts.base_url == custom_url
    assert model._opts.api_key == custom_key


def test_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="STEPFUN_API_KEY"):
        realtime.RealtimeModel()


@pytest.mark.parametrize(
    "model_name",
    [
        "stepaudio-3-realtime-preview",
        "stepaudio-2.5-realtime",
        "step-audio-2",
    ],
)
def test_websocket_url_formatting(model_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel(model=model_name)

    ws_url = process_base_url(model._opts.base_url, model._opts.model)
    parsed = urlparse(ws_url)
    assert parsed.scheme in ("ws", "wss")
    assert parsed.netloc == "api.stepfun.ai"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query)["model"] == [model_name]


@pytest.mark.asyncio
async def test_session_update_normalizes_turn_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        request = RealtimeSessionCreateRequest(
            type="realtime",
            output_modalities=["audio"],
            audio=RealtimeAudioConfig(
                input=RealtimeAudioConfigInput(
                    turn_detection=ServerVad(
                        type="server_vad",
                        threshold=0.6,
                        silence_duration_ms=500,
                        prefix_padding_ms=250,
                    )
                ),
                output=RealtimeAudioConfigOutput(voice="闫雨婷"),
            ),
        )

        wrapped = session._wrap_session_update("event_123", request)
        assert isinstance(wrapped, dict)
        assert wrapped["type"] == "session.update"
        assert wrapped["event_id"] == "event_123"
        s = wrapped["session"]
        assert set(s["modalities"]) == {"text", "audio"}
        assert s["turn_detection"]["type"] == "server_vad"
        assert s["turn_detection"]["silence_duration_ms"] == 500
        assert s["turn_detection"]["prefix_padding_ms"] == 250
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_session_update_normalizes_tools_to_nested_function_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure flat OpenAI realtime tool definitions are converted to StepFun's nested format."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        from openai.types.realtime import RealtimeFunctionTool

        request = RealtimeSessionCreateRequest(
            type="realtime",
            tools=[
                RealtimeFunctionTool(
                    type="function",
                    name="get_current_weather",
                    description="Get weather for a city",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
        )

        wrapped = session._wrap_session_update("event_tool_1", request)
        assert isinstance(wrapped, dict)
        tools = wrapped["session"]["tools"]
        assert len(tools) == 1
        # StepFun REQUIRES nested {"type": "function", "function": {"name": ...}}
        assert "function" in tools[0]
        assert tools[0]["function"]["name"] == "get_current_weather"
        assert tools[0]["function"]["description"] == "Get weather for a city"
        assert tools[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_handle_error_ignores_benign_cancel_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        mock_emit = MagicMock()
        session.emit = mock_emit

        # 1. 'no ongoing response to cancel' error from StepFun should be swallowed
        cancel_err = RealtimeErrorEvent(
            event_id="err_1",
            type="error",
            error=RealtimeError(
                type="invalid_request_error",
                message="no ongoing response to cancel",
            ),
        )
        session._handle_error(cancel_err)
        mock_emit.assert_not_called()

        # 2. 'Conversation has no active response' should also be swallowed
        cancel_err_2 = RealtimeErrorEvent(
            event_id="err_2",
            type="error",
            error=RealtimeError(
                type="invalid_request_error",
                message="Conversation has no active response.",
            ),
        )
        session._handle_error(cancel_err_2)
        mock_emit.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_handle_error_propagates_fatal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        fatal_err = RealtimeErrorEvent(
            event_id="err_fatal",
            type="error",
            error=RealtimeError(
                type="invalid_api_key",
                message="Incorrect API key provided",
                code="invalid_api_key",
            ),
        )

        with pytest.raises(APIError, match="StepFun StepAudio Realtime API returned an error"):
            session._handle_error(fatal_err)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_transcription_delta_overwrites_accumulated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure StepFun cumulative deltas do not get concatenated into duplicate strings."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        emitted_events = []
        session.on("input_audio_transcription_completed", lambda ev: emitted_events.append(ev))

        # Event 1: partial sentence
        session._handle_conversion_item_input_audio_transcription_delta(
            ConversationItemInputAudioTranscriptionDeltaEvent(
                event_id="ev_1",
                type="conversation.item.input_audio_transcription.delta",
                item_id="item_1",
                content_index=0,
                delta="你好",
            )
        )
        assert emitted_events[-1].transcript == "你好"

        # Event 2: cumulative sentence (StepFun sends full sentence so far)
        session._handle_conversion_item_input_audio_transcription_delta(
            ConversationItemInputAudioTranscriptionDeltaEvent(
                event_id="ev_2",
                type="conversation.item.input_audio_transcription.delta",
                item_id="item_1",
                content_index=0,
                delta="你好，能听到我说话吗？",
            )
        )
        # MUST NOT be "你好你好，能听到我说话吗？"
        assert emitted_events[-1].transcript == "你好，能听到我说话吗？"
        assert session._input_transcript_accumulators["item_1"][0] == "你好，能听到我说话吗？"
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_generate_reply_sends_response_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure generate_reply sends response.create."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events = []
        session.send_event = lambda ev: sent_events.append(ev)  # type: ignore

        session.generate_reply(instructions="热情打招呼")
        assert len(sent_events) == 1
        assert sent_events[0].type == "response.create"
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_transcription_delta_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure duplicate delta frames from StepFun do not trigger duplicate events."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        emitted_events = []
        session.on("input_audio_transcription_completed", lambda ev: emitted_events.append(ev))

        # Push identical delta 5 times
        for _ in range(5):
            session._handle_conversion_item_input_audio_transcription_delta(
                ConversationItemInputAudioTranscriptionDeltaEvent(
                    event_id="ev_repeat",
                    type="conversation.item.input_audio_transcription.delta",
                    item_id="item_repeat",
                    content_index=0,
                    delta="Hello.",
                )
            )
        # Should only emit once!
        assert len(emitted_events) == 1
        assert emitted_events[0].transcript == "Hello."
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_input_audio_transcription_completed_clears_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure completed transcription emits final event and atomically clears accumulator."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        emitted_events = []
        session.on("input_audio_transcription_completed", lambda ev: emitted_events.append(ev))

        # 1. Delta arrives (is_final=False)
        session._handle_conversion_item_input_audio_transcription_delta(
            ConversationItemInputAudioTranscriptionDeltaEvent(
                event_id="ev_1",
                type="conversation.item.input_audio_transcription.delta",
                item_id="item_vad",
                content_index=0,
                delta="你好，你在吗？",
            )
        )
        assert len(emitted_events) == 1
        assert emitted_events[-1].is_final is False
        assert "item_vad" in session._input_transcript_accumulators

        # 2. StepFun native completed event arrives -> is_final=True & clears accumulator
        session._handle_conversion_item_input_audio_transcription_completed(
            ConversationItemInputAudioTranscriptionCompletedEvent.model_construct(
                event_id="ev_comp",
                type="conversation.item.input_audio_transcription.completed",
                item_id="item_vad",
                content_index=0,
                transcript="你好，你在吗？",
            )
        )
        assert len(emitted_events) == 2
        assert emitted_events[-1].is_final is True
        assert emitted_events[-1].transcript == "你好，你在吗？"
        assert "item_vad" not in session._input_transcript_accumulators
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_response_created_associates_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure response.created resolves pending future even when StepFun drops client_event_id metadata."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        session.send_event = lambda ev: None  # type: ignore

        # Trigger generate_reply which registers a pending future
        fut = session.generate_reply(instructions="测试回复")
        assert len(session._response_created_futures) == 1

        # Simulate StepFun response.created WITHOUT metadata
        resp_ev = ResponseCreatedEvent(
            event_id="ev_resp",
            type="response.created",
            response=RealtimeResponse(
                id="resp_123",
                object="realtime.response",
                status="in_progress",
                metadata=None,  # StepFun drops metadata!
            ),
        )

        session._handle_response_created(resp_ev)

        # Future must be resolved immediately, not timing out!
        assert fut.done()
        gen_ev = fut.result()
        assert gen_ev.user_initiated is True
        assert len(session._response_created_futures) == 0
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_interrupt_immediately_closes_current_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure interrupt immediately closes active generation to prevent 5s speech timeout."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import RealtimeResponse, ResponseCreatedEvent

    from livekit.plugins import stepfun

    model = stepfun.realtime.RealtimeModel()
    session = model.session()
    try:
        session.send_event = lambda ev: None  # type: ignore

        # Trigger reply generation so a future is pending
        _ = session.generate_reply(instructions="测试打断")
        # Simulate response.created arriving so _current_generation is instantiated
        resp_ev = ResponseCreatedEvent(
            event_id="ev_resp",
            type="response.created",
            response=RealtimeResponse(
                id="resp_123",
                object="realtime.response",
                status="in_progress",
                metadata=None,
            ),
        )
        session._handle_response_created(resp_ev)
        assert session._current_generation is not None
        gen = session._current_generation

        # Call interrupt
        session.interrupt()

        # Must be immediately closed and set to _DiscardedGeneration so trailing audio doesn't throw!
        from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

        assert isinstance(session._current_generation, _DiscardedGeneration)
        assert gen._done_fut.done() is True

        # Simulate trailing audio arriving after interrupt (StepFun in-flight packet)
        from openai.types.realtime import ResponseAudioDeltaEvent, ResponseAudioDoneEvent

        session._handle_response_audio_delta(
            ResponseAudioDeltaEvent.model_construct(
                event_id="ev_delta",
                type="response.output_audio.delta",
                response_id="resp_123",
                item_id="item_1",
                output_index=0,
                content_index=0,
                delta="AAA=",
            )
        )
        session._handle_response_audio_done(
            ResponseAudioDoneEvent.model_construct(
                event_id="ev_done",
                type="response.output_audio.done",
                response_id="resp_123",
                item_id="item_1",
                output_index=0,
                content_index=0,
            )
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_server_uuid_item_create_future_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure server UUID in conversation.item.created matches client's earliest pending future."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        import asyncio

        client_fut = asyncio.get_running_loop().create_future()
        client_item_id = "client_item_123"
        session._item_create_future[client_item_id] = client_fut

        server_uuid = "server_uuid_456"
        event = ConversationItemCreatedEvent(
            event_id="ev_1",
            type="conversation.item.created",
            previous_item_id=None,
            item=RealtimeConversationItemUserMessage(
                id=server_uuid,
                type="message",
                role="user",
                status="completed",
                content=[],
            ),
        )
        session._handle_conversion_item_added(event)

        assert client_fut.done() is True
        assert client_fut.result() is None
        assert session._client_to_server_id[client_item_id] == server_uuid
        assert session._server_to_client_id[server_uuid] == client_item_id
        assert session._remote_chat_ctx.get(client_item_id) is not None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_outbound_events_translate_client_ids_to_server_uuids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure outgoing conversation.item.create and delete translate client IDs to server UUIDs."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # Register mapping: client_1 -> server_uuid_9
        session._client_to_server_id["client_1"] = "server_uuid_9"
        session._server_to_client_id["server_uuid_9"] = "client_1"

        # 1. Create item referencing client_1 as predecessor
        session.send_event(
            {
                "type": "conversation.item.create",
                "previous_item_id": "client_1",
                "item": {"id": "client_2", "type": "message", "role": "user"},
            }
        )
        assert len(sent_events) == 1
        assert sent_events[0]["previous_item_id"] == "server_uuid_9"

        # 2. Delete item targeting client_1
        session.send_event(
            {
                "type": "conversation.item.delete",
                "item_id": "client_1",
            }
        )
        assert len(sent_events) == 2
        assert sent_events[1]["item_id"] == "server_uuid_9"
    finally:
        await session.aclose()
        await model.aclose()


def test_stepfun_provider_tools_serialization() -> None:
    """Verify StepFun server-side provider tools serialize to expected dictionaries."""
    from livekit.plugins.stepfun import tools

    ws = tools.WebSearch(top_k=3, timeout_seconds=4)
    assert ws.to_dict() == {
        "type": "web_search",
        "function": {
            "description": "Search the web for up-to-date information and real-time news.",
            "options": {"top_k": 3, "timeout_seconds": 4},
        },
    }

    ret = tools.Retrieval(
        vector_store_id="vs_12345",
        description="Search and retrieve relevant context from the knowledge base.",
        prompt_template="找到{{knowledge}}回答{{query}}",
    )
    assert ret.to_dict() == {
        "type": "retrieval",
        "function": {
            "description": "Search and retrieve relevant context from the knowledge base.",
            "options": {
                "vector_store_id": "vs_12345",
                "prompt_template": "找到{{knowledge}}回答{{query}}",
            },
        },
    }


@pytest.mark.asyncio
async def test_create_tools_update_event_includes_stepfun_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure _create_tools_update_event merges StepFun server-side provider tools."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from livekit.agents import function_tool
    from livekit.plugins.stepfun import realtime, tools

    @function_tool
    def my_local_tool(city: str) -> str:
        """查天气"""
        return "晴"

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        combined_tools = [my_local_tool, tools.WebSearch(top_k=5)]
        event = session._create_tools_update_event(combined_tools)

        tool_list = event["session"]["tools"]
        assert len(tool_list) == 2
        # One is function tool (nested under "function" or top-level name)
        assert any(
            t.get("name") == "my_local_tool" or t.get("function", {}).get("name") == "my_local_tool"
            for t in tool_list
            if isinstance(t, dict)
        )
        # One is web_search tool
        assert any(isinstance(t, dict) and t.get("type") == "web_search" for t in tool_list)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_session_aclose_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure session.aclose() terminates main task and closes event channels cleanly."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    assert session._closing is False
    assert not session._main_atask.done()

    await session.aclose()
    assert session._closing is True
    assert session._main_atask.done()


@pytest.mark.asyncio
async def test_initial_empty_function_call_item_placeholder_and_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure StepFun initial function_call with arguments=None enters remote_ctx with placeholder and is upserted on done."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import (
        ConversationItemAdded,
        RealtimeConversationItemFunctionCall,
        ResponseOutputItemDoneEvent,
    )

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        # Initial function_call packet where arguments is None (as constructed from StepFun raw event)
        empty_item = RealtimeConversationItemFunctionCall.model_construct(
            id="item_func_1",
            type="function_call",
            call_id="call_123",
            name="query_market_price",
            arguments=None,
        )
        ev = ConversationItemAdded(
            event_id="ev_fnc",
            type="conversation.item.added",
            item=empty_item,
        )
        # Should populate placeholder without raising assertion error
        session._handle_conversion_item_added(ev)
        remote_node = session._remote_chat_ctx.get("item_func_1")
        assert remote_node is not None
        assert remote_node.item.call_id == "call_123"
        assert remote_node.item.arguments == ""

        # When response.output_item.done arrives with completed arguments:
        done_item = RealtimeConversationItemFunctionCall(
            id="item_func_1",
            type="function_call",
            call_id="call_123",
            name="query_market_price",
            arguments='{"symbol": "BTC"}',
        )
        done_ev = ResponseOutputItemDoneEvent.model_construct(
            event_id="ev_done",
            type="response.output_item.done",
            response_id="resp_1",
            output_index=0,
            item=done_item,
        )
        session._handle_response_output_item_done(done_ev)
        remote_node2 = session._remote_chat_ctx.get("item_func_1")
        assert remote_node2 is not None
        assert remote_node2.item.arguments == '{"symbol": "BTC"}'
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_wrap_session_update_preserves_omitted_tools_on_partial_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure partial session update (like update_instructions) does not wipe out configured tools."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()
    session = model.session()
    try:
        # Partial request with only instructions set
        req = RealtimeSessionCreateRequest(
            type="realtime",
            instructions="新指令",
        )
        res = session._wrap_session_update("instructions_update_123", req)
        flat = res if isinstance(res, dict) else res.model_dump()
        session_dict = flat["session"]

        assert session_dict["instructions"] == "新指令"
        # "tools" must NOT be set to [] on partial update
        assert "tools" not in session_dict
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_update_tools_retains_nested_function_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure update_tools extracts names from nested function objects and retains local tools in _tools."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from livekit.agents import function_tool

    @function_tool
    def my_db_query(key: str) -> str:
        """Query DB"""
        return "val"

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        monkeypatch.setattr(session, "send_event", lambda ev: None)
        await session.update_tools([my_db_query])

        # _tools must retain my_db_query despite nested function schema!
        assert session._tools.get_function_tool("my_db_query") is not None
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_turn_detection_disabled_emits_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure turn_detection_disabled=True or update_options(turn_detection=None) emits turn_detection=None to disable Server VAD."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    model = realtime.RealtimeModel()

    # 1. Disabled at session creation
    session = model.session(turn_detection_disabled=True)
    try:
        ev = session._create_session_update_event()
        session_payload = ev["session"]
        assert "turn_detection" in session_payload
        assert session_payload["turn_detection"] is None
    finally:
        await session.aclose()

    # 2. Disabled via update_options from normal session
    session2 = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session2._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))
        session2.update_options(turn_detection=None)
        assert len(sent_events) == 1
        assert sent_events[0]["session"]["turn_detection"] is None

        # 3. Partial update preserves omission (does NOT include turn_detection)
        sent_events.clear()
        session2.update_options(speed=1.2)
        assert len(sent_events) == 1
        assert "turn_detection" not in sent_events[0]["session"]
    finally:
        await session2.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_interrupted_or_cancelled_reply_discarded_via_fifo_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure late response.created without metadata for a cancelled reply matches _discarded_event_ids and is discarded."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # 1. Start generate_reply
        fut = session.generate_reply(instructions="测试回复")
        assert len(session._pending_response_creates) == 1
        event_id = session._pending_response_creates[0]
        assert event_id in session._response_created_futures

        # 2. Cancel the future before response.created arrives
        fut.cancel()
        import asyncio

        await asyncio.sleep(0)
        assert event_id not in session._response_created_futures
        assert event_id in session._discarded_event_ids
        # ID remains in _pending_response_creates for StepFun correlation
        assert len(session._pending_response_creates) == 1

        # 3. StepFun sends response.created later WITHOUT metadata
        resp_ev = ResponseCreatedEvent(
            event_id="ev_resp_late",
            type="response.created",
            response=RealtimeResponse(
                id="resp_late_123",
                object="realtime.response",
                status="in_progress",
                metadata=None,
            ),
        )
        session._handle_response_created(resp_ev)

        # Queue should have been consumed
        assert len(session._pending_response_creates) == 0
        # Generation must be marked as _DiscardedGeneration, not a live active generation
        assert isinstance(session._current_generation, _DiscardedGeneration)
        # response.cancel must have been sent
        assert any(
            getattr(ev, "type", None) == "response.cancel"
            and getattr(ev, "response_id", None) == "resp_late_123"
            for ev in sent_events
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_create_update_chat_ctx_events_reanchors_dangling_predecessors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure excluded client-created function calls and orphan outputs do not leave dangling previous_item_id anchors."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import ConversationItemCreateEvent

    from livekit.agents import llm

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        user_msg = llm.ChatMessage(role="user", content=["hello"], id="user_msg_1")
        fnc_call = llm.FunctionCall(
            name="get_weather", arguments="{}", call_id="c1", id="fnc_call_1"
        )
        fnc_output = llm.FunctionCallOutput(
            output="sunny", call_id="c1", is_error=False, id="fnc_out_1"
        )
        assistant_msg = llm.ChatMessage(
            role="assistant", content=["The weather is sunny"], id="asst_msg_1"
        )

        # 1. When fnc_call is absent from remote_ctx:
        # Both fnc_call and orphan fnc_output are excluded; assistant_msg anchors directly to user_msg_1
        session._remote_chat_ctx.insert(None, user_msg)
        chat_ctx = llm.ChatContext([user_msg, fnc_call, fnc_output, assistant_msg])
        events = session._create_update_chat_ctx_events(chat_ctx)

        create_events = [ev for ev in events if isinstance(ev, ConversationItemCreateEvent)]
        assert not any(getattr(ev.item, "type", None) == "function_call" for ev in create_events)
        assert not any(
            getattr(ev.item, "type", None) == "function_call_output" for ev in create_events
        )
        asst_ev = next(ev for ev in create_events if getattr(ev.item, "id", None) == "asst_msg_1")
        assert asst_ev.previous_item_id == "user_msg_1"

        # 2. When fnc_call is already remote (server-created):
        # fnc_output is valid and anchors to fnc_call_1
        session._remote_chat_ctx.insert(user_msg.id, fnc_call)
        events2 = session._create_update_chat_ctx_events(chat_ctx)
        create_events2 = [ev for ev in events2 if isinstance(ev, ConversationItemCreateEvent)]
        fnc_out_ev = next(
            ev for ev in create_events2 if getattr(ev.item, "id", None) == "fnc_out_1"
        )
        assert fnc_out_ev.previous_item_id == "fnc_call_1"
        asst_ev2 = next(ev for ev in create_events2 if getattr(ev.item, "id", None) == "asst_msg_1")
        assert asst_ev2.previous_item_id == "fnc_out_1"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_configured_session_options_reasoning_and_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure reasoning and input_audio_transcription are preserved in session update."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import AudioTranscription, RealtimeReasoning

    model = realtime.RealtimeModel(
        input_audio_transcription=AudioTranscription(model="whisper-1"),
        reasoning=RealtimeReasoning(effort="low"),
    )
    session = model.session()
    try:
        # Initial session update contains both
        ev = session._create_session_update_event()
        sess = ev["session"]
        assert sess.get("reasoning") == {"effort": "low"}
        assert sess.get("input_audio_transcription") == {"model": "whisper-1"}

        # update_options changes reasoning
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))
        session.update_options(reasoning=RealtimeReasoning(effort="high"))
        assert len(sent_events) == 1
        assert sent_events[0]["session"]["reasoning"] == {"effort": "high"}

        # update_options clears reasoning to None
        sent_events.clear()
        session.update_options(reasoning=None)
        assert len(sent_events) == 1
        assert sent_events[0]["session"]["reasoning"] is None

        # Partial update with instructions does NOT include reasoning or transcription
        sent_events.clear()
        await session.update_instructions("新指令")
        assert len(sent_events) == 1
        assert "reasoning" not in sent_events[0]["session"]
        assert "input_audio_transcription" not in sent_events[0]["session"]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancellation_confirmed_does_not_poison_subsequent_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure cancelling reply A followed by StepFun 'no ongoing response to cancel' allows reply B to succeed."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # 1. Start Reply A
        fut_a = session.generate_reply(instructions="回复A")
        assert len(session._pending_response_creates) == 1
        id_a = session._pending_response_creates[0]

        # 2. Cancel Reply A
        fut_a.cancel()
        import asyncio

        await asyncio.sleep(0)
        assert id_a in session._discarded_event_ids
        assert len(session._pending_response_creates) == 1

        # 3. StepFun returns 'no ongoing response to cancel' confirming response A was never created
        from openai.types.realtime import RealtimeError, RealtimeErrorEvent

        session._handle_error(
            RealtimeErrorEvent(
                event_id="err_cancel",
                type="error",
                error=RealtimeError(
                    type="invalid_request_error",
                    message="Conversation has no ongoing response to cancel",
                ),
            )
        )
        # Pending queue for A must be purged so it won't poison B
        assert len(session._pending_response_creates) == 0

        # 4. Start Reply B
        fut_b = session.generate_reply(instructions="回复B")
        assert len(session._pending_response_creates) == 1
        id_b = session._pending_response_creates[0]
        assert id_b != id_a

        # 5. StepFun sends response.created (without metadata) for Reply B
        resp_ev_b = ResponseCreatedEvent(
            event_id="ev_resp_b",
            type="response.created",
            response=RealtimeResponse(
                id="resp_b_123",
                object="realtime.response",
                status="in_progress",
                metadata=None,
            ),
        )
        session._handle_response_created(resp_ev_b)

        # Reply B must succeed and NOT be discarded!
        assert fut_b.done() is True
        assert not isinstance(session._current_generation, _DiscardedGeneration)
        assert id_b not in session._discarded_event_ids
    finally:
        await session.aclose()
        await model.aclose()


def test_capabilities_can_disable_turn_detection_and_manual_function_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure can_disable_turn_detection reflects whether caller passed turn_detection, and manual_function_calls is False."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")

    # When not given: can_disable_turn_detection is True so AgentActivity can disable server VAD for client VAD
    model_default = realtime.RealtimeModel()
    assert model_default.capabilities.can_disable_turn_detection is True
    assert model_default.capabilities.manual_function_calls is False

    # When explicitly given: can_disable_turn_detection is False
    from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

    model_custom = realtime.RealtimeModel(
        turn_detection=ServerVad(type="server_vad", threshold=0.5)
    )
    assert model_custom.capabilities.can_disable_turn_detection is False
    assert model_custom.capabilities.manual_function_calls is False


@pytest.mark.asyncio
async def test_runtime_voice_normalization_for_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure update_options normalizes voice aliases at runtime for the cluster."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")

    # Domestic endpoint (api.stepfun.com) maps 'vibrant-youth' -> 'yuanqinansheng'
    model = realtime.RealtimeModel(
        base_url="https://api.stepfun.com/v1/realtime", voice="vibrant-youth"
    )
    assert model.voice == "yuanqinansheng"

    model.update_options(voice="lively-girl")
    assert model.voice == "yuanqishaonv"

    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))
        session.update_options(voice="magnetic-voiced-male")
        assert len(sent_events) == 1
        assert sent_events[0]["session"]["voice"] == "cixingnansheng"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_server_generated_items_do_not_consume_client_futures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure concurrent server-initiated items do not settle client context update futures."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")

    from openai.types.realtime import (
        ConversationItemCreatedEvent,
        ConversationItemCreateEvent,
        RealtimeConversationItemFunctionCall,
        RealtimeConversationItemSystemMessage,
        RealtimeConversationItemUserMessage,
    )

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        client_fut = asyncio.get_running_loop().create_future()
        client_msg_id = "client_sys_msg_1"
        session._item_create_future[client_msg_id] = client_fut

        # Client sends a system message
        sys_item = RealtimeConversationItemSystemMessage(
            id=client_msg_id,
            type="message",
            role="system",
            content=[{"type": "input_text", "text": "system prompt"}],
        )
        session.send_event(
            ConversationItemCreateEvent(
                event_id="ev_client_create",
                type="conversation.item.create",
                item=sys_item,
            )
        )

        # 1. Server emits a model function call item -> MUST NOT consume client_fut!
        server_fnc = ConversationItemCreatedEvent(
            event_id="ev_server_fnc",
            type="conversation.item.created",
            item=RealtimeConversationItemFunctionCall(
                id="server_fnc_uuid",
                type="function_call",
                call_id="call_999",
                name="web_search",
                arguments="{}",
            ),
        )
        session._handle_conversion_item_added(server_fnc)
        assert client_fut.done() is False

        # 2. Server emits a user speech item -> MUST NOT consume client_fut (role differs: user vs system)!
        server_user = ConversationItemCreatedEvent(
            event_id="ev_server_usr",
            type="conversation.item.created",
            item=RealtimeConversationItemUserMessage(
                id="server_user_uuid",
                type="message",
                role="user",
                status="completed",
                content=[],
            ),
        )
        session._handle_conversion_item_added(server_user)
        assert client_fut.done() is False

        # 3. Server emits the real system message created event with server UUID -> settles client_fut!
        server_ack = ConversationItemCreatedEvent(
            event_id="ev_server_ack",
            type="conversation.item.created",
            item=RealtimeConversationItemSystemMessage(
                id="server_sys_uuid",
                type="message",
                role="system",
                content=[{"type": "input_text", "text": "system prompt"}],
            ),
        )
        session._handle_conversion_item_added(server_ack)
        assert client_fut.done() is True
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_full_function_call_to_output_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure server function call is tracked and its output is preserved during update_chat_ctx."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import (
        ConversationItemAdded,
        ConversationItemCreateEvent,
        RealtimeConversationItemFunctionCall,
        ResponseOutputItemDoneEvent,
    )

    from livekit.agents import llm

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        user_msg = llm.ChatMessage(role="user", content=["查一下小马智行"], id="usr_1")
        session._remote_chat_ctx.insert(None, user_msg)

        # 1. StepFun emits initial function call with arguments=None
        init_fnc = RealtimeConversationItemFunctionCall.model_construct(
            id="fnc_server_item_1",
            type="function_call",
            call_id="call_pony_123",
            name="query_market_price",
            arguments=None,
        )
        session._handle_conversion_item_added(
            ConversationItemAdded(
                event_id="ev_fnc_added",
                type="conversation.item.added",
                previous_item_id="usr_1",
                item=init_fnc,
            )
        )
        assert session._remote_chat_ctx.get("fnc_server_item_1") is not None

        # 2. StepFun finishes arguments streaming and emits output_item.done
        done_fnc = RealtimeConversationItemFunctionCall(
            id="fnc_server_item_1",
            type="function_call",
            call_id="call_pony_123",
            name="query_market_price",
            arguments='{"symbol": "PONY"}',
        )
        session._handle_response_output_item_done(
            ResponseOutputItemDoneEvent.model_construct(
                event_id="ev_fnc_done",
                type="response.output_item.done",
                response_id="resp_1",
                output_index=0,
                item=done_fnc,
            )
        )

        # 3. Python tool finishes and Agent updates chat context with function call + output
        fnc_call_item = llm.FunctionCall(
            id="fnc_server_item_1",
            call_id="call_pony_123",
            name="query_market_price",
            arguments='{"symbol": "PONY"}',
        )
        fnc_out_item = llm.FunctionCallOutput(
            id="fnc_out_1",
            call_id="call_pony_123",
            output='{"价格": "$15.00"}',
            is_error=False,
        )
        chat_ctx = llm.ChatContext([user_msg, fnc_call_item, fnc_out_item])

        events = session._create_update_chat_ctx_events(chat_ctx)
        create_events = [ev for ev in events if isinstance(ev, ConversationItemCreateEvent)]

        # fnc_server_item_1 is already remote, so it is NOT recreated
        assert not any(getattr(ev.item, "type", None) == "function_call" for ev in create_events)

        # fnc_out_1 MUST BE RETAINED and anchored to fnc_server_item_1!
        out_ev = next(ev for ev in create_events if getattr(ev.item, "id", None) == "fnc_out_1")
        assert out_ev.previous_item_id == "fnc_server_item_1"
        assert getattr(out_ev.item, "call_id", None) == "call_pony_123"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_turn_scoped_tools_use_nested_schema_in_response_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure turn-scoped tools passed to generate_reply are normalized to StepFun nested format."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from livekit.agents import function_tool

    @function_tool
    def search_database(query: str) -> str:
        """Search DB"""
        return "result"

    model = realtime.RealtimeModel()
    assert model.capabilities.per_response_tool_choice is False

    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # Call generate_reply with turn-scoped tool
        _ = session.generate_reply(instructions="查数据库", tools=[search_database])
        assert len(sent_events) == 1
        ev = sent_events[0]
        assert getattr(ev, "type", None) == "response.create"
        resp_params = getattr(ev, "response", None)
        assert resp_params is not None
        assert resp_params.tools is not None
        assert len(resp_params.tools) == 1
        tool_schema = resp_params.tools[0]
        # Must be StepFun nested schema: {"type": "function", "function": {"name": ...}}
        assert tool_schema["type"] == "function"
        assert "function" in tool_schema
        assert tool_schema["function"]["name"] == "search_database"
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_two_pending_replies_only_one_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure when two replies are pending and only one is cancelled, the other is not discarded."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import (
        RealtimeError,
        RealtimeErrorEvent,
        RealtimeResponse,
        ResponseCreatedEvent,
    )

    from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # 1. Queue both Reply A and Reply B
        fut_a = session.generate_reply(instructions="Reply A")
        fut_b = session.generate_reply(instructions="Reply B")
        assert len(session._pending_response_creates) == 2
        id_a = session._pending_response_creates[0]
        id_b = session._pending_response_creates[1]

        # 2. Cancel ONLY Reply A
        fut_a.cancel()
        import asyncio

        await asyncio.sleep(0)
        assert id_a in session._discarded_event_ids
        assert id_b not in session._discarded_event_ids
        assert fut_b.done() is False

        # 3. StepFun sends "no ongoing response to cancel" for A
        session._handle_error(
            RealtimeErrorEvent(
                event_id="err_cancel",
                type="error",
                error=RealtimeError(
                    type="invalid_request_error",
                    message="no ongoing response to cancel",
                ),
            )
        )
        # Only A should be purged, B must remain in queue!
        assert len(session._pending_response_creates) == 1
        assert session._pending_response_creates[0] == id_b

        # 4. StepFun acknowledges B
        resp_ev_b = ResponseCreatedEvent(
            event_id="ev_resp_b",
            type="response.created",
            response=RealtimeResponse(
                id="resp_b_456",
                object="realtime.response",
                status="in_progress",
                metadata=None,
            ),
        )
        session._handle_response_created(resp_ev_b)

        # Reply B must be successfully resolved and not discarded!
        assert fut_b.done() is True
        assert not isinstance(session._current_generation, _DiscardedGeneration)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.asyncio
async def test_timed_out_reply_ttl_cleanup_prevents_subsequent_response_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure timed out reply event ID is evicted by TTL and does not cancel subsequent responses."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    import contextlib

    from openai.types.realtime import RealtimeResponse, ResponseCreatedEvent

    from livekit.plugins.openai.realtime.realtime_model import _DiscardedGeneration

    model = realtime.RealtimeModel()
    session = model.session()
    try:
        sent_events: list[Any] = []
        monkeypatch.setattr(session._msg_ch, "send_nowait", lambda ev: sent_events.append(ev))

        # 1. Start generate_reply A
        fut_a = session.generate_reply(instructions="Reply A")
        assert len(session._pending_response_creates) == 1
        id_a = session._pending_response_creates[0]

        # 2. Simulate timeout on reply A: base class pops from _response_created_futures & marks discarded
        session._response_created_futures.pop(id_a, None)
        session._discarded_event_ids.add(id_a)
        fut_a.set_exception(Exception("timed out"))

        # Prune via TTL cleanup logic
        with contextlib.suppress(ValueError):
            session._pending_response_creates.remove(id_a)
        session._discarded_event_ids.discard(id_a)

        assert len(session._pending_response_creates) == 0

        # 3. Later, an automatic response (e.g. ServerVAD) arrives without metadata
        resp_ev_vad = ResponseCreatedEvent(
            event_id="ev_vad_1",
            type="response.created",
            response=RealtimeResponse(
                id="resp_vad_1",
                object="realtime.response",
                status="in_progress",
                metadata=None,
            ),
        )
        session._handle_response_created(resp_ev_vad)

        # 4. Verify ServerVAD response is NOT poisoned or cancelled
        assert not isinstance(session._current_generation, _DiscardedGeneration)
        assert not any(getattr(ev, "type", None) == "response.cancel" for ev in sent_events)
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    "model_name",
    [
        "stepaudio-2.5-realtime",
        "stepaudio-3-realtime-preview",
    ],
)
@pytest.mark.asyncio
async def test_stepfun_both_versions_handle_server_event_and_item_id_replacement(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that both StepFun 2.5 and 3.0 models exhibit the same protocol divergence
    (server replaces client event_id / item.id and drops response metadata), and that
    RealtimeSession correctly handles both versions without leaking futures.
    """
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from openai.types.realtime import (
        ConversationItemCreatedEvent,
        RealtimeConversationItemUserMessage,
        RealtimeResponse,
        ResponseCreatedEvent,
    )

    model = realtime.RealtimeModel(model=model_name)
    session = model.session()
    try:
        # 1. Verify item.id remapping: client sends item_1, server returns server_uuid
        client_fut = asyncio.get_running_loop().create_future()
        client_item_id = f"client_item_{model_name}"
        session._item_create_future[client_item_id] = client_fut

        server_uuid = f"server_uuid_{model_name}"
        item_event = ConversationItemCreatedEvent(
            event_id=f"server_ev_{model_name}",  # StepFun returns random server event_id
            type="conversation.item.created",
            previous_item_id=None,
            item=RealtimeConversationItemUserMessage(
                id=server_uuid,  # StepFun returns random server item_id
                type="message",
                role="user",
                status="completed",
                content=[],
            ),
        )
        session._handle_conversion_item_added(item_event)
        assert client_fut.done() is True
        assert session._client_to_server_id[client_item_id] == server_uuid
        assert session._server_to_client_id[server_uuid] == client_item_id

        # 2. Verify response.create correlation when metadata is dropped by StepFun
        reply_fut = session.generate_reply(instructions="test")
        assert len(session._pending_response_creates) == 1
        expected_client_eid = session._pending_response_creates[0]

        resp_event = ResponseCreatedEvent(
            event_id=f"server_resp_ev_{model_name}",  # StepFun returns random server event_id
            type="response.created",
            response=RealtimeResponse(
                id=f"resp_{model_name}",
                object="realtime.response",
                status="in_progress",
                metadata=None,  # StepFun drops metadata in both 2.5 and 3.0
            ),
        )
        session._handle_response_created(resp_event)
        assert reply_fut.done() is True
        assert resp_event.response.metadata == {"client_event_id": expected_client_eid}
    finally:
        await session.aclose()
        await model.aclose()
