import pytest

from livekit.agents import llm
from livekit.plugins.openai import responses


@pytest.mark.asyncio
async def test_previous_response_id_filters_only_pending_function_calls() -> None:
    openai_llm = responses.LLM(api_key="test", use_websocket=False)
    prev_message = llm.ChatMessage(role="user", content=["check the weather"])
    openai_llm._prev_chat_ctx = llm.ChatContext(items=[prev_message])
    openai_llm._prev_resp_id = "resp_123"
    openai_llm._pending_tool_calls = {"call_pending"}

    duplicate_call = llm.FunctionCall(call_id="call_pending", name="get_weather", arguments="{}")
    tool_output = llm.FunctionCallOutput(
        call_id="call_pending", name="get_weather", output="sunny", is_error=False
    )
    new_call = llm.FunctionCall(call_id="call_new", name="get_time", arguments="{}")
    current_ctx = llm.ChatContext(items=[prev_message, duplicate_call, tool_output, new_call])

    stream = openai_llm.chat(chat_ctx=current_ctx)
    try:
        assert stream._extra_kwargs["previous_response_id"] == "resp_123"
        assert stream._chat_ctx.items == [tool_output, new_call]
    finally:
        await stream.aclose()
        await openai_llm.aclose()
