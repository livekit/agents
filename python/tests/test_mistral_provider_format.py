"""Conversation entries satisfy the SDK's required input contract without SDK defaults."""

import pytest
from mistralai.client.models import InputEntriesTypedDict
from pydantic import TypeAdapter

from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput

pytestmark = pytest.mark.unit


async def test_conversation_history_is_accepted_as_sdk_input_entries() -> None:
    context = ChatContext.empty()
    context.add_message(role="system", content="Be helpful.")
    context.add_message(role="user", content="Find my booking.")
    context.insert(FunctionCall(call_id="lookup", name="lookup", arguments="{}"))
    context.insert(
        FunctionCallOutput(call_id="lookup", name="lookup", output="found", is_error=False)
    )
    context.add_message(role="assistant", content="I found it.")

    entries, data = await context.to_provider_format("mistralai")
    accepted = TypeAdapter(list[InputEntriesTypedDict]).validate_python(entries)
    assert accepted == entries
    assert data.instructions == "Be helpful."


async def test_chat_mode_sends_developer_instructions_as_system_messages() -> None:
    from types import SimpleNamespace
    from typing import Any

    from livekit.agents import APIConnectOptions
    from livekit.plugins.mistralai.llm import LLM, ApiMode

    sent: list[dict[str, Any]] = []

    async def stream_async(**kwargs: Any) -> Any:
        sent.extend(message.model_dump(exclude_unset=True) for message in kwargs["messages"])

        async def events() -> Any:
            for event in ():
                yield event

        return events()

    client = SimpleNamespace(chat=SimpleNamespace(stream_async=stream_async))
    model = LLM(client=client, model="test", api_mode=ApiMode.CHAT_COMPLETIONS)
    context = ChatContext.empty()
    context.add_message(role="developer", content="Follow these instructions.")
    context.add_message(role="user", content="Hello.")
    try:
        async with model.chat(
            chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)
        ) as stream:
            async for _ in stream:
                pass
    finally:
        await model.aclose()
    assert sent == [
        {"role": "system", "content": "Follow these instructions."},
        {"role": "user", "content": "Hello."},
    ]
