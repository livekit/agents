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
