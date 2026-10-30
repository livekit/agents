"""ToolError log bodies must not carry argument values (#7801).

Argument-preparation failures put pydantic / raw JSON detail into the ToolError
message returned to the model. That detail is fine for the model; the warning
that logs the same failure must keep it under ``lk.pii.*`` because a log body
cannot be redacted.
"""

from __future__ import annotations

import json
import logging

import pytest

from livekit.agents import Agent, AgentSession, function_tool, llm

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

SECRET = "jane.doe@example.com"


@function_tool
async def lookup_orders(count: int) -> str:
    """Look up the customer's latest orders.

    Args:
        count: how many orders to return
    """
    return "ok"


@pytest.mark.parametrize(
    "arguments",
    [
        json.dumps({"count": SECRET}),  # valid JSON, wrong type
        '{"count": "' + SECRET,  # truncated JSON (repaired, then fails validation)
    ],
    ids=["wrong_type", "truncated_json"],
)
async def test_toolerror_warning_keeps_argument_values_out_of_log_body(
    arguments: str, caplog: pytest.LogCaptureFixture
) -> None:
    llm_ = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="show my orders",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="lookup_orders",
                        arguments=arguments,
                        call_id="call_1",
                    )
                ],
            ),
        ]
    )
    session = AgentSession(llm=llm_)

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await session.start(agent=Agent(instructions="test", tools=[lookup_orders]))
        await session.run(user_input="show my orders")
        await session.aclose()

    tool_warnings = [
        r
        for r in caplog.records
        if r.name.startswith("livekit.agents")
        and "ToolError while executing tool" in r.getMessage()
    ]
    assert tool_warnings, "expected a ToolError warning from the invalid tool call"
    for record in tool_warnings:
        assert SECRET not in record.getMessage()
        assert getattr(record, "lk.pii.error", None), "detail should be under lk.pii.error"
        assert SECRET in str(getattr(record, "lk.pii.error"))
