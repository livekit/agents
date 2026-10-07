from __future__ import annotations

import mcp.types
import pytest

from livekit.agents.beta.workflows.task_group import TaskGroup
from livekit.agents.llm.mcp import MCPToolResultContext, _default_tool_result_resolver

pytestmark = pytest.mark.unit

HINDI = "आपका ऑर्डर कल पहुँचेगा"


def test_mcp_multi_item_result_keeps_non_ascii() -> None:
    result = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(type="text", text=HINDI),
            mcp.types.TextContent(type="text", text="ट्रैकिंग नंबर 1029"),
        ]
    )
    out = _default_tool_result_resolver(
        MCPToolResultContext(tool_name="order_status", arguments={}, result=result)
    )
    assert HINDI in out
    assert "ट्रैकिंग नंबर 1029" in out


def test_task_group_out_of_scope_description_keeps_non_ascii() -> None:
    group = TaskGroup().add(lambda: None, id="email_task", description="ईमेल पता पूछें")  # type: ignore[arg-type, return-value]
    group._visited_tasks.add("email_task")
    tool = group._build_out_of_scope_tool(active_task_id="other_task")
    assert tool is not None
    assert "ईमेल पता पूछें" in tool.info.description
