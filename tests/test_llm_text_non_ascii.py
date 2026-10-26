from __future__ import annotations

import mcp.types
import pytest

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
