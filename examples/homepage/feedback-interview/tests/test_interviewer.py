from livekit.agents.llm import mcp
from mcp.types import CallToolResult, TextContent

from interviewer import (
    DOCS_RESULT_CHARS,
    EMPTY_DOCS_RESULT,
    INTERVIEW_MINUTES,
    describe_clock,
    docs_result,
)


def test_clock_counts_down_then_says_to_close():
    assert describe_clock(60) == "Interview clock: 1 min in, about 4 min left."
    assert "closing question" in describe_clock((INTERVIEW_MINUTES - 1) * 60)
    assert "at time" in describe_clock(INTERVIEW_MINUTES * 60)
    assert "at time" in describe_clock((INTERVIEW_MINUTES + 2) * 60)


def test_docs_results_are_cut_short():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a" * DOCS_RESULT_CHARS),
            TextContent(type="text", text="b" * 10),
        ]
    )
    text = docs_result(mcp.MCPToolResultContext("docs_search", {}, result))
    assert text == "a" * DOCS_RESULT_CHARS


def test_an_empty_docs_result_says_how_to_search_again():
    result = CallToolResult(content=[TextContent(type="text", text="")])
    ctx = mcp.MCPToolResultContext("docs_search", {"queries": []}, result)
    assert docs_result(ctx) == EMPTY_DOCS_RESULT
