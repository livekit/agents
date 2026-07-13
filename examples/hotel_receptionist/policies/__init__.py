"""Progressive-disclosure knowledge base over the *.md files in this package.

The receptionist's prompt keeps only hot-path facts inline; everything
long-tail lives here as one markdown file per topic. The first line of each
file is a one-sentence description that becomes the topic's index entry in
the tool schema; the rest of the file is what the model receives when it
looks the topic up.

Adding a topic = adding a file. The tool's enum and index are rebuilt from
the directory at startup, so they can never drift from the corpus.
"""

from __future__ import annotations

from pathlib import Path

from livekit.agents import RunContext, ToolError, function_tool
from livekit.agents.llm import RawFunctionTool

_POLICY_DIR = Path(__file__).parent


def build_lookup_policy_tool() -> RawFunctionTool:
    policies: dict[str, str] = {}
    index: list[str] = []
    for path in sorted(_POLICY_DIR.glob("*.md")):
        description, _, body = path.read_text().partition("\n")
        policies[path.stem] = body.strip()
        index.append(f"- {path.stem}: {description.strip()}")

    @function_tool(
        raw_schema={
            "name": "lookup_policy",
            "description": (
                "Fetch the full hotel or restaurant policy text for one topic. Call this "
                "before answering any question beyond the quick facts in your instructions - "
                "look it up rather than answering from memory. Topics:\n" + "\n".join(index)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The policy topic to fetch.",
                        "enum": sorted(policies),
                    }
                },
                "required": ["topic"],
            },
        }
    )
    async def lookup_policy(raw_arguments: dict[str, object], ctx: RunContext) -> str:
        topic = str(raw_arguments.get("topic", ""))
        if topic not in policies:
            raise ToolError(
                f"unknown topic {topic!r} - valid topics: {', '.join(sorted(policies))}"
            )
        return policies[topic]

    return lookup_policy
