"""Progressive-disclosure knowledge base backed by bundled Markdown files.

The agent's prompt keeps LiveKit Agents knowledge inline; every other LiveKit
product lives in ``products/`` as one Markdown file per product. The first line of each
file is a one-sentence description that becomes the product's index entry in
the tool schema; the rest of the file is what the model receives when it
looks the product up.

``KnowledgeBase`` owns the current storage mechanism and exposes the lookup
tool used by the agent. If retrieval grows beyond bundled Markdown later, the
agent can depend on a different knowledge-base implementation without changing
its conversational role.
"""

from __future__ import annotations

from importlib.resources import files
from typing import NamedTuple

from livekit.agents import RunContext, ToolError, function_tool
from livekit.agents.llm import RawFunctionTool


class _Entry(NamedTuple):
    description: str
    body: str


class KnowledgeBase:
    """The agent's progressively disclosed LiveKit product knowledge."""

    def __init__(self) -> None:
        self._entries = self._load_entries()

    @staticmethod
    def _load_entries() -> dict[str, _Entry]:
        entries: dict[str, _Entry] = {}
        products = files(f"{__package__}.products")
        for resource in sorted(products.iterdir(), key=lambda item: item.name):
            if not resource.name.endswith(".md"):
                continue
            description, _, body = resource.read_text(encoding="utf-8").partition("\n")
            name = resource.name.removesuffix(".md")
            entries[name] = _Entry(description.strip(), body.strip())
        return entries

    def lookup_tool(self) -> RawFunctionTool:
        names = sorted(self._entries)
        index = "\n".join(f"- {name}: {entry.description}" for name, entry in self._entries.items())

        @function_tool(
            raw_schema={
                "name": "lookup_product",
                "description": (
                    "Fetch the full knowledge base for one LiveKit product. Call this "
                    "before answering any question about a LiveKit product other than "
                    "the Agents SDKs - look it up rather than answering from memory. "
                    "Products:\n" + index
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "The product to fetch information about.",
                            "enum": names,
                        }
                    },
                    "required": ["product"],
                },
            }
        )
        async def lookup_product(raw_arguments: dict[str, object], ctx: RunContext) -> str:
            product = str(raw_arguments.get("product", ""))
            if product not in self._entries:
                raise ToolError(f"unknown product {product!r} - valid products: {', '.join(names)}")
            return self._entries[product].body

        return lookup_product
