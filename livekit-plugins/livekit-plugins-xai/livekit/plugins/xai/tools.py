from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import ProviderTool
from livekit.agents.llm.tool_context import Tool, Toolset, get_fnc_tool_names


class XAITool(ProviderTool):
    """Base class for xAI server-side provider tools."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches.

    Do not also register a function named ``web_search`` or ``browse_page``;
    xAI's WebSearch already uses those names.
    """

    def __post_init__(self) -> None:
        super().__init__(id="xai_web_search")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass
class XSearch(XAITool):
    """Enable X (Twitter) search tool for searching posts.

    Do not also register a function named ``x_keyword_search``,
    ``x_semantic_search``, ``x_user_search``, or ``x_thread_fetch``;
    xAI's XSearch already uses those names.
    """

    allowed_x_handles: list[str] | None = None

    def __post_init__(self) -> None:
        super().__init__(id="xai_x_search")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": "x_search"}
        if self.allowed_x_handles:
            result["allowed_x_handles"] = self.allowed_x_handles
        return result


@dataclass
class FileSearch(XAITool):
    """Enable file search tool for searching uploaded document collections.

    Do not also register a function named ``collections_search`` or ``file_search``;
    xAI's FileSearch already uses those names.
    """

    vector_store_ids: list[str] = field(default_factory=list)
    max_num_results: int | None = None

    def __post_init__(self) -> None:
        super().__init__(id="xai_file_search")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": self.vector_store_ids,
        }
        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        return result


# Measured against grok-voice-latest: a client function of these names plus the
# matching provider tool makes the first response.create return server_error.
_XAI_TOOL_RESERVED_FUNCTION_NAMES: dict[type[XAITool], frozenset[str]] = {
    WebSearch: frozenset({"web_search", "browse_page"}),
    XSearch: frozenset(
        {"x_keyword_search", "x_semantic_search", "x_user_search", "x_thread_fetch"}
    ),
    FileSearch: frozenset({"collections_search", "file_search"}),
}


def _raise_if_xai_tool_reserved_name_conflict(tools: Sequence[Tool | Toolset]) -> None:
    reserved: set[str] = set()
    owners: dict[str, str] = {}
    for tool in tools:
        for tool_cls, names in _XAI_TOOL_RESERVED_FUNCTION_NAMES.items():
            if isinstance(tool, tool_cls):
                reserved |= names
                for name in names:
                    owners[name] = tool_cls.__name__
    if not reserved:
        return
    for name in get_fnc_tool_names(tools):
        if name in reserved:
            raise ValueError(
                f"xAI {owners[name]} already uses the function name {name!r}. "
                "Rename or remove the function; mixing the provider tool with a "
                "client function of a reserved name makes grok-voice-latest "
                "return server_error/internal_error on the first response.create."
            )
