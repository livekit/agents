from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from livekit.agents import ProviderTool
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool, Tool


class XAITool(ProviderTool):
    """Base class for xAI server-side provider tools."""

    # function names the server answers to once this tool is enabled. Measured against
    # grok-voice-latest: a client function of the same name makes the first
    # response.create return server_error.
    _reserved_function_names: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches.

    Do not also register a function named ``web_search`` or ``browse_page``;
    xAI's WebSearch already uses those names.
    """

    _reserved_function_names: ClassVar[frozenset[str]] = frozenset({"web_search", "browse_page"})

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

    _reserved_function_names: ClassVar[frozenset[str]] = frozenset(
        {"x_keyword_search", "x_semantic_search", "x_user_search", "x_thread_fetch"}
    )
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

    _reserved_function_names: ClassVar[frozenset[str]] = frozenset(
        {"collections_search", "file_search"}
    )
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


def _raise_if_xai_tool_reserved_name_conflict(tools: Sequence[Tool]) -> None:
    """Reject a client function whose name an enabled xAI provider tool already answers to."""
    fnc_names = {
        tool.info.name for tool in tools if isinstance(tool, (FunctionTool, RawFunctionTool))
    }
    for tool in tools:
        if not isinstance(tool, XAITool):
            continue
        if conflicts := sorted(tool._reserved_function_names & fnc_names):
            names = ", ".join(repr(name) for name in conflicts)
            raise ValueError(
                f"xAI {type(tool).__name__} already uses the function name(s) {names}. "
                "Rename or remove them; a client function that shadows a provider tool's "
                "own name makes grok-voice-latest return server_error/internal_error on "
                "the first response.create."
            )
