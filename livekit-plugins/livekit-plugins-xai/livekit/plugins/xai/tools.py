from abc import abstractmethod
from dataclasses import dataclass, field

from pydantic import JsonValue

from livekit.agents import ProviderTool


class XAITool(ProviderTool):
    """Base class for xAI server-side provider tools."""

    provider = "xai"

    @abstractmethod
    def to_dict(self) -> dict[str, JsonValue]: ...


@dataclass
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches.

    Do not also register a function named ``web_search`` or ``browse_page``;
    xAI's WebSearch already uses those names.
    """

    def __post_init__(self) -> None:
        super().__init__(id="xai_web_search")

    def to_dict(self) -> dict[str, JsonValue]:
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

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"type": "x_search"}
        if self.allowed_x_handles:
            result["allowed_x_handles"] = list(self.allowed_x_handles)
        return result


@dataclass
class FileSearch(XAITool):
    """Enable file search tool for searching uploaded document collections.

    Do not also register a function named ``collections_search`` or ``file_search``;
    xAI's FileSearch already uses those names.
    """

    vector_store_ids: list[str] = field(default_factory=list[str])
    max_num_results: int | None = None

    def __post_init__(self) -> None:
        super().__init__(id="xai_file_search")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "type": "file_search",
            "vector_store_ids": list(self.vector_store_ids),
        }
        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        return result
