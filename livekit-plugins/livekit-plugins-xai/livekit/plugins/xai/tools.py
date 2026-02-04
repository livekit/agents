from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from livekit.agents import ProviderTool


class XAITool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "xai_web_search")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass(frozen=True)
class XSearch(XAITool):
    """Enable X (Twitter) search tool for searching posts."""

    allowed_x_handles: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "xai_x_search")
        if self.allowed_x_handles is not None:
            object.__setattr__(self, "allowed_x_handles", tuple(self.allowed_x_handles))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": "x_search"}
        if self.allowed_x_handles:
            result["allowed_x_handles"] = self.allowed_x_handles
        return result


@dataclass(frozen=True)
class FileSearch(XAITool):
    """Enable file search tool for searching uploaded document collections."""

    vector_store_ids: list[str] | tuple[str, ...] = ()
    max_num_results: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "xai_file_search")
        object.__setattr__(self, "vector_store_ids", tuple(self.vector_store_ids))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": self.vector_store_ids,
        }
        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        return result
