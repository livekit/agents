from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import ProviderTool


class XAITool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass(slots=True)
class XSearch(XAITool):
    """Enable X (Twitter) search tool for searching posts."""

    allowed_x_handles: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": "x_search"}
        if self.allowed_x_handles:
            result["allowed_x_handles"] = self.allowed_x_handles
        return result


@dataclass(slots=True)
class FileSearch(XAITool):
    """Enable file search tool for searching uploaded document collections."""

    vector_store_ids: list[str] = field(default_factory=list)
    max_num_results: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": self.vector_store_ids,
        }
        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        return result
