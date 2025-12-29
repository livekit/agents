from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from livekit.agents import ProviderTool


class OpenAITool(ProviderTool):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class WebSearch(OpenAITool):
    """Enable web search tool to access up-to-date information from the internet"""

    filters: Optional[dict[str, list[str]]] = None
    search_context_size: Optional[Literal["low", "medium", "high"]] = "medium"
    user_location: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": "web_search",
            "search_context_size": self.search_context_size,
        }
        if self.user_location is not None:
            result["user_location"] = self.user_location
        if self.filters is not None:
            result["filters"] = self.filters

        return result


@dataclass(slots=True)
class FileSearch(OpenAITool):
    """Enable file search tool to search uploaded document collections"""

    vector_store_ids: list[str] = field(default_factory=list)
    filters: Optional[dict[str, Any]] = None
    max_num_results: Optional[int] = None
    ranking_options: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": "file_search",
            "vector_store_ids": self.vector_store_ids,
        }
        if self.filters is not None:
            result["filters"] = self.filters

        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        if self.ranking_options is not None:
            result["ranking_options"] = self.ranking_options

        return result


@dataclass(slots=True)
class CodeInterpreter(OpenAITool):
    """Enable the code interpreter tool to write and execute Python code in a sandboxed environment"""

    container: Optional[str | dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        result = {"type": "code_interpreter", "container": self.container}

        return result
