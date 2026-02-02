from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from livekit.agents import ProviderTool
from openai.types import responses


class OpenAITool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(eq=False)
class WebSearch(OpenAITool):
    """Enable web search tool to access up-to-date information from the internet"""

    filters: Optional[responses.web_search_tool.Filters] = None
    search_context_size: Optional[Literal["low", "medium", "high"]] = "medium"
    user_location: Optional[responses.web_search_tool.UserLocation] = None

    def __post_init__(self) -> None:
        super().__init__(id="openai_web_search")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self.search_context_size,
        }
        if self.user_location is not None:
            result["user_location"] = self.user_location
        if self.filters is not None:
            result["filters"] = self.filters

        return result


@dataclass(eq=False)
class FileSearch(OpenAITool):
    """Enable file search tool to search uploaded document collections"""

    vector_store_ids: list[str] = field(default_factory=list)
    filters: Optional[responses.file_search_tool.Filters] = None
    max_num_results: Optional[int] = None
    ranking_options: Optional[responses.file_search_tool.RankingOptions] = None

    def __post_init__(self) -> None:
        super().__init__(id="openai_file_search")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
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


@dataclass(eq=False)
class CodeInterpreter(OpenAITool):
    """Enable the code interpreter tool to write and execute Python code in a sandboxed environment"""

    container: Optional[str | dict[str, Any]] = None

    def __post_init__(self) -> None:
        super().__init__(id="openai_code_interpreter")

    def to_dict(self) -> dict[str, Any]:
        result = {"type": "code_interpreter", "container": self.container}

        return result
