from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from livekit.agents import llm


class GeminiTool(llm.ProviderTool, ABC):
    @abstractmethod
    def to_tool_config(self) -> types.Tool: ...


@dataclass(frozen=True)
class GoogleSearch(GeminiTool):
    exclude_domains: Optional[list[str] | tuple[str, ...]] = None
    blocking_confidence: Optional[types.PhishBlockThreshold] = None
    time_range_filter: Optional[types.Interval] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "gemini_google_search")
        if self.exclude_domains is not None:
            object.__setattr__(self, "exclude_domains", tuple(self.exclude_domains))

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            google_search=types.GoogleSearch(
                exclude_domains=self.exclude_domains,
                blocking_confidence=self.blocking_confidence,
                time_range_filter=self.time_range_filter,
            )
        )


@dataclass(frozen=True)
class GoogleMaps(GeminiTool):
    auth_config: Optional[types.AuthConfig] = None
    enable_widget: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "gemini_google_maps")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            google_maps=types.GoogleMaps(
                auth_config=self.auth_config,
                enable_widget=self.enable_widget,
            )
        )


@dataclass(frozen=True)
class URLContext(GeminiTool):
    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "gemini_url_context")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            url_context=types.UrlContext(),
        )


@dataclass(frozen=True)
class FileSearch(GeminiTool):
    file_search_store_names: list[str] | tuple[str, ...]
    top_k: Optional[int] = None
    metadata_filter: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "gemini_file_search")
        object.__setattr__(self, "file_search_store_names", tuple(self.file_search_store_names))

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=self.file_search_store_names,
                top_k=self.top_k,
                metadata_filter=self.metadata_filter,
            )
        )


@dataclass(frozen=True)
class ToolCodeExecution(GeminiTool):
    def __post_init__(self) -> None:
        object.__setattr__(self, "_id", "gemini_code_execution")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            code_execution=types.ToolCodeExecution(),
        )
