from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from livekit.agents import llm


class GeminiTool(llm.ProviderTool, ABC):
    @abstractmethod
    def to_tool_config(self) -> types.Tool: ...


@dataclass(eq=False)
class GoogleSearch(GeminiTool):
    exclude_domains: Optional[list[str]] = None
    blocking_confidence: Optional[types.PhishBlockThreshold] = None
    time_range_filter: Optional[types.Interval] = None

    def __post_init__(self) -> None:
        super().__init__(id="gemini_google_search")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            google_search=types.GoogleSearch(
                exclude_domains=self.exclude_domains,
                blocking_confidence=self.blocking_confidence,
                time_range_filter=self.time_range_filter,
            )
        )


@dataclass(eq=False)
class GoogleMaps(GeminiTool):
    auth_config: Optional[types.AuthConfig] = None
    enable_widget: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__init__(id="gemini_google_maps")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            google_maps=types.GoogleMaps(
                auth_config=self.auth_config,
                enable_widget=self.enable_widget,
            )
        )


class URLContext(GeminiTool):
    def __init__(self) -> None:
        super().__init__(id="gemini_url_context")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            url_context=types.UrlContext(),
        )


@dataclass(eq=False)
class FileSearch(GeminiTool):
    file_search_store_names: list[str]
    top_k: Optional[int] = None
    metadata_filter: Optional[str] = None

    def __post_init__(self) -> None:
        super().__init__(id="gemini_file_search")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=self.file_search_store_names,
                top_k=self.top_k,
                metadata_filter=self.metadata_filter,
            )
        )


class ToolCodeExecution(GeminiTool):
    def __init__(self) -> None:
        super().__init__(id="gemini_code_execution")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            code_execution=types.ToolCodeExecution(),
        )
