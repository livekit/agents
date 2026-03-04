from abc import ABC, abstractmethod
from dataclasses import dataclass

from google.genai import types
from livekit.agents import llm


class GeminiTool(llm.ProviderTool, ABC):
    @abstractmethod
    def to_tool_config(self) -> types.Tool: ...


@dataclass
class GoogleSearch(GeminiTool):
    exclude_domains: list[str] | None = None
    blocking_confidence: types.PhishBlockThreshold | None = None
    time_range_filter: types.Interval | None = None

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


@dataclass
class GoogleMaps(GeminiTool):
    auth_config: types.AuthConfig | None = None
    enable_widget: bool | None = None

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


@dataclass
class FileSearch(GeminiTool):
    file_search_store_names: list[str]
    top_k: int | None = None
    metadata_filter: str | None = None

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
