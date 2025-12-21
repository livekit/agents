from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from google.genai import types
from livekit.agents import llm


class GeminiTool(llm.ProviderTool, ABC):
    @abstractmethod
    def to_tool_config(self) -> types.Tool: ...


@dataclass
class GoogleSearch(GeminiTool):
    exclude_domains: Optional[list[str]] = None
    blocking_confidence: Optional[types.PhishBlockThreshold] = None
    time_range_filter: Optional[types.Interval] = None

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
    auth_config: Optional[types.AuthConfig] = None
    enable_widget: Optional[bool] = None

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            google_maps=types.GoogleMaps(
                auth_config=self.auth_config,
                enable_widget=self.enable_widget,
            )
        )


class URLContext(GeminiTool):
    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            url_context=types.UrlContext(),
        )


@dataclass
class FileSearch(GeminiTool):
    file_search_store_names: Optional[list[str]]
    top_k: Optional[int]
    metadata_filter: Optional[str]

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=self.file_search_store_names,
                top_k=self.top_k,
                metadata_filter=self.metadata_filter,
            )
        )


class ToolCodeExecution(GeminiTool):
    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            code_execution=types.ToolCodeExecution(),
        )
