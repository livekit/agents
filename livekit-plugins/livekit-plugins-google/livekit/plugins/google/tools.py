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


@dataclass
class VertexRAGRetrieval(GeminiTool):
    """Vertex AI RAG Engine retrieval tool for server-side grounding.

    Enables single-pass retrieval during Gemini inference with no tool-call
    round-trip.  Works like Google Search grounding but against your own
    document corpus managed by Vertex AI RAG Engine.

    Args:
        rag_resources: RAG corpus resource names
            (e.g. ``["projects/123/locations/us-central1/ragCorpora/456"]``).
        similarity_top_k: Number of top results to retrieve.
        vector_distance_threshold: Optional distance threshold for filtering.
    """

    rag_resources: list[str]
    similarity_top_k: int = 3
    vector_distance_threshold: float | None = None

    def __post_init__(self) -> None:
        super().__init__(id="gemini_vertex_rag_retrieval")

    def to_tool_config(self) -> types.Tool:
        return types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(rag_corpus=corpus)
                        for corpus in self.rag_resources
                    ],
                    similarity_top_k=self.similarity_top_k,
                    vector_distance_threshold=self.vector_distance_threshold,
                ),
            )
        )
