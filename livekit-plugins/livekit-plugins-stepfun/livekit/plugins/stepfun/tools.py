from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from livekit.agents import ProviderTool


class StepFunTool(ProviderTool):
    """Base class for StepFun server-side provider tools."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WebSearch(StepFunTool):
    """Enable StepFun native server-side web search tool."""

    top_k: int = 5
    timeout_seconds: int = 3
    description: str = "Search the web for up-to-date information and real-time news."

    def __post_init__(self) -> None:
        super().__init__(id="stepfun_web_search")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "web_search",
            "function": {
                "description": self.description,
                "options": {
                    "top_k": self.top_k,
                    "timeout_seconds": self.timeout_seconds,
                },
            },
        }


@dataclass
class Retrieval(StepFunTool):
    """Enable StepFun native server-side vector store retrieval tool."""

    vector_store_id: str
    description: str = "Search and retrieve relevant context from the knowledge base."
    prompt_template: str | None = None

    def __post_init__(self) -> None:
        super().__init__(id="stepfun_retrieval")

    def to_dict(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "vector_store_id": self.vector_store_id,
        }
        if self.prompt_template is not None:
            options["prompt_template"] = self.prompt_template

        return {
            "type": "retrieval",
            "function": {
                "description": self.description,
                "options": options,
            },
        }
