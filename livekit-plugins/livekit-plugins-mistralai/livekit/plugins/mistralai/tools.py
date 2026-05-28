from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from livekit.agents import ProviderTool


class MistralTool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class WebSearch(MistralTool):
    """Enable web search tool to access up-to-date information from the internet."""

    def __post_init__(self) -> None:
        super().__init__(id="mistral_web_search")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass
class DocumentLibrary(MistralTool):
    """Enable document library tool to search uploaded document collections."""

    library_ids: list[str]

    def __post_init__(self) -> None:
        super().__init__(id="mistral_document_library")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "document_library", "library_ids": self.library_ids}


@dataclass
class CodeInterpreter(MistralTool):
    """Enable the code interpreter tool to write and execute Python code."""

    def __post_init__(self) -> None:
        super().__init__(id="mistral_code_interpreter")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "code_interpreter"}


@dataclass
class Connector(MistralTool):
    """Enable the connector tool"""

    connector_id: str

    def __post_init__(self) -> None:
        super().__init__(id=f"mistral_connector_{self.connector_id}")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "connector", "connector_id": self.connector_id}
