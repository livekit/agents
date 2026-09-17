from abc import abstractmethod
from typing import Any

from ... import ProviderTool


class OpenAITool(ProviderTool):
    """Base class for OpenAI server-side provider tools."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...
