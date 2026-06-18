from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from livekit.agents import ProviderTool


class AnthropicTool(ProviderTool, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @property
    def beta_flag(self) -> str | None:
        return None


_TOOL_VERSION_BETA_FLAGS: dict[str, str] = {
    "computer_20251124": "computer-use-2025-11-24",
    "computer_20250124": "computer-use-2025-01-24",
}


@dataclass
class ComputerUse(AnthropicTool):
    display_width_px: int = 1280
    display_height_px: int = 720
    display_number: int = 1
    tool_version: str = "computer_20251124"

    def __post_init__(self) -> None:
        super().__init__(id="computer")

    @property
    def beta_flag(self) -> str | None:
        return _TOOL_VERSION_BETA_FLAGS.get(self.tool_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.tool_version,
            "name": "computer",
            "display_width_px": self.display_width_px,
            "display_height_px": self.display_height_px,
            "display_number": self.display_number,
        }
