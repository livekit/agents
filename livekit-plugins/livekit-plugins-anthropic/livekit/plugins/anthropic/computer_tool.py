"""ComputerTool — Anthropic computer_use Toolset backed by browser PageActions."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import llm

from .tools import ComputerUse

if TYPE_CHECKING:
    from livekit.plugins.browser import PageActions  # type: ignore[import-untyped]


class ComputerTool(llm.Toolset):
    """Anthropic computer_use tool backed by a CEF BrowserPage.

    Usage::

        from livekit.plugins.browser import PageActions

        actions = PageActions(page=page)
        tool = ComputerTool(actions=actions, width=1280, height=720)
    """

    def __init__(
        self,
        *,
        actions: PageActions,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        super().__init__(id="computer")
        self._actions = actions
        self._provider_tool = ComputerUse(
            display_width_px=width,
            display_height_px=height,
        )

    @property
    def tools(self) -> list[llm.Tool]:
        return [self._provider_tool]

    async def execute(self, action: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Execute a computer_use action and return Anthropic screenshot content."""
        frame = await self._actions.execute(action, **kwargs)
        if frame is None:
            return [{"type": "text", "text": "(no frame available yet)"}]
        return _screenshot_content(frame)

    def aclose(self) -> None:
        self._actions.aclose()


def _screenshot_content(frame: rtc.VideoFrame) -> list[dict[str, Any]]:
    """Build Anthropic tool_result content blocks with a screenshot."""
    from livekit.agents.utils.images import EncodeOptions, encode

    png_bytes = encode(frame, EncodeOptions(format="PNG"))
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        }
    ]
