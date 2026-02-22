"""ComputerTool — Anthropic computer_use Toolset backed by browser PageActions."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Any

from livekit import rtc
from livekit.agents import llm

from .tools import ComputerUse

if TYPE_CHECKING:
    from livekit.plugins.browser import PageActions  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_POST_ACTION_DELAY = 0.3


class ComputerTool(llm.Toolset):
    """Anthropic computer_use tool backed by browser PageActions.

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
        """Dispatch an Anthropic computer_use action and return screenshot content."""
        actions = self._actions

        match action:
            case "screenshot":
                pass
            case "left_click":
                await actions.left_click(
                    _require(kwargs, "coordinate"),
                    modifiers=kwargs.get("text"),
                )
            case "right_click":
                await actions.right_click(_require(kwargs, "coordinate"))
            case "double_click":
                await actions.double_click(_require(kwargs, "coordinate"))
            case "triple_click":
                await actions.triple_click(_require(kwargs, "coordinate"))
            case "middle_click":
                await actions.middle_click(_require(kwargs, "coordinate"))
            case "mouse_move":
                await actions.mouse_move(_require(kwargs, "coordinate"))
            case "left_click_drag":
                await actions.left_click_drag(
                    start=_require(kwargs, "start_coordinate"),
                    end=_require(kwargs, "coordinate"),
                )
            case "left_mouse_down":
                await actions.left_mouse_down(_require(kwargs, "coordinate"))
            case "left_mouse_up":
                await actions.left_mouse_up(_require(kwargs, "coordinate"))
            case "scroll":
                await actions.scroll(
                    _require(kwargs, "coordinate"),
                    direction=kwargs.get("scroll_direction", "down"),
                    amount=int(kwargs.get("scroll_amount", 3)),
                )
            case "type":
                await actions.type_text(_require(kwargs, "text"))
            case "key":
                await actions.key(_require(kwargs, "text"))
            case "hold_key":
                await actions.hold_key(
                    _require(kwargs, "text"),
                    duration=float(kwargs.get("duration", 0.5)),
                )
            case "wait":
                await actions.wait()
            case _:
                raise ValueError(f"Unknown computer_use action: {action!r}")

        await asyncio.sleep(_POST_ACTION_DELAY)

        frame = actions.last_frame
        if frame is None:
            return [{"type": "text", "text": "(no frame available yet)"}]
        return _screenshot_content(frame)

    def aclose(self) -> None:
        self._actions.aclose()


def _require(kwargs: dict[str, Any], key: str) -> Any:
    """Extract a required argument, raising ValueError if missing."""
    if key not in kwargs:
        raise ValueError(f"Missing required argument: {key!r}")
    return kwargs[key]


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
