"""ComputerTool — Anthropic computer_use tool backed by a BrowserPage."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from livekit import rtc
from livekit.agents import llm
from livekit.browser import BrowserPage  # type: ignore[import-untyped]

from ._keys import (
    CHAR,
    KEY_NAME_TO_VK,
    KEYUP,
    MOD_ALT,
    MOD_CTRL,
    MOD_META,
    MOD_SHIFT,
    MODIFIER_MAP,
    NATIVE_KEY_CODES,
    NON_CHAR_KEYS,
    RAWKEYDOWN,
)

logger = logging.getLogger(__name__)

# Delay after actions to let the page repaint before screenshot
_POST_ACTION_DELAY = 0.3


def _encode_frame_png(frame: rtc.VideoFrame) -> str:
    """Encode a VideoFrame as a base64 PNG string."""
    from livekit.agents.utils.images import encode, EncodeOptions

    png_bytes = encode(frame, EncodeOptions(format="PNG"))
    return base64.b64encode(png_bytes).decode("utf-8")


def _screenshot_content(frame: rtc.VideoFrame) -> list[dict[str, Any]]:
    """Build Anthropic tool_result content blocks with a screenshot."""
    b64 = _encode_frame_png(frame)
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


class ComputerTool(llm.Toolset):
    """Anthropic computer_use tool backed by a CEF BrowserPage.

    Usage::

        tool = ComputerTool(page=page, width=1280, height=720)
        # pass tool.tools to LLM.chat(tools=[...])
    """

    def __init__(
        self,
        *,
        page: BrowserPage,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        super().__init__(id="computer")
        self._page = page
        self._width = width
        self._height = height
        self._last_frame: rtc.VideoFrame | None = None

        # Subscribe to paint events for screenshots
        self._page.on("paint", self._on_paint)

        self._provider_tool = llm.ProviderTool(
            id="computer",
            definition={
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": width,
                "display_height_px": height,
                "display_number": 1,
            },
        )

    def _on_paint(self, data: Any) -> None:
        self._last_frame = data.frame

    @property
    def tools(self) -> list[llm.Tool]:
        return [self._provider_tool]

    async def _take_screenshot(self) -> list[dict[str, Any]]:
        """Capture the current frame as screenshot content blocks."""
        if self._last_frame is None:
            return [{"type": "text", "text": "(no frame available yet)"}]
        return _screenshot_content(self._last_frame)

    async def execute(self, action: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Execute a computer_use action and return screenshot content."""
        handler = _ACTION_HANDLERS.get(action)
        if handler is None:
            return [{"type": "text", "text": f"Unknown action: {action}"}]

        await handler(self, **kwargs)

        # Small delay to let the browser repaint
        await asyncio.sleep(_POST_ACTION_DELAY)
        return await self._take_screenshot()

    # ---- Action implementations ----

    async def _action_screenshot(self, **kwargs: Any) -> None:
        pass  # Just return the screenshot

    async def _action_left_click(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        modifiers = _text_to_modifiers(kwargs.get("text"))
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)

    async def _action_right_click(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 2, False, 1)
        await self._page.send_mouse_click(x, y, 2, True, 1)

    async def _action_double_click(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)

    async def _action_triple_click(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)
        await self._page.send_mouse_click(x, y, 0, False, 3)
        await self._page.send_mouse_click(x, y, 0, True, 3)

    async def _action_middle_click(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 1, False, 1)
        await self._page.send_mouse_click(x, y, 1, True, 1)

    async def _action_mouse_move(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)

    async def _action_left_click_drag(self, **kwargs: Any) -> None:
        start = kwargs.get("start_coordinate", [0, 0])
        end = kwargs.get("coordinate", [0, 0])
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        await self._page.send_mouse_move(sx, sy)
        await self._page.send_mouse_click(sx, sy, 0, False, 1)  # mousedown
        await asyncio.sleep(0.05)
        await self._page.send_mouse_move(ex, ey)
        await asyncio.sleep(0.05)
        await self._page.send_mouse_click(ex, ey, 0, True, 1)  # mouseup

    async def _action_left_mouse_down(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)

    async def _action_left_mouse_up(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, True, 1)

    async def _action_scroll(self, **kwargs: Any) -> None:
        coord = kwargs.get("coordinate", [0, 0])
        x, y = int(coord[0]), int(coord[1])
        direction = kwargs.get("scroll_direction", "down")
        amount = int(kwargs.get("scroll_amount", 3))
        pixels = amount * 120

        delta_x, delta_y = 0, 0
        if direction == "down":
            delta_y = -pixels
        elif direction == "up":
            delta_y = pixels
        elif direction == "left":
            delta_x = pixels
        elif direction == "right":
            delta_x = -pixels

        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_wheel(x, y, delta_x, delta_y)

    async def _action_type(self, **kwargs: Any) -> None:
        text = kwargs.get("text", "")
        for ch in text:
            code = ord(ch)
            # Send CHAR event for each character
            await self._page.send_key_event(CHAR, 0, code, 0, code)
            await asyncio.sleep(0.01)

    async def _action_key(self, **kwargs: Any) -> None:
        text = kwargs.get("text", "")
        await _send_key_combo(self._page, text)

    async def _action_hold_key(self, **kwargs: Any) -> None:
        text = kwargs.get("text", "")
        duration = float(kwargs.get("duration", 0.5))
        keys = [k.strip().lower() for k in text.split("+")]

        # Press all keys down
        modifiers = 0
        for key in keys:
            if key in MODIFIER_MAP:
                modifiers |= MODIFIER_MAP[key]
                vk = KEY_NAME_TO_VK.get(key, 0)
                nkc = NATIVE_KEY_CODES.get(vk, 0)
                await self._page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)
            else:
                vk = KEY_NAME_TO_VK.get(key, 0)
                nkc = NATIVE_KEY_CODES.get(vk, 0)
                await self._page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)

        await asyncio.sleep(duration)

        # Release all keys in reverse order
        for key in reversed(keys):
            vk = KEY_NAME_TO_VK.get(key, 0)
            await self._page.send_key_event(KEYUP, 0, vk, 0, 0)

    async def _action_wait(self, **kwargs: Any) -> None:
        await asyncio.sleep(1)

    def aclose(self) -> None:
        self._page.off("paint", self._on_paint)


def _text_to_modifiers(text: str | None) -> int:
    """Convert modifier text (e.g. 'shift', 'ctrl') to CEF modifier flags."""
    if not text:
        return 0
    flags = 0
    for part in text.split("+"):
        flags |= MODIFIER_MAP.get(part.strip().lower(), 0)
    return flags


async def _send_key_combo(page: BrowserPage, text: str) -> None:
    """Send a key combination like 'ctrl+a' or 'Return'."""
    keys = [k.strip().lower() for k in text.split("+")]

    # Separate modifiers from the main key
    modifiers = 0
    main_keys: list[str] = []
    for key in keys:
        if key in MODIFIER_MAP:
            modifiers |= MODIFIER_MAP[key]
        else:
            main_keys.append(key)

    # Press modifier keys down
    for key in keys:
        if key in MODIFIER_MAP:
            vk = KEY_NAME_TO_VK.get(key, 0)
            nkc = NATIVE_KEY_CODES.get(vk, 0)
            await page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)

    # Press and release main keys
    for key in main_keys:
        vk = KEY_NAME_TO_VK.get(key, 0)
        if vk == 0 and len(key) == 1:
            # Single character — use its char code as VK
            vk = ord(key.upper())
        nkc = NATIVE_KEY_CODES.get(vk, 0)
        await page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)
        # Send CHAR for printable keys only
        if vk not in NON_CHAR_KEYS and len(key) == 1:
            char_code = ord(key)
            await page.send_key_event(CHAR, modifiers, vk, nkc, char_code)
        # Don't send native_key_code on KEYUP
        await page.send_key_event(KEYUP, modifiers, vk, 0, 0)

    # Release modifier keys
    for key in reversed(keys):
        if key in MODIFIER_MAP:
            vk = KEY_NAME_TO_VK.get(key, 0)
            await page.send_key_event(KEYUP, 0, vk, 0, 0)


# Action dispatch table
_ACTION_HANDLERS: dict[str, Any] = {
    "screenshot": ComputerTool._action_screenshot,
    "left_click": ComputerTool._action_left_click,
    "right_click": ComputerTool._action_right_click,
    "double_click": ComputerTool._action_double_click,
    "triple_click": ComputerTool._action_triple_click,
    "middle_click": ComputerTool._action_middle_click,
    "mouse_move": ComputerTool._action_mouse_move,
    "left_click_drag": ComputerTool._action_left_click_drag,
    "left_mouse_down": ComputerTool._action_left_mouse_down,
    "left_mouse_up": ComputerTool._action_left_mouse_up,
    "scroll": ComputerTool._action_scroll,
    "type": ComputerTool._action_type,
    "key": ComputerTool._action_key,
    "hold_key": ComputerTool._action_hold_key,
    "wait": ComputerTool._action_wait,
}
