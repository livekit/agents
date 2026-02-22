"""PageActions — typed input API for a CEF BrowserPage."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from livekit import rtc
from livekit.browser import BrowserPage  # type: ignore[import-untyped]

from ._keys import (
    CHAR,
    KEY_NAME_TO_VK,
    KEYUP,
    MODIFIER_MAP,
    NATIVE_KEY_CODES,
    NON_CHAR_KEYS,
    RAWKEYDOWN,
)

Coordinate = Sequence[float]


class PageActions:
    """Typed input API for a CEF BrowserPage with frame capture.

    Usage::

        actions = PageActions(page=page)
        await actions.left_click([100, 200])
        frame = actions.last_frame
    """

    def __init__(self, *, page: BrowserPage) -> None:
        self._page = page
        self._last_frame: rtc.VideoFrame | None = None
        self._page.on("paint", self._on_paint)

    def _on_paint(self, data: Any) -> None:
        self._last_frame = data.frame

    @property
    def last_frame(self) -> rtc.VideoFrame | None:
        return self._last_frame

    # -- mouse actions -------------------------------------------------------

    async def left_click(self, coordinate: Coordinate, *, modifiers: str | None = None) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        _text_to_modifiers(modifiers)
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)

    async def right_click(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 2, False, 1)
        await self._page.send_mouse_click(x, y, 2, True, 1)

    async def double_click(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)

    async def triple_click(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)
        await self._page.send_mouse_click(x, y, 0, False, 3)
        await self._page.send_mouse_click(x, y, 0, True, 3)

    async def middle_click(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 1, False, 1)
        await self._page.send_mouse_click(x, y, 1, True, 1)

    async def mouse_move(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)

    async def left_click_drag(self, *, start: Coordinate, end: Coordinate) -> None:
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        await self._page.send_mouse_move(sx, sy)
        await self._page.send_mouse_click(sx, sy, 0, False, 1)
        await asyncio.sleep(0.05)
        await self._page.send_mouse_move(ex, ey)
        await asyncio.sleep(0.05)
        await self._page.send_mouse_click(ex, ey, 0, True, 1)

    async def left_mouse_down(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)

    async def left_mouse_up(self, coordinate: Coordinate) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, True, 1)

    async def scroll(
        self,
        coordinate: Coordinate,
        *,
        direction: str = "down",
        amount: int = 3,
    ) -> None:
        x, y = int(coordinate[0]), int(coordinate[1])
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

    # -- keyboard actions ----------------------------------------------------

    async def type_text(self, text: str) -> None:
        for ch in text:
            code = ord(ch)
            await self._page.send_key_event(CHAR, 0, code, 0, code)
            await asyncio.sleep(0.01)

    async def key(self, text: str) -> None:
        await _send_key_combo(self._page, text)

    async def hold_key(self, text: str, *, duration: float = 0.5) -> None:
        keys = [k.strip().lower() for k in text.split("+")]

        modifiers = 0
        for k in keys:
            if k in MODIFIER_MAP:
                modifiers |= MODIFIER_MAP[k]
            vk = KEY_NAME_TO_VK.get(k, 0)
            nkc = NATIVE_KEY_CODES.get(vk, 0)
            await self._page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)

        await asyncio.sleep(duration)

        for k in reversed(keys):
            vk = KEY_NAME_TO_VK.get(k, 0)
            await self._page.send_key_event(KEYUP, 0, vk, 0, 0)

    async def wait(self) -> None:
        await asyncio.sleep(1)

    # -- lifecycle -----------------------------------------------------------

    def aclose(self) -> None:
        self._page.off("paint", self._on_paint)


# -- helpers -----------------------------------------------------------------


def _text_to_modifiers(text: str | None) -> int:
    if not text:
        return 0
    flags = 0
    for part in text.split("+"):
        flags |= MODIFIER_MAP.get(part.strip().lower(), 0)
    return flags


async def _send_key_combo(page: BrowserPage, text: str) -> None:
    keys = [k.strip().lower() for k in text.split("+")]

    modifiers = 0
    main_keys: list[str] = []
    for k in keys:
        if k in MODIFIER_MAP:
            modifiers |= MODIFIER_MAP[k]
        else:
            main_keys.append(k)

    # Press modifier keys down
    for k in keys:
        if k in MODIFIER_MAP:
            vk = KEY_NAME_TO_VK.get(k, 0)
            nkc = NATIVE_KEY_CODES.get(vk, 0)
            await page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)

    # Press and release main keys
    for k in main_keys:
        vk = KEY_NAME_TO_VK.get(k, 0)
        if vk == 0 and len(k) == 1:
            vk = ord(k.upper())
        nkc = NATIVE_KEY_CODES.get(vk, 0)
        await page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)
        if vk not in NON_CHAR_KEYS and len(k) == 1:
            char_code = ord(k)
            await page.send_key_event(CHAR, modifiers, vk, nkc, char_code)
        await page.send_key_event(KEYUP, modifiers, vk, 0, 0)

    # Release modifier keys
    for k in reversed(keys):
        if k in MODIFIER_MAP:
            vk = KEY_NAME_TO_VK.get(k, 0)
            await page.send_key_event(KEYUP, 0, vk, 0, 0)
