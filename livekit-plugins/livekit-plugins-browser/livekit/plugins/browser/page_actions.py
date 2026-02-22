"""PageActions — typed input API for a CEF BrowserPage."""

from __future__ import annotations

import asyncio
from typing import Any

from livekit import rtc
from livekit.browser import BrowserPage  # type: ignore[import-untyped]

from ._keys import (
    CHAR,
    KEY_NAME_TO_VK,
    KEYUP,
    MOD_SHIFT,
    MODIFIER_MAP,
    NATIVE_KEY_CODES,
    NON_CHAR_KEYS,
    RAWKEYDOWN,
    SHIFTED_CHAR_TO_VK,
)

VK_SHIFT = 16


class PageActions:
    """Typed input API for a CEF BrowserPage with frame capture.

    Usage::

        actions = PageActions(page=page)
        await actions.left_click(100, 200)
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

    async def left_click(self, x: int, y: int, *, modifiers: str | None = None) -> None:
        mod_keys = _parse_modifier_keys(modifiers)
        await self._page.send_mouse_move(x, y)
        await _press_modifiers(self._page, mod_keys)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await _release_modifiers(self._page, mod_keys)

    async def right_click(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 2, False, 1)
        await self._page.send_mouse_click(x, y, 2, True, 1)

    async def double_click(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)

    async def triple_click(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)
        await self._page.send_mouse_click(x, y, 0, True, 1)
        await self._page.send_mouse_click(x, y, 0, False, 2)
        await self._page.send_mouse_click(x, y, 0, True, 2)
        await self._page.send_mouse_click(x, y, 0, False, 3)
        await self._page.send_mouse_click(x, y, 0, True, 3)

    async def middle_click(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 1, False, 1)
        await self._page.send_mouse_click(x, y, 1, True, 1)

    async def mouse_move(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)

    async def left_click_drag(self, *, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        await self._page.send_mouse_move(start_x, start_y)
        await self._page.send_mouse_click(start_x, start_y, 0, False, 1)
        await asyncio.sleep(0.05)
        await self._page.send_mouse_move(end_x, end_y)
        await asyncio.sleep(0.05)
        await self._page.send_mouse_click(end_x, end_y, 0, True, 1)

    async def left_mouse_down(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, False, 1)

    async def left_mouse_up(self, x: int, y: int) -> None:
        await self._page.send_mouse_move(x, y)
        await self._page.send_mouse_click(x, y, 0, True, 1)

    async def scroll(
        self,
        x: int,
        y: int,
        *,
        direction: str = "down",
        amount: int = 3,
    ) -> None:
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
            char_code = ord(ch)
            shifted = False

            if ch.isalpha():
                vk = ord(ch.upper())
                shifted = ch.isupper()
            elif ch in SHIFTED_CHAR_TO_VK:
                vk = SHIFTED_CHAR_TO_VK[ch]
                shifted = True
            else:
                vk = KEY_NAME_TO_VK.get(ch, 0)

            if not vk:
                # Unknown character (e.g. unicode) — CHAR-only
                await self._page.send_key_event(CHAR, 0, 0, 0, char_code)
                await asyncio.sleep(0.01)
                continue

            modifiers = MOD_SHIFT if shifted else 0
            nkc = NATIVE_KEY_CODES.get(vk, 0)

            if shifted:
                shift_nkc = NATIVE_KEY_CODES.get(VK_SHIFT, 0)
                await self._page.send_key_event(RAWKEYDOWN, MOD_SHIFT, VK_SHIFT, shift_nkc, 0)

            await self._page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)
            await self._page.send_key_event(CHAR, modifiers, char_code, nkc, char_code)
            await self._page.send_key_event(KEYUP, modifiers, vk, 0, 0)

            if shifted:
                await self._page.send_key_event(KEYUP, 0, VK_SHIFT, 0, 0)

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

    # -- navigation ----------------------------------------------------------

    async def navigate(self, url: str) -> None:
        await self._page.navigate(url)

    async def go_back(self) -> None:
        await self._page.go_back()

    async def go_forward(self) -> None:
        await self._page.go_forward()

    # -- lifecycle -----------------------------------------------------------

    def aclose(self) -> None:
        self._page.off("paint", self._on_paint)


# -- helpers -----------------------------------------------------------------


def _parse_modifier_keys(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip().lower() for part in text.split("+") if part.strip()]


async def _press_modifiers(page: BrowserPage, keys: list[str]) -> None:
    modifiers = 0
    for k in keys:
        modifiers |= MODIFIER_MAP.get(k, 0)
        vk = KEY_NAME_TO_VK.get(k, 0)
        nkc = NATIVE_KEY_CODES.get(vk, 0)
        await page.send_key_event(RAWKEYDOWN, modifiers, vk, nkc, 0)


async def _release_modifiers(page: BrowserPage, keys: list[str]) -> None:
    for k in reversed(keys):
        vk = KEY_NAME_TO_VK.get(k, 0)
        await page.send_key_event(KEYUP, 0, vk, 0, 0)


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
            await page.send_key_event(CHAR, modifiers, char_code, nkc, char_code)
        await page.send_key_event(KEYUP, modifiers, vk, 0, 0)

    # Release modifier keys
    for k in reversed(keys):
        if k in MODIFIER_MAP:
            vk = KEY_NAME_TO_VK.get(k, 0)
            await page.send_key_event(KEYUP, 0, vk, 0, 0)
