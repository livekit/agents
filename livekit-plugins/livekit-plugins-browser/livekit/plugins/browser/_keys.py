"""Shared key code mappings for CEF input events.

Used by both BrowserSession (human input) and ComputerTool (AI agent input).
"""

from __future__ import annotations

import sys


def build_native_keycode_map() -> dict[int, int]:
    """Map JS keyCode (Windows VK codes) to platform-specific native key codes.

    CEF uses native_key_code for editing commands (deleteBackward, etc.)
    on macOS. On Linux it uses X11 key codes. On Windows, windows_key_code
    is primary so native_key_code is less critical.
    """
    if sys.platform == "darwin":
        return {
            8: 51,  # Backspace
            9: 48,  # Tab
            13: 36,  # Enter/Return
            27: 53,  # Escape
            37: 123,  # Arrow Left
            38: 126,  # Arrow Up
            39: 124,  # Arrow Right
            40: 125,  # Arrow Down
            46: 117,  # Delete (forward)
            33: 116,  # Page Up
            34: 121,  # Page Down
            35: 119,  # End
            36: 115,  # Home
        }
    elif sys.platform == "linux":
        return {
            8: 22,  # Backspace
            9: 23,  # Tab
            13: 36,  # Enter/Return
            27: 9,  # Escape
            37: 113,  # Arrow Left
            38: 111,  # Arrow Up
            39: 114,  # Arrow Right
            40: 116,  # Arrow Down
            46: 119,  # Delete (forward)
            33: 112,  # Page Up
            34: 117,  # Page Down
            35: 115,  # End
            36: 110,  # Home
        }
    else:
        return {}


NATIVE_KEY_CODES = build_native_keycode_map()

KEY_NAME_TO_VK: dict[str, int] = {
    # Letters
    **{chr(c): c - 32 for c in range(ord("a"), ord("z") + 1)},  # a=65 .. z=90
    # Digits
    **{str(d): 0x30 + d for d in range(10)},
    # Special keys
    "return": 13,
    "enter": 13,
    "tab": 9,
    "escape": 27,
    "esc": 27,
    "backspace": 8,
    "delete": 46,
    "space": 32,
    " ": 32,
    # Arrow keys
    "arrowup": 38,
    "up": 38,
    "arrowdown": 40,
    "down": 40,
    "arrowleft": 37,
    "left": 37,
    "arrowright": 39,
    "right": 39,
    # Navigation
    "home": 36,
    "end": 35,
    "pageup": 33,
    "page_up": 33,
    "pagedown": 34,
    "page_down": 34,
    # Function keys
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    # Modifier keys (as standalone)
    "shift": 16,
    "control": 17,
    "ctrl": 17,
    "alt": 18,
    "meta": 91,
    "super": 91,
    "command": 91,
    "cmd": 91,
    # Punctuation
    ";": 186,
    "=": 187,
    ",": 188,
    "-": 189,
    ".": 190,
    "/": 191,
    "`": 192,
    "[": 219,
    "\\": 220,
    "]": 221,
    "'": 222,
}

# Shifted character → (base VK code, MOD_SHIFT) for US keyboard layout.
# Maps characters produced by Shift+key to their underlying VK code.
SHIFTED_CHAR_TO_VK: dict[str, int] = {
    "!": 0x31,  # Shift+1
    "@": 0x32,  # Shift+2
    "#": 0x33,  # Shift+3
    "$": 0x34,  # Shift+4
    "%": 0x35,  # Shift+5
    "^": 0x36,  # Shift+6
    "&": 0x37,  # Shift+7
    "*": 0x38,  # Shift+8
    "(": 0x39,  # Shift+9
    ")": 0x30,  # Shift+0
    "_": 189,  # Shift+-
    "+": 187,  # Shift+=
    "{": 219,  # Shift+[
    "}": 221,  # Shift+]
    "|": 220,  # Shift+\
    ":": 186,  # Shift+;
    '"': 222,  # Shift+'
    "<": 188,  # Shift+,
    ">": 190,  # Shift+.
    "?": 191,  # Shift+/
    "~": 192,  # Shift+`
}

# CEF modifier flags
MOD_SHIFT = 1 << 1
MOD_CTRL = 1 << 2
MOD_ALT = 1 << 3
MOD_META = 1 << 7

MODIFIER_MAP: dict[str, int] = {
    "shift": MOD_SHIFT,
    "control": MOD_CTRL,
    "ctrl": MOD_CTRL,
    "alt": MOD_ALT,
    "option": MOD_ALT,
    "meta": MOD_META,
    "super": MOD_META,
    "command": MOD_META,
    "cmd": MOD_META,
}

# CEF key event types
RAWKEYDOWN = 0
KEYUP = 2
CHAR = 3

# Keys that should NOT receive a CHAR event (handled by RAWKEYDOWN alone)
NON_CHAR_KEYS = {
    8,
    9,
    13,
    27,  # Backspace, Tab, Enter, Escape
    37,
    38,
    39,
    40,  # Arrow keys
    33,
    34,
    35,
    36,  # Page Up/Down, End, Home
    46,  # Delete
    16,
    17,
    18,
    91,  # Modifier keys
    *range(0x70, 0x7C),  # F1-F12
}
