from __future__ import annotations

import sys
from typing import Callable, ClassVar

__all__ = ["readchar", "readkey", "key"]


class _BaseKey:
    # Common control characters
    LF: ClassVar[str] = "\x0a"
    CR: ClassVar[str] = "\x0d"
    SPACE: ClassVar[str] = "\x20"
    ESC: ClassVar[str] = "\x1b"
    TAB: ClassVar[str] = "\x09"

    # CTRL keys – mapping from letter to control code
    CTRL_A: ClassVar[str] = "\x01"
    CTRL_B: ClassVar[str] = "\x02"
    CTRL_C: ClassVar[str] = "\x03"
    CTRL_D: ClassVar[str] = "\x04"
    CTRL_E: ClassVar[str] = "\x05"
    CTRL_F: ClassVar[str] = "\x06"
    CTRL_G: ClassVar[str] = "\x07"
    CTRL_H: ClassVar[str] = "\x08"
    CTRL_I: ClassVar[str] = TAB  # Alias for TAB
    CTRL_J: ClassVar[str] = LF  # Alias for LF
    CTRL_K: ClassVar[str] = "\x0b"
    CTRL_L: ClassVar[str] = "\x0c"
    CTRL_M: ClassVar[str] = CR  # Alias for CR
    CTRL_N: ClassVar[str] = "\x0e"
    CTRL_O: ClassVar[str] = "\x0f"
    CTRL_P: ClassVar[str] = "\x10"
    CTRL_Q: ClassVar[str] = "\x11"
    CTRL_R: ClassVar[str] = "\x12"
    CTRL_S: ClassVar[str] = "\x13"
    CTRL_T: ClassVar[str] = "\x14"
    CTRL_U: ClassVar[str] = "\x15"
    CTRL_V: ClassVar[str] = "\x16"
    CTRL_W: ClassVar[str] = "\x17"
    CTRL_X: ClassVar[str] = "\x18"
    CTRL_Y: ClassVar[str] = "\x19"
    CTRL_Z: ClassVar[str] = "\x1a"


class _PosixKey(_BaseKey):
    """Namespace of key codes specific to POSIX platforms (Linux, macOS, BSD).

    These values mirror those defined in the upstream ``_posix_key.py``
    module.  All attributes from :class:`_BaseKey` are inherited.
    """

    # Common additional control character
    BACKSPACE: ClassVar[str] = "\x7f"

    # Cursor movement (escape sequences)
    UP: ClassVar[str] = "\x1b\x5b\x41"
    DOWN: ClassVar[str] = "\x1b\x5b\x42"
    LEFT: ClassVar[str] = "\x1b\x5b\x44"
    RIGHT: ClassVar[str] = "\x1b\x5b\x43"

    # Navigation keys
    INSERT: ClassVar[str] = "\x1b\x5b\x32\x7e"
    SUPR: ClassVar[str] = "\x1b\x5b\x33\x7e"
    HOME: ClassVar[str] = "\x1b\x5b\x48"
    END: ClassVar[str] = "\x1b\x5b\x46"
    PAGE_UP: ClassVar[str] = "\x1b\x5b\x35\x7e"
    PAGE_DOWN: ClassVar[str] = "\x1b\x5b\x36\x7e"

    # Function keys
    F1: ClassVar[str] = "\x1b\x4f\x50"
    F2: ClassVar[str] = "\x1b\x4f\x51"
    F3: ClassVar[str] = "\x1b\x4f\x52"
    F4: ClassVar[str] = "\x1b\x4f\x53"
    F5: ClassVar[str] = "\x1b\x5b\x31\x35\x7e"
    F6: ClassVar[str] = "\x1b\x5b\x31\x37\x7e"
    F7: ClassVar[str] = "\x1b\x5b\x31\x38\x7e"
    F8: ClassVar[str] = "\x1b\x5b\x31\x39\x7e"
    F9: ClassVar[str] = "\x1b\x5b\x32\x30\x7e"
    F10: ClassVar[str] = "\x1b\x5b\x32\x31\x7e"
    F11: ClassVar[str] = "\x1b\x5b\x32\x33\x7e"
    F12: ClassVar[str] = "\x1b\x5b\x32\x34\x7e"

    # Shift/other combinations
    SHIFT_TAB: ClassVar[str] = "\x1b\x5b\x5a"
    CTRL_ALT_SUPR: ClassVar[str] = "\x1b\x5b\x33\x5e"

    # ALT combinations
    ALT_A: ClassVar[str] = "\x1b\x61"

    # CTRL+ALT combinations
    CTRL_ALT_A: ClassVar[str] = "\x1b\x01"

    # Aliases to improve readability
    ENTER: ClassVar[str] = _BaseKey.LF
    DELETE: ClassVar[str] = SUPR


class _WinKey(_BaseKey):
    """Namespace of key codes specific to Windows platforms.

    These values mirror those defined in the upstream ``_win_key.py``
    module.  All attributes from :class:`_BaseKey` are inherited.
    """

    # Additional common control character on Windows
    BACKSPACE: ClassVar[str] = "\x08"

    # Cursor movement (two‑byte scan codes)
    UP: ClassVar[str] = "\x00\x48"
    DOWN: ClassVar[str] = "\x00\x50"
    LEFT: ClassVar[str] = "\x00\x4b"
    RIGHT: ClassVar[str] = "\x00\x4d"

    # Navigation keys
    INSERT: ClassVar[str] = "\x00\x52"
    SUPR: ClassVar[str] = "\x00\x53"
    HOME: ClassVar[str] = "\x00\x47"
    END: ClassVar[str] = "\x00\x4f"
    PAGE_UP: ClassVar[str] = "\x00\x49"
    PAGE_DOWN: ClassVar[str] = "\x00\x51"

    # Function keys
    F1: ClassVar[str] = "\x00\x3b"
    F2: ClassVar[str] = "\x00\x3c"
    F3: ClassVar[str] = "\x00\x3d"
    F4: ClassVar[str] = "\x00\x3e"
    F5: ClassVar[str] = "\x00\x3f"
    F6: ClassVar[str] = "\x00\x40"
    F7: ClassVar[str] = "\x00\x41"
    F8: ClassVar[str] = "\x00\x42"
    F9: ClassVar[str] = "\x00\x43"
    F10: ClassVar[str] = "\x00\x44"
    # F11 and F12 values are taken from FreePascal documentation
    F11: ClassVar[str] = "\x00\x85"
    F12: ClassVar[str] = "\x00\x86"

    # Other special sequences
    ESC_2: ClassVar[str] = "\x00\x01"
    ENTER_2: ClassVar[str] = "\x00\x1c"

    # Aliases to improve readability
    ENTER: ClassVar[str] = _BaseKey.CR
    DELETE: ClassVar[str] = SUPR


# Use the CTRL+C definition from the base key set.  The choice of
# base key here mirrors the upstream behaviour: on both Windows and
# POSIX, pressing CTRL+C should raise KeyboardInterrupt instead of
# being returned by readkey().
INTERRUPT_KEYS: list[str] = [_BaseKey.CTRL_C]


def _posix_readchar() -> str:
    """Read a single character from standard input on POSIX systems.

    This function blocks until a character is available.  It uses
    ``termios`` to disable canonical input processing and echo so
    characters are returned immediately and without being echoed to
    the terminal.  The implementation closely follows the upstream
    ``_posix_read.readchar`` function.
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    term = termios.tcgetattr(fd)
    try:
        term[3] &= ~(termios.ICANON | termios.ECHO)
        term[3] |= termios.ISIG
        termios.tcsetattr(fd, termios.TCSAFLUSH, term)

        ch = sys.stdin.read(1)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            try:
                tty.setcbreak(fd)
                cur = termios.tcgetattr(fd)
                cur[3] |= termios.ICANON | termios.ECHO | termios.ISIG
                termios.tcsetattr(fd, termios.TCSADRAIN, cur)
            except Exception:
                pass
    return ch


def _posix_readkey() -> str:
    """Read the next keypress on POSIX systems.

    If a multi‑byte escape sequence is encountered (for example, an arrow
    key or function key), the entire sequence is read and returned.
    ``KeyboardInterrupt`` is raised when a key listed in
    :data:`config.INTERRUPT_KEYS` is pressed.
    """
    c1 = _posix_readchar()

    if c1 in INTERRUPT_KEYS:
        raise KeyboardInterrupt

    # Not an escape sequence; return immediately
    if c1 != "\x1b":
        return c1

    # Escape sequence – read second byte
    c2 = _posix_readchar()
    if c2 not in "\x4f\x5b":
        return c1 + c2

    # Third byte distinguishes between simple arrows and multi‑byte
    c3 = _posix_readchar()
    if c3 not in "\x31\x32\x33\x35\x36":
        return c1 + c2 + c3

    # Fourth byte for multi‑byte function/navigation keys
    c4 = _posix_readchar()
    if c4 not in "\x30\x31\x33\x34\x35\x37\x38\x39":
        return c1 + c2 + c3 + c4

    # Fifth byte for the remainder of the sequence
    c5 = _posix_readchar()
    return c1 + c2 + c3 + c4 + c5


def _win_readchar() -> str:
    """Read a single UTF‑16 code unit from standard input on Windows systems.

    This function blocks until a character is available.  It wraps
    ``msvcrt.getwch()`` from the standard library, which returns a
    single wide character (as a Python string).  The implementation is
    equivalent to the upstream ``_win_read.readchar`` function.
    """
    import msvcrt

    return msvcrt.getwch()  # type: ignore


def _win_readkey() -> str:
    """Read the next keypress on Windows systems.

    This function interprets Windows scan codes and surrogate pairs to
    return a key sequence that is compatible with the constants defined
    in :class:`_WinKey`.  ``KeyboardInterrupt`` is raised when a key
    listed in :data:`config.INTERRUPT_KEYS` is pressed.
    """
    ch = _win_readchar()

    if ch in INTERRUPT_KEYS:
        raise KeyboardInterrupt

    if ch in "\x00\xe0":
        ch = "\x00" + _win_readchar()

    # Handle UTF‑16 surrogate pairs (high surrogate from \uD800 to \uDFFF)
    # See https://docs.python.org/3/c-api/unicode.html#c.Py_UNICODE_IS_SURROGATE
    if "\ud800" <= ch <= "\udfff":
        ch += _win_readchar()
        # Combine surrogate pair into a single UTF‑16 character
        ch = ch.encode("utf-16", errors="surrogatepass").decode("utf-16")

    return ch


key: type[_PosixKey | _WinKey]
readchar: Callable[[], str]
readkey: Callable[[], str]

if sys.platform.startswith(("linux", "darwin", "freebsd", "openbsd")):
    key = _PosixKey
    readchar = _posix_readchar
    readkey = _posix_readkey
elif sys.platform in ("win32", "cygwin"):
    key = _WinKey
    readchar = _win_readchar
    readkey = _win_readkey
else:
    raise NotImplementedError(f"The platform {sys.platform} is not supported yet")
