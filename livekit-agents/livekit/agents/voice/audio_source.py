from __future__ import annotations

import atexit
import contextlib
import enum
from collections.abc import AsyncIterator
from importlib.resources import as_file, files

from livekit import rtc

_resource_stack = contextlib.ExitStack()
atexit.register(_resource_stack.close)


class BuiltinAudioClip(enum.Enum):
    CITY_AMBIENCE = "city-ambience.ogg"
    FOREST_AMBIENCE = "forest-ambience.ogg"
    OFFICE_AMBIENCE = "office-ambience.ogg"
    CROWDED_ROOM = "crowded-room.ogg"
    KEYBOARD_TYPING = "keyboard-typing.ogg"
    KEYBOARD_TYPING2 = "keyboard-typing2.ogg"
    HOLD_MUSIC = "hold_music.ogg"

    def path(self) -> str:
        file_path = files("livekit.agents.resources") / self.value
        return str(_resource_stack.enter_context(as_file(file_path)))


AudioSource = AsyncIterator[rtc.AudioFrame] | str | BuiltinAudioClip
