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


# Note: consumers interpret the `str` arm differently. AgentSession (error-message
# playout) treats a str as a file path only when os.path.isfile() is true, otherwise
# as text to synthesize via TTS; BackgroundAudioPlayer always treats a str as a file
# path (no TTS fallback). The contracts genuinely differ, so there is no shared resolver.
AudioSource = AsyncIterator[rtc.AudioFrame] | str | BuiltinAudioClip
