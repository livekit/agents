# Copyright 2026 Atmanity
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atmee avatar plugin for LiveKit Agents.

Bring your own LiveKit voice agent; Atmee renders its avatar into your room.
Any Atmee avatar renders by its id, v1 or v2 (create avatars in the Atmee
studio or with the Atmee API)::

    from livekit.plugins import atmee

    avatar = atmee.AvatarSession(avatar_id="...")       # ATMEE_API_KEY in the environment
    await avatar.start(session, room=ctx.room)
    await session.start(agent=..., room=ctx.room)       # the agent's TTS drives the avatar's video

See https://docs.livekit.io/agents/models/avatar/plugins/atmee/ for more information.
"""

from .api import (
    AtmeeAPI,
    AtmeeAvatarNotReadyError,
    AtmeeAvatarWarmingError,
    AtmeeException,
    AtmeeNoCapacityError,
    AvatarInfo,
    AvatarSessionInfo,
)
from .avatar import AvatarSession
from .version import __version__

__all__ = [
    "AtmeeAPI",
    "AtmeeAvatarNotReadyError",
    "AtmeeAvatarWarmingError",
    "AtmeeException",
    "AtmeeNoCapacityError",
    "AvatarInfo",
    "AvatarSession",
    "AvatarSessionInfo",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class AtmeePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AtmeePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
