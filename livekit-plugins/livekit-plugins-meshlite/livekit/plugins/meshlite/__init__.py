# Copyright 2026 LiveKit, Inc.
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

"""MeshLite avatar plugin for LiveKit Agents.

MeshLite renders a lip-synced avatar in the visitor's own browser, with the
MeshLite web SDK. The agent keeps its own STT, LLM and TTS (or realtime model);
this plugin routes the agent's voice to the visitor's page::

    from livekit.plugins import meshlite

    avatar = meshlite.AvatarSession()
    await avatar.start(session, room=ctx.room)
    await session.start(agent=..., room=ctx.room)

See https://meshlite.io/docs#livekit for the page side.
"""

from .avatar import DEFAULT_SAMPLE_RATE, AvatarSession
from .version import __version__

__all__ = ["AvatarSession", "DEFAULT_SAMPLE_RATE", "__version__"]

from livekit.agents import Plugin

from .log import logger


class MeshLitePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(MeshLitePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
