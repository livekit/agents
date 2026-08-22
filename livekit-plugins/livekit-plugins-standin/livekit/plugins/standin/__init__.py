# Copyright 2026 Komaa DigiTech
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

"""StandIn plugin for LiveKit Agents.

Answer Microsoft Teams calls and chat messages with a LiveKit agent. StandIn
(https://standin.komaa.com) answers the Teams call and dials this worker; the
plugin answers that dial, creates one LiveKit room per call, dispatches this
worker's own agent into it, and relays the audio both ways.

Your file is shaped like every other agent example - nothing starts except
through cli.run_app(server):

    from livekit.plugins import standin

    server = AgentServer()

    @server.rtc_session(agent_name="msteams-agent")
    async def entrypoint(ctx: JobContext):
        session = AgentSession(llm=...)
        call = await standin.TeamsCall().start(session, ctx=ctx)
        await session.start(agent=MyAgent(call), room=ctx.room)

    if __name__ == "__main__":
        cli.run_app(server)

Importing the plugin registers it with the worker; setting STANDIN_SECRET arms
it. The call listener starts with the worker and stops with it, TeamsCall binds
the Teams-only surface (caller identity, call context, the governor's goodbye)
inside the entrypoint, and ChatChannel answers Teams chat on managed
connections over a socket the worker dials out.

See https://docs.komaa.com/livekit/installation for setup.
"""

from ._exceptions import StandInError
from .bridge import CallBridge
from .call import TOPIC_CONTEXT, TOPIC_GOODBYE, CallInfo, TeamsCall
from .chat import SCHEMA_VERSION, ChatChannel, InboundMessage, build_reply, parse_inbound
from .version import __version__

__all__ = [
    "CallBridge",
    "CallInfo",
    "ChatChannel",
    "InboundMessage",
    "SCHEMA_VERSION",
    "StandInError",
    "TOPIC_CONTEXT",
    "TOPIC_GOODBYE",
    "TeamsCall",
    "__version__",
    "build_reply",
    "parse_inbound",
]

from livekit.agents import Plugin

from .log import logger


class StandInPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(StandInPlugin())

# Arm the zero-wiring startup: every AgentServer gets the worker_started hook,
# which starts the call listener only when STANDIN_SECRET is set. See service.py.
from .service import _install as _service_install  # noqa: E402

_service_install()

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
