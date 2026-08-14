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

"""The Microsoft Teams call, attached to your AgentSession.

StandIn owns the whole Teams side: it answers the call, creates the LiveKit
room with the credentials you configured in the portal, dispatches your agent
into it by name, publishes the caller's audio into the room, and relays your
agent's audio back to Teams. By the time your entrypoint runs, the call is an
ordinary LiveKit room - the caller's voice is a room track like any other
participant's, so the session needs no special audio wiring.

What is left for the plugin is the Teams-specific context, and that is all
:class:`TeamsCall` does:

    @server.rtc_session(agent_name="msteams-agent")
    async def entrypoint(ctx: JobContext):
        session = AgentSession(llm=...)
        call = await standin.TeamsCall().start(session, ctx=ctx)
        await session.start(agent=MyAgent(call), room=ctx.room)

It reads :class:`CallInfo` from the job metadata StandIn dispatched with, and
wires the two Teams data topics onto the session:

    msteams.context   non-interrupting context (participants, DTMF, speakers)
    msteams.goodbye   StandIn is ending the call and wants this line spoken
                      first; the default handler interrupts the turn and says it

Nothing here speaks the Teams media protocol, listens on a port, or dials a
socket. That side lives with StandIn, unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from livekit import rtc

from ._exceptions import StandInError
from .log import logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession, JobContext

#: The data topics StandIn publishes into the room, both carrying ``{"text": ...}``.
TOPIC_CONTEXT = "msteams.context"
TOPIC_GOODBYE = "msteams.goodbye"


@dataclass(frozen=True)
class CallInfo:
    """The Teams call this job is answering.

    StandIn attaches it as job metadata when it dispatches the agent, so it is
    available before the first audio frame::

        call = standin.CallInfo.from_job(ctx)
        if call.is_teams_call:
            print(call.caller_name, call.direction)
    """

    caller_name: str = ""
    tenant_id: str = ""
    call_id: str = ""
    thread_id: str = ""
    #: The caller's AAD object id, when Teams reported one. Empty for guest and
    #: anonymous callers, so never use it as a bare key for per-caller memory
    #: without checking it first, or two anonymous callers share one identity.
    user_id: str = ""
    direction: str = "inbound"
    _source: str = ""

    @property
    def is_teams_call(self) -> bool:
        """False when this job was dispatched by something other than StandIn,
        which is the normal case in a worker that also serves web or SIP rooms."""
        return self._source == "msteams"

    @classmethod
    def from_job(cls, ctx: JobContext | None) -> CallInfo:
        """Read the metadata StandIn attached when it dispatched this job.

        Never raises. A job dispatched by anything else has no metadata, or
        metadata in someone else's shape, and both must read as "not a Teams
        call" rather than taking the worker down.
        """
        raw = getattr(getattr(ctx, "job", None), "metadata", "") or ""
        try:
            data: Any = json.loads(raw)
        except ValueError:
            return cls()
        if not isinstance(data, dict) or data.get("source") != "msteams":
            return cls()

        def field(name: str) -> str:
            value = data.get(name)
            return value if isinstance(value, str) else ""

        return cls(
            caller_name=field("caller_name"),
            tenant_id=field("tenant_id"),
            call_id=field("call_id"),
            thread_id=field("thread_id"),
            user_id=field("user_id"),
            direction=field("call_direction") or "inbound",
            _source="msteams",
        )


class TeamsCall:
    """Attach the Microsoft Teams call this job was dispatched for.

    Args:
        on_context: called with the context text whenever StandIn publishes an
            update (participant changes, DTMF digits, speaker changes). By
            default the text is only logged; the audio already flows without it.
        on_goodbye: called with the line StandIn wants spoken before it ends the
            call. The default interrupts the current turn and says it, which is
            what you want: the call is torn down shortly after, so a goodbye
            queued behind a long answer never reaches the caller.

    Example:
        ```python
        session = AgentSession(llm=...)
        call = await standin.TeamsCall().start(session, ctx=ctx)
        await session.start(agent=MyAgent(call), room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        on_context: Callable[[str], Any] | None = None,
        on_goodbye: Callable[[str], Any] | None = None,
    ) -> None:
        self._on_context = on_context
        self._on_goodbye = on_goodbye
        self._session: AgentSession | None = None
        self._info: CallInfo | None = None

    @property
    def info(self) -> CallInfo:
        """Who is calling, and about what. Available after :meth:`start`."""
        if self._info is None:
            raise StandInError("TeamsCall.start() has not run yet")
        return self._info

    async def start(
        self,
        session: AgentSession,
        *,
        ctx: JobContext | None = None,
        room: rtc.Room | None = None,
    ) -> CallInfo:
        """Bind the Teams context of this job's call to ``session``.

        The caller's audio needs nothing from this method - StandIn publishes it
        into the room and ``session.start(room=ctx.room)`` picks it up like any
        other participant. What is bound here is the Teams-only surface: the
        caller identity and the two data topics.
        """
        info = CallInfo.from_job(ctx)
        if not info.is_teams_call:
            raise StandInError(
                "this job was not dispatched by StandIn, so there is no Teams call to "
                "attach. Guard with CallInfo.from_job(ctx).is_teams_call when one worker "
                "serves both Teams and non-Teams rooms."
            )
        target = room if room is not None else getattr(ctx, "room", None)
        if target is None:
            raise StandInError("TeamsCall.start() needs the job's room (pass ctx= or room=)")

        self._info = info
        self._session = session
        target.on("data_received", self._on_data)
        logger.info("standin: attached to Teams call from %s", info.caller_name or "unknown")
        return info

    # ---- the data topics ----

    def _on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic not in (TOPIC_CONTEXT, TOPIC_GOODBYE):
            return
        try:
            payload = json.loads(packet.data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text:
            return
        if packet.topic == TOPIC_CONTEXT:
            self._handle_context(text)
        else:
            self._handle_goodbye(text)

    def _handle_context(self, text: str) -> None:
        if self._on_context is None:
            logger.debug("standin: call context: %s", text)
            return
        self._invoke(self._on_context, text)

    def _handle_goodbye(self, text: str) -> None:
        if self._on_goodbye is not None:
            self._invoke(self._on_goodbye, text)
            return
        session = self._session
        if session is None:
            return
        # Interrupt: the call ends shortly after this, so a goodbye queued
        # behind a long answer is a goodbye the caller never hears.
        with contextlib.suppress(Exception):
            session.interrupt()
            session.say(text)

    @staticmethod
    def _invoke(handler: Callable[[str], Any], text: str) -> None:
        try:
            result = handler(text)
        except Exception:
            logger.exception("standin: topic handler failed")
            return
        if asyncio.iscoroutine(result):
            task = asyncio.ensure_future(result)
            task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None  # surface, never raise
            )


__all__ = ["TOPIC_CONTEXT", "TOPIC_GOODBYE", "CallInfo", "TeamsCall"]
