"""The call as the demo website draws it: livekit.io's telephony demo page, when it's watching.

The page joins the call's room as the browser caller (`caller-*`), or as a viewer (`viewer-*`)
of a call to the demo's phone number or of one the agent placed to a real phone. It renders
whatever arrives over RPC: the call's stage, the service desk's state, a log of the agent's tool
calls, and a suggested next line for the caller. This module keeps that state and pushes it.

Without a page (console, the agents playground, simulations, tests) it's inert: nothing is
sent, and no suggestion is written, so there are no extra LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend import ServiceDesk

from livekit import api, rtc
from livekit.agents import AgentSession, inference, llm
from livekit.agents.voice import AgentStateChangedEvent, ConversationItemAddedEvent

logger = logging.getLogger("vehicle-service")

# the page's name for this demo
SCENARIO = "auto_service"
STATE_RPC_METHOD = "demo.set_state"
# the page in the browser: the caller on a browser call, or a viewer of a phone call
CALLER_PREFIX = "caller-"
VIEWER_PREFIX = "viewer-"
# why a phone the agent rang never answered, read off the agent before the room closes
CALL_ERROR_ATTRIBUTE = "call.error"


@dataclass(frozen=True)
class Stage:
    id: str
    label: str


# the steps the page shows; the desk's state says which one the call is at (ServiceDesk.stage)
STAGES = [
    Stage("greeting", "Greeting"),
    Stage("customer", "Find customer"),
    Stage("vehicle", "Vehicle"),
    Stage("symptoms", "Symptoms"),
    Stage("service", "Service"),
    Stage("recalls", "Recalls"),
    Stage("time", "Time & ride"),
    Stage("readback", "Confirm"),
    Stage("booked", "Booked"),
    # the follow-up call, once the technician has found the cause
    Stage("reach", "Reach customer"),
    Stage("quote", "Findings"),
    Stage("pickup", "Ready & pickup"),
    Stage("wrapup", "Wrap up"),
]


@dataclass(frozen=True)
class LogEntry:
    at: str
    actor: str
    tool: str
    detail: str


class Page:
    """What the page shows, sent whenever it changes to the page and every viewer."""

    def __init__(self, persona: str, desk: ServiceDesk) -> None:
        self._desk = desk
        self.persona = persona  # the agent's name, as the caller hears it
        self.direction = "outbound" if desk.visit else "inbound"
        self.stage = STAGES[0].id
        self.data: dict[str, Any] = {}
        self.log: list[LogEntry] = []
        self.end_reason: str | None = None
        self.suggestions: list[str] = []  # what the caller might say next
        self._room: rtc.Room | None = None
        self._page: str | None = None
        self._pushed = False
        self._tasks: set[asyncio.Task[Any]] = set()

    def watch(
        self,
        room: rtc.Room,
        session: AgentSession[Any],
        *,
        page: str | None = None,
        caller: rtc.RemoteParticipant | None = None,
    ) -> None:
        """Pushes to the page from now on: `page` names it when the agent rings a phone (the
        job metadata's `page`); on a browser call it's the caller. Viewers are found as they
        join."""
        if page is None and caller and caller.identity.startswith(CALLER_PREFIX):
            page = caller.identity
        self._room, self._page = room, page
        room.on("participant_connected", self._on_participant_connected)
        _suggest_replies(session, self)

    @property
    def watching(self) -> bool:
        return bool(self._pages())

    # ---- what the agent tells it ----

    def push(self, stage: str | None = None) -> None:
        """Copies the desk into the page's state, moves to `stage` if given, and sends it."""
        self.data = {**self.data, **self._desk.snapshot()}
        if stage and self.stage != "ended":
            self.stage = stage
        self._send()

    def record(self, tool: str, detail: str) -> None:
        """Logs a tool call for the page, e.g. ("select_vehicle", "2023 BMW X3 xDrive30i").
        The stage follows the desk, so a correction that undoes a step moves it back too."""
        self._log(tool, detail)
        self.push(self._desk.stage())

    async def ended(self, reason: str) -> None:
        """The agent is hanging up: the page leaves on this, before the room closes."""
        self.stage = "ended"
        self.end_reason = reason
        self._log("end_call", reason)
        self.push()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def dial_failed(self, error: api.TwirpError, dial: str) -> None:
        """Tells the page why the phone it asked to ring never did, before the room closes."""
        if not self.watching or self._room is None:
            return
        await self._room.local_participant.set_attributes(
            {CALL_ERROR_ATTRIBUTE: dial_error(error, dial)}
        )
        await asyncio.sleep(3)

    def _log(self, tool: str, detail: str) -> None:
        at = datetime.now().strftime("%H:%M:%S")
        self.log.append(LogEntry(at=at, actor=self.persona, tool=tool, detail=detail))

    # ---- sending ----

    def payload(self) -> str:
        return json.dumps(
            {
                "scenario": SCENARIO,
                "direction": self.direction,
                "persona": self.persona,
                "stages": [{"id": s.id, "label": s.label} for s in STAGES],
                "stage": self.stage,
                "data": self.data,
                "log": [
                    {"at": e.at, "actor": e.actor, "tool": e.tool, "detail": e.detail}
                    for e in self.log
                ],
                "endReason": self.end_reason,
                "suggestions": self.suggestions,
            },
            default=str,
        )

    def _pages(self) -> list[str]:
        if self._room is None:
            return []
        viewers = [
            p.identity
            for p in self._room.remote_participants.values()
            if p.identity.startswith(VIEWER_PREFIX)
        ]
        # the page that asked to ring a phone is a viewer too: one push each
        return list(dict.fromkeys([self._page, *viewers] if self._page else viewers))

    def _send(self) -> None:
        self._pushed = True
        if pages := self._pages():
            payload = self.payload()
            for identity in pages:
                self._send_soon(identity, payload)

    def _on_participant_connected(self, p: rtc.RemoteParticipant) -> None:
        # a viewer who joins mid-call sees it as it stands
        if p.identity.startswith(VIEWER_PREFIX) and self._pushed:
            self._send_soon(p.identity, self.payload())

    def _send_soon(self, identity: str, payload: str) -> None:
        task = asyncio.create_task(self._rpc(identity, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _rpc(self, identity: str, payload: str) -> None:
        assert self._room is not None
        try:
            await self._room.local_participant.perform_rpc(
                destination_identity=identity, method=STATE_RPC_METHOD, payload=payload
            )
        except Exception:
            logger.warning("state push to %s failed", identity, exc_info=True)


def visitor_name(caller: rtc.RemoteParticipant | None) -> str:
    """The name a website visitor gave, on a browser call; "" for anyone else."""
    if caller and caller.identity.startswith(CALLER_PREFIX):
        return caller.name
    return ""


def dial_error(e: api.TwirpError, dial: str) -> str:
    """Why a call to a phone couldn't be placed, the way the page says it."""
    sip = e.metadata or {}
    status = sip.get("sip_status_code", "")
    if status in ("408", "480", "487"):
        return "The phone didn't answer."
    if status in ("486", "600"):
        return "The line was busy."
    if status == "603":
        return "The call was declined."
    detail = f"SIP {status}: {sip.get('sip_status', '')}" if status else e.message
    return f"The call couldn't be placed to {dial} ({detail})."


SUGGEST_PROMPT = """\
You help someone trying out a phone demo. Given the call so far, write the one most useful,
natural thing the caller could say next: usually the answer to what the agent just asked. Only
correct something the caller actually said earlier; never question the agent's own name or
company. Write it as the caller, in English, in at most twelve words.
Never invent the caller's personal details (name, date of birth, phone, address, ID numbers,
income): where one is needed, leave a gap with "…", e.g. "It's …, born …". Details the caller
already said may be reused.
It must be something the agent can actually accept. Any choice it names (an insurance plan, a
provider, a day or time, a menu item or size, a car, a service, a program, an amount) must be
one that exists in the business's facts below or that the agent or a tool result just offered,
named the way they name it. Never make up an option, a plan, or an item that isn't there.
Reply with only a JSON array holding that one string.

The business's facts and rules (the agent's instructions):
{facts}"""


def _suggest_replies(session: AgentSession[Any], page: Page) -> None:
    """After each agent turn, writes what the caller might say next, while a page watches."""
    model: llm.LLM | None = None
    pending: list[asyncio.Task[None]] = []

    async def write(history: str) -> None:
        nonlocal model
        model = model or inference.LLM("openai/gpt-4.1-mini")
        facts = str(session.current_agent.instructions) if session.current_agent else ""
        chat = llm.ChatContext()
        chat.add_message(role="system", content=SUGGEST_PROMPT.format(facts=facts))
        chat.add_message(role="user", content=history)
        text = ""
        async with model.chat(chat_ctx=chat) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
        try:
            lines = json.loads(text[text.index("[") : text.rindex("]") + 1])
        except ValueError:
            return
        page.suggestions = [str(s).strip() for s in lines if str(s).strip()][:1]
        page.push()

    def clear() -> None:
        # a suggestion is for the turn it was written for: drop it the moment that turn is over,
        # so the page never offers a stale line while the new one is being written
        for task in pending:
            task.cancel()
        pending.clear()
        if page.suggestions:
            page.suggestions = []
            page.push()

    @session.on("agent_state_changed")
    def _on_agent_state(ev: AgentStateChangedEvent) -> None:
        if ev.new_state in ("thinking", "speaking"):
            clear()

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            clear()
            return
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        if page.stage == "ended" or not page.watching:
            return
        # the last few turns, with what the tools returned (the real slots, cars, and prices)
        lines: list[str] = []
        for it in session.history.items[-16:]:
            if isinstance(it, llm.ChatMessage) and it.role in ("assistant", "user"):
                if it.text_content:
                    who = "Agent" if it.role == "assistant" else "Caller"
                    lines.append(f"{who}: {it.text_content}")
            elif isinstance(it, llm.FunctionCallOutput) and not it.is_error:
                lines.append(f"Tool result ({it.name}): {it.output[:600]}")
        # only the newest turn's suggestion matters
        for task in pending:
            task.cancel()
        pending.clear()
        task = asyncio.create_task(write("\n".join(lines)))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        pending.append(task)
