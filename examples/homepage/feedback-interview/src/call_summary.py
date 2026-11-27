"""Keeps a summary of the interview on the agent participant, for the page to show after the call.

After each of Morgan's turns, a small model reads the call so far and writes the interviewee's
name, their role, and the main points, which become a Markdown list. It's published as the
``interview.summary`` attribute, so the page has the latest one even when the visitor hangs up
first. When Morgan ends the call, ``finish()`` writes a last one before the agent leaves.
"""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.agents import AgentSession, ConversationItemAddedEvent, llm
from pydantic import BaseModel

logger = logging.getLogger("call-summary")

SUMMARY_ATTRIBUTE = "interview.summary"
SUMMARY_MODEL = "google/gemma-4-31b-it"

INSTRUCTIONS = """\
You keep the record of a short interview between Morgan, LiveKit's AI interviewer, and someone \
who evaluated LiveKit. From the call so far, write:
- name: the interviewee's name as they said it, or null if they haven't said it.
- role: their job title or role as they described it, such as "CTO" or "runs engineering", or null if they haven't said it.
- points: the main things they told the team, one short sentence each, usually three to six: \
what they were building, how they found LiveKit, what they compared it with and what they chose, \
what got in the way, and anything else they asked the team to hear. Leave out anything they \
didn't cover. Plain text only, with no bullets, numbering or formatting.
Use only what the interviewee said. Never guess a name or role, and never add Morgan's opinions. \
Refer to the interviewee by name or as "they", never as "he" or "she".
"""


class Points(BaseModel):
    name: str | None
    role: str | None
    points: list[str]


class Summary(BaseModel):
    name: str | None
    role: str | None
    summary: str


class CallSummary:
    def __init__(self, session: AgentSession, room: rtc.Room, *, llm: llm.LLM) -> None:
        self._session = session
        self._room = room
        self._llm = llm
        self._task: asyncio.Task[None] | None = None

    def attach(self) -> CallSummary:
        self._session.on("conversation_item_added", self._on_item)
        return self

    async def finish(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        await self._run()

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, llm.ChatMessage) or ev.item.role != "assistant":
            return
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self._summarize()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("call summary failed")

    async def _summarize(self) -> None:
        transcript = self._transcript()
        if transcript is None:
            return
        ctx = llm.ChatContext()
        ctx.add_message(role="system", content=INSTRUCTIONS)
        ctx.add_message(role="user", content=f"The call so far:\n{transcript}")
        reply = await self._llm.chat(chat_ctx=ctx, response_format=Points).collect()
        found = Points.model_validate_json(reply.text)
        summary = Summary(
            name=found.name, role=found.role, summary=markdown_list(found.points)
        )
        await self._room.local_participant.set_attributes(
            {SUMMARY_ATTRIBUTE: summary.model_dump_json()}
        )

    def _transcript(self) -> str | None:
        messages = [
            item
            for item in self._session.history.items
            if isinstance(item, llm.ChatMessage)
            and item.role in ("user", "assistant")
            and item.text_content
        ]
        if not any(m.role == "user" for m in messages):
            return None
        return "\n".join(
            f"{'Interviewee' if m.role == 'user' else 'Morgan'}: {m.text_content}"
            for m in messages
        )


def markdown_list(points: list[str]) -> str:
    """The points as a Markdown bulleted list, one line each."""
    lines = (" ".join(p.split()).lstrip("-*• ").strip() for p in points)
    return "\n".join(f"- {line}" for line in lines if line)
