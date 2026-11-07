"""Suggests what the caller could say next, for the demo page's "try saying" line.

This is plumbing for the homepage demo page, not part of the voice agent: the agent works the
same without it, and never reads what it publishes. Many people trying the demo don't know
what to say to a pizzeria or a service desk, so after each of the agent's turns a second, small
model reads the call so far and the caller's script below (who they are and what they want
right now) and proposes one to three short lines the caller could say next. They're published
on the agent participant as the ``lk.agent.suggestions`` attribute, which the page shows as its
"try saying" line and lets the visitor click to send as text.

Attached in one line after the session is built:

    SuggestedReplies(session, ctx.room, llm=inference.LLM(SUGGESTIONS_MODEL),
                     persona="Nick", script=lambda: caller_script(desk)).attach()
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from livekit import rtc
from livekit.agents import AgentSession, ConversationItemAddedEvent, llm
from pydantic import BaseModel

logger = logging.getLogger("suggested-replies")

SUGGESTIONS_ATTRIBUTE = "lk.agent.suggestions"
SUGGESTIONS_MODEL = "google/gemma-4-31b-it"
MAX_SUGGESTIONS = 3
RECENT_TURNS = 8

INSTRUCTIONS = """\
You help the CALLER on a phone call with {persona}. Given the call so far, suggest 1-3 short \
things the caller could say next, so someone unsure what to say has a line ready. Each has a \
label of three words or fewer, for a button, and the exact words the caller says.

How to choose:
- The words are a natural first-person line, one or two short sentences, answering exactly what {persona} just asked and nothing more.
- The caller reads the line out loud, so write it the way people speak: no symbols or \
abbreviations, "forty-two dollars and eighteen cents" rather than "$42.18".
- The caller's details come only from the script below. Never invent any other name, date, number, or amount.
- If {persona} is reading something back to confirm it, the reply is a short yes, or the correction from the script.
- If {persona} offered choices, pick from them, in {persona}'s words.
- Put the most useful reply first.
- While {persona} is still doing something they said they'd do ("one sec", "pulling that up", \
"let me check"), the caller just waits: suggest nothing. Otherwise always suggest at least one.

{script}
"""


class Suggestion(BaseModel):
    label: str
    value: str


class Suggestions(BaseModel):
    suggestions: list[Suggestion]


class SuggestedReplies:
    def __init__(
        self,
        session: AgentSession,
        room: rtc.Room,
        *,
        llm: llm.LLM,
        persona: str,
        script: Callable[[], str | None],
    ) -> None:
        self._session = session
        self._room = room
        self._llm = llm
        self._persona = persona
        self._script = script
        self._task: asyncio.Task[None] | None = None

    def attach(self) -> SuggestedReplies:
        self._session.on("conversation_item_added", self._on_item)
        return self

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        # The caller is talking over the agent, maybe reading the line out: leave it up. Their turn
        # gets a reply of its own, and that reply brings the next suggestions.
        if item.interrupted or self._session.user_state == "speaking":
            return
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._run(item.text_content or ""))

    async def _run(self, line: str) -> None:
        try:
            await self._suggest(line)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("suggested replies failed")

    async def _suggest(self, line: str) -> None:
        await self._publish([])
        script = self._script()
        if script is None:
            return
        ctx = llm.ChatContext()
        ctx.add_message(
            role="system",
            content=INSTRUCTIONS.format(persona=self._persona, script=script),
        )
        ctx.add_message(role="user", content=self._prompt(line))
        reply = await self._llm.chat(
            chat_ctx=ctx, response_format=Suggestions
        ).collect()
        if self._session.user_state == "speaking":
            return
        suggestions = Suggestions.model_validate_json(reply.text).suggestions
        await self._publish([s.model_dump() for s in suggestions[:MAX_SUGGESTIONS]])

    def _prompt(self, line: str) -> str:
        turns = [
            f"{'Caller' if item.role == 'user' else self._persona}: {item.text_content}"
            for item in self._session.history.items
            if isinstance(item, llm.ChatMessage)
            and item.role in ("user", "assistant")
            and item.text_content
        ]
        recent = "\n".join(turns[-RECENT_TURNS:])
        return (
            f"The call so far:\n{recent}\n\n"
            f'{self._persona} just said: "{line}"\n'
            "What does the caller say next?"
        )

    async def _publish(self, items: list[dict[str, str]]) -> None:
        await self._room.local_participant.set_attributes(
            {SUGGESTIONS_ATTRIBUTE: json.dumps(items)}
        )
