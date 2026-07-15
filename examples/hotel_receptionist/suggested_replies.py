"""A sidecar that suggests what the *caller* could say next.

Built specifically for the frontend of the demo at http://livekit.com/agents/hotel-receptionist

After each receptionist turn, a small LLM proposes one to three short replies the
caller might give - so a guest who isn't sure how to respond has a ready line to
speak. The suggestions are published on the agent participant as the
``lk.agent.suggestions`` attribute, so the web frontend can render
clickable nudge chips.

The whole feature is one self-contained object attached in a single line:

SuggestedReplies(
    session,
    ctx.room,
    llm=inference.LLM("google/gemma-4-31b-it"),
    expressive=expressive,
).attach()

"""

from __future__ import annotations

import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import AgentSession, ConversationItemAddedEvent, llm

logger = logging.getLogger("hotel-receptionist.suggestions")

SUGGESTIONS_ATTRIBUTE = "lk.agent.suggestions"
MAX_SUGGESTIONS = 4

_SYSTEM_PROMPT = """\
You help a guest who is ON A PHONE CALL with a hotel receptionist. Given the \
receptionist's most recent line, suggest 1-3 short things the guest could say next. \
This is a demo, so the guest should NEVER be left without something to say - always \
return at least one ready reply.

Return ONLY JSON, no markdown fences, in exactly this shape:
{"suggestions": [{"label": "<=3 words for a button", "value": "the exact words the guest speaks"}]}

How to choose replies:
- Phrase "value" as a natural first-person guest utterance (e.g. "Non-smoking, please.").
- If the receptionist offered explicit choices (yes/no, listed room types, listed \
times), suggest from those.
- If the receptionist asks for the guest's OWN details - name, email, phone, card - \
just make one up so the guest can say it. Use obviously-fake demo values: a name like \
"Alex Morgan", an address at example.com for email, a 555 phone number like \
"415-555-0142", and the test card "4242 4242 4242 4242".
- Do NOT invent facts the HOTEL controls that weren't stated - real prices, room types \
not offered, availability, or confirmation codes. If unsure there, suggest a neutral \
reply like "Yes, that works" or a short clarifying question instead.
- Always return at least one suggestion; never return an empty list.
"""

_EXPRESSIVE_GUIDANCE = """\

Expressive demo mode:
- Give the receptionist a clear emotion to respond to at every turn. Put the emotion \
in the guest's words, not only in the button label.
- For an open-ended greeting, introduce varied situations such as:
  - distressed: "My flight was cancelled and I have nowhere to sleep tonight."
  - excited: "It's my birthday and I'm so excited to celebrate!"
  - nervous: "I'm proposing tonight and I really want everything to be perfect."
- Once a situation is established, continue its emotional thread instead of inventing \
a new story. Even practical answers should carry it naturally: "It's Alex Morgan—thank \
you, I'm so relieved."
- Offer a mix of believable emotions across suggestions, without inventing hotel facts.
"""


def _system_prompt(*, expressive: bool) -> str:
    return _SYSTEM_PROMPT + _EXPRESSIVE_GUIDANCE if expressive else _SYSTEM_PROMPT


class SuggestedReplies:
    """Watches an ``AgentSession`` and publishes per-turn reply suggestions.

    Attach it once after building the session; it does the rest for the life of
    the call.
    """

    def __init__(
        self,
        session: AgentSession,
        room: rtc.Room,
        *,
        llm: llm.LLM,
        expressive: bool = False,
    ) -> None:
        self._session = session
        self._room = room
        self._llm = llm
        self._system_prompt = _system_prompt(expressive=expressive)
        self._task: asyncio.Task[None] | None = None

    def attach(self) -> SuggestedReplies:
        """Subscribe to the session. Returns self so it reads as one expression."""
        self._session.on("conversation_item_added", self._on_item)
        return self

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        # React only to the receptionist's own turns; ignore the caller's.
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        # The newest turn wins: drop any suggestion still being generated for a
        # previous line so we can never publish stale chips.
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._run(item.text_content or ""))

    async def _run(self, assistant_line: str) -> None:
        try:
            await self._suggest(assistant_line)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("suggested-reply generation failed")
            await self._publish([])  # fail safe to no chips, never stale ones

    async def _suggest(self, assistant_line: str) -> None:
        # Clear last turn's chips immediately; we'll set this turn's below.
        await self._publish([])

        ctx = llm.ChatContext()
        ctx.add_message(role="system", content=self._system_prompt)
        ctx.add_message(role="user", content=f'The receptionist just said: "{assistant_line}"')

        stream = self._llm.chat(chat_ctx=ctx)
        raw = ""
        try:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    raw += chunk.delta.content
        finally:
            await stream.aclose()

        await self._publish(_parse(raw))

    async def _publish(self, items: list[dict[str, str]]) -> None:
        # set_attributes merges keys, so this leaves lk.agent.state etc. untouched.
        await self._room.local_participant.set_attributes(
            {SUGGESTIONS_ATTRIBUTE: json.dumps(items)}
        )


def _parse(raw: str) -> list[dict[str, str]]:
    """Pull a clean [{label, value}] list out of the model's reply, or [] on any doubt."""
    data = _loads(raw)
    items = data.get("suggestions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    clean: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        label, value = it.get("label"), it.get("value")
        if isinstance(label, str) and isinstance(value, str) and label.strip() and value.strip():
            clean.append({"label": label.strip(), "value": value.strip()})
    return clean[:MAX_SUGGESTIONS]


def _loads(raw: str) -> object:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        # Salvage a JSON object if the model wrapped it in prose or ``` fences.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except (ValueError, TypeError):
                pass
        return {}
