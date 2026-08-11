"""Publishes each receptionist turn's raw LLM output, expressive markup intact.

Built for the frontend of the demo at http://livekit.com/agents/hotel-receptionist

The transcript the room sees has expressive markup stripped — the tags drive
TTS delivery, not display. The demo's "Markup" view shows what expressive is
actually doing, so each assistant turn's pre-strip text is republished on the
``lk.raw_generation`` text-stream topic, one message per completed turn.

The whole feature is one self-contained call attached in a single line:

    publish_raw_generations(session, ctx.room)
"""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.agents import AgentSession, ConversationItemAddedEvent, llm

logger = logging.getLogger("hotel-receptionist.raw-generation")

RAW_GENERATION_TOPIC = "lk.raw_generation"


def publish_raw_generations(session: AgentSession, room: rtc.Room) -> None:
    """Republish each assistant turn's raw text (markup intact) to the room."""
    pending: set[asyncio.Task[None]] = set()

    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        # raw_text_content keeps the expressive markers that text_content strips
        text = item.raw_text_content
        if not text:
            return

        async def send() -> None:
            try:
                await room.local_participant.send_text(text, topic=RAW_GENERATION_TOPIC)
            except Exception:
                logger.exception("failed to publish raw generation")

        task = asyncio.create_task(send())
        pending.add(task)
        task.add_done_callback(pending.discard)

    session.on("conversation_item_added", _on_item)
