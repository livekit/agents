import asyncio
import logging
import re
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

from livekit import rtc
from livekit.agents import (
    ChatMessage,
    ConversationItemAddedEvent,
    ModelSettings,
    RunContext,
    UserStateChangedEvent,
    VoiceSession,
    get_job_context,
)

logger = logging.getLogger("agent")

CHECK_IN = "The user has been away for a while. Check in with them to see if they are still there."


def check_in_when_user_away(session: VoiceSession[Any]) -> None:
    """Ask whether the user is still there once they go quiet.

    The user turns "away" after `user_away_timeout` seconds of silence (15 by default) and back
    only when they speak, so this asks once per quiet spell.
    """

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "away":
            session.generate_reply(instructions=CHECK_IN, allow_interruptions=True)


MAX_TURNS_CHECKED = 3
_NOT_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize(text: str) -> str:
    """Lowercase alphanumerics with single spaces, so STT punctuation and casing don't matter."""
    return " ".join(_NOT_ALNUM.sub("", text.lower()).split())


def load_scripted_questions() -> frozenset[str]:
    path = Path(__file__).with_name("scripted_questions.txt")
    lines = (line.strip() for line in path.read_text().splitlines())
    return frozenset(normalize(line) for line in lines if line and not line.startswith("#"))


SCRIPTED_QUESTIONS = load_scripted_questions()


def disconnect_scripted_callers(session: VoiceSession[Any]) -> None:
    """Hang up on automated callers, which open with a line from `scripted_questions.txt`.

    Only committed user turns are matched, since STT can split one question into several
    segments, and only the first `MAX_TURNS_CHECKED`, so a real conversation is never cut off.
    """
    checked = 0

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        nonlocal checked
        item = ev.item
        if not isinstance(item, ChatMessage) or item.role != "user" or checked >= MAX_TURNS_CHECKED:
            return
        checked += 1
        text = item.text_content or ""
        if normalize(text) not in SCRIPTED_QUESTIONS:
            return

        turn, checked = checked, MAX_TURNS_CHECKED
        logger.info(
            "disconnecting scripted caller",
            extra={"lk.pii.user_input": text, "caller_turn": turn},
        )
        get_job_context().delete_room()
        session.interrupt(force=True)


_in_flight: set[asyncio.Task[None]] = set()


def publish_tts_voice(label: str) -> None:
    """Publish the TTS voice as the `tts_voice` attribute, for the homepage dev panel.

    The panel reads the rest of the session config from the `lk.agent.session` stream, which
    has no voice. Publish nothing else: attributes override the stream and would freeze live
    values.
    """
    ctx = get_job_context()

    async def publish() -> None:
        await ctx.connect()
        await ctx.room.local_participant.set_attributes({"tts_voice": label})

    task = asyncio.create_task(publish())
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


LIVEKIT_PRONUNCIATION = "Lyve Kit"
LIVEKITS_PRONUNCIATION = "Lyve Kit's"
_LIVEKIT_RE = re.compile(r"\blivekit(?P<possessive>['’]s)?\b", re.IGNORECASE)


async def respell_livekit(chunks: AsyncIterable[str]) -> AsyncIterable[str]:
    """Respell "LiveKit" on whole words, since LLM chunks can split it into "Live" and "Kit"."""

    def replace(match: re.Match[str]) -> str:
        return LIVEKITS_PRONUNCIATION if match.group("possessive") else LIVEKIT_PRONUNCIATION

    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        while " " in buffer:
            word, buffer = buffer.split(" ", 1)
            yield _LIVEKIT_RE.sub(replace, word + " ")
    if buffer:
        yield _LIVEKIT_RE.sub(replace, buffer)


def pronounce_livekit(session: VoiceSession[Any]) -> None:
    """Make the TTS say "LiveKit" right. Only the speech changes; the transcript keeps "LiveKit"."""

    @session.tts_node
    def _speak(
        ctx: RunContext[Any], text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        return session.default.tts_node(ctx, respell_livekit(text), model_settings)
