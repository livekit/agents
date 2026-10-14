"""What any phone agent here needs, whatever the business: how to talk on a call, pacing for the
voice, and hanging up without cutting the caller off, or once they've gone silent.

The helpers that take a RunContext expect its userdata to carry `page` (page.py), the demo
page's view of the call, which hears about a hang-up before the room closes."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterable
from typing import Any

from page import Page

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    RunContext,
    UserStateChangedEvent,
    get_job_context,
    room_io,
)
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from livekit.plugins import noise_cancellation

logger = logging.getLogger("vehicle-service")

# a caller who's gone quiet: the session marks them away after this many seconds in which
# neither side has spoken (AgentSession's user_away_timeout)
USER_AWAY_TIMEOUT = 12.0
STILL_THERE = "Are you still there?"
SILENT_GOODBYE = "I'll let you go. Give us a call back anytime."
NO_RESPONSE = "No response"

VOICE_RULES = """\
You're on a phone call. Speak in short, plain sentences: no markdown, lists, emoji, or symbols.
Sound like a real person who does this job every day, not a script: use contractions, everyday
words, and quick acknowledgments ("Sure.", "Got it.", "Okay, so..."), and vary your phrasing. When
you look something up, say so the way a person would ("Let me pull that up."). Don't
over-apologize, over-thank, or repeat the caller's words back unless you're confirming a detail.
Write numbers in digits, the way they're normally written ($79, $3,800, 7:30 a.m., Oct 14,
2023, 26,200 miles), never spelled out as words: the voice reads them naturally.
Don't read out confirmation or reference numbers unless asked.
Take dates, times, and numbers however the caller says them, and convert them yourself for
your tools. Never tell the caller what format to use.
If they ask something you don't have the answer to (hours, parking, directions, policies), say
so plainly and offer what you can do; never make it up. Never read out internal IDs or codes.
Keep each turn to one or two sentences. Ask one thing at a time: a single question per turn,
never two requests bundled together ("your name and phone number", "the size and the sauce").
Wait for the answer, then ask the next. If they give two things at once anyway, take both.
The caller can't see anything: never say "look right", "looks good", "as you can see", or "on your screen". Ask "Does that sound right?" instead.
People interrupt, correct themselves, and talk over you. When they do, stop, take the
correction, and carry on from there: never repeat the old value, and never restart a list
from the top when only one item changed. A short "mm-hm" or "yeah" while you talk means
keep going.
When they thank you or say goodbye at the end, answer in kind ("You're welcome, have a good
day.") before you hang up. Never hang up on their last words.
Once you've said goodbye ("Thanks for calling, have a great day!"), call end_call in that same
turn: the call ends after your words finish. Don't wait for a reply to a goodbye, and don't say
goodbye without ending the call."""


_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
# a period that ends an abbreviation ("Dr.", "a.m."), not a sentence
_ABBREVIATION = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|vs|No)|\b[a-zA-Z]\.[a-zA-Z])\.\Z")


async def pause_between_sentences(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """Cartesia runs sentences together; a short break after each one paces it like a person."""
    buf = ""
    async for chunk in text:
        buf += chunk
        while m := _SENTENCE_END.search(buf):
            sentence, buf = buf[: m.start()], buf[m.end() :]
            pause = "" if _ABBREVIATION.search(sentence) else ' <break time="300ms"/>'
            yield f"{sentence}{pause} "
    if buf:
        yield buf


def room_options(participant: NotGivenOr[str] = NOT_GIVEN) -> room_io.RoomOptions:
    return room_io.RoomOptions(
        participant_identity=participant,
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=_cancel_background_voices, auto_gain_control=False
        ),
    )


def _cancel_background_voices(
    params: NoiseCancellationParams,
) -> rtc.NoiseCancellationOptions | None:
    # removes background voices and noise; phone audio is narrowband, so it gets the model
    # trained on calls
    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


def spoken_number(e164: str) -> str:
    """A North American number the way it's said on a voicemail: 415-555-0142."""
    d = re.sub(r"\D", "", e164)[-10:]
    return f"{d[:3]}-{d[3:6]}-{d[6:]}" if len(d) == 10 else e164


async def hang_up(reason: str, page: Page | None = None) -> None:
    logger.info("hanging up: %s", reason)
    if page:
        await page.ended(reason)
    try:
        ctx = get_job_context()
    except RuntimeError:
        return  # a text-only test: there's no room to close
    if ctx.simulation_context():
        return  # the simulator ends the run itself, then grades it; a deleted room skips that
    await asyncio.sleep(0.5)
    ctx.delete_room()


_goodbyes: set[asyncio.Task[None]] = set()


def hang_up_when_silent(session: AgentSession[Any], page: Page) -> None:
    """Asks once if the caller's still there when they go quiet, and hangs up if the line stays
    silent. Fixed lines, not the LLM, so silence never agrees to anything: nothing is booked,
    approved, moved, or cancelled on it. For a live call only, never a simulation or a test.

    The session sets the caller "away" after USER_AWAY_TIMEOUT seconds in which neither side has
    spoken (its user_away_timeout); a second time in a row, with nothing from them since the
    check, ends the call. Speaking again starts the count over."""
    away = 0

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal away
        if ev.new_state == "speaking":
            away = 0
        elif ev.new_state == "away":
            away += 1
            if away == 1:
                session.say(STILL_THERE)
                # back to listening, so the silence after the check is timed afresh
                session.reset_away_timer()
            elif away == 2:
                task = asyncio.create_task(_hang_up_on_silence(session, page))
                _goodbyes.add(task)
                task.add_done_callback(_goodbyes.discard)


async def _hang_up_on_silence(session: AgentSession[Any], page: Page) -> None:
    await session.say(SILENT_GOODBYE, allow_interruptions=False).wait_for_playout()
    await hang_up(NO_RESPONSE, page)


async def end_after_reply(ctx: RunContext[Any], reason: str = "Call complete") -> None:
    """Hangs up, but not on the caller's last words: if they spoke last, say goodbye first."""
    # a goodbye said in this same step lands in the history once it has played
    await ctx.wait_for_playout()
    if _caller_spoke_last(ctx):
        bye = ctx.session.generate_reply(
            instructions=(
                "Answer what they just said in a few warm words, like \"You're welcome, have a "
                'good day," and say goodbye.'
            ),
            tool_choice="none",
        )
        task = asyncio.create_task(_hang_up_after(bye, reason, ctx.userdata.page))
        _goodbyes.add(task)
        task.add_done_callback(_goodbyes.discard)
        return
    await hang_up(reason, ctx.userdata.page)


async def _hang_up_after(speech: SpeechHandle, reason: str, page: Page) -> None:
    await speech.wait_for_playout()
    await hang_up(reason, page)


def _caller_spoke_last(ctx: RunContext[Any]) -> bool:
    messages = ctx.session.history.messages()
    spoken = [m.created_at for m in messages if m.role == "assistant" and m.text_content]
    heard = [m.created_at for m in messages if m.role == "user"]
    return bool(heard) and not any(t > heard[-1] for t in spoken)
