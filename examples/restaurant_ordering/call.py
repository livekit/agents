"""Phone-call helpers: the speaking rules, the audio input, and hanging up
without cutting the caller off, or when they've gone silent.

The helpers that take a RunContext expect its userdata to carry `page` (page.py), the demo
page's view of the call, which hears about a hang-up before the room closes."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from page import Page

from livekit import rtc
from livekit.agents import AgentSession, RunContext, get_job_context, room_io
from livekit.agents.voice import SpeechHandle, UserStateChangedEvent
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from livekit.plugins import noise_cancellation

logger = logging.getLogger("restaurant-ordering")

# how long neither side has spoken before the caller counts as away (AgentSession's
# user_away_timeout); the first time, the agent checks in, the second, it hangs up
USER_AWAY_TIMEOUT = 20.0
STILL_THERE = "Are you still there?"
SILENT_GOODBYE = "I'll let you go. Call us back anytime."

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
If they ask something you don't have the answer to (parking, directions, policies), say
so plainly and offer what you can do; never make it up. Never read out internal IDs or codes.
Keep each turn to one or two sentences. Ask one thing at a time: a single question per turn,
never two requests bundled together ("your name and phone number", "the size and the sauce").
Wait for the answer, then ask the next. If they give two things at once anyway, take both.
The caller can't see anything: never say "look right", "looks good", "as you can see", or "on your screen"; when you do check something with them, it's "Does that sound right?". Don't repeat what you said in your last turn (a total, a time, how they pay): say it once, and again only if they ask.
Never say you've noted, saved, added, booked, changed, or passed on something without calling
the tool for it in that same reply, never a turn later. If no tool does it, say plainly that
you can't.
People interrupt, correct themselves, and talk over you. When they do, stop, take the
correction, and carry on from there: never repeat the old value, and never restart a list
from the top when only one item changed. A short "mm-hm" or "yeah" while you talk means
keep going.
When they thank you or say goodbye at the end, answer in kind ("You're welcome, have a good
day.") before you hang up. Never hang up on their last words.
Once you've said goodbye ("Thanks for calling, have a great day!"), call end_call in that same
turn: the call ends after your words finish. Don't wait for a reply to a goodbye, and don't say
goodbye without ending the call."""


def room_options(participant: str) -> room_io.RoomOptions:
    """Listen to `participant` only, with background voices removed."""
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


def first_name(name: str) -> str:
    return name.split()[0] if name.strip() else "there"


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


def _caller_spoke_last(ctx: RunContext[Any]) -> bool:
    messages = ctx.session.history.messages()
    spoken = [m.created_at for m in messages if m.role == "assistant" and m.text_content]
    heard = [m.created_at for m in messages if m.role == "user"]
    return bool(heard) and not any(t > heard[-1] for t in spoken)


_goodbyes: set[asyncio.Task[None]] = set()


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


def hang_up_when_silent(session: AgentSession[Any], page: Page | None = None) -> None:
    """Checks in once when the line goes quiet, and hangs up if it stays quiet.

    The session's user state turns "away" once neither side has spoken for its
    user_away_timeout. The first time, the agent asks if they're still there, in a fixed line;
    the second time in a row, with no word from the caller in between, it says goodbye and
    hangs up. Silence never agrees to anything: no tool runs on it. Only for real calls: a
    simulation or a text-only test never arms it."""
    away = 0
    tasks: set[asyncio.Task[None]] = set()

    async def check_in() -> None:
        session.say(STILL_THERE)
        # back to listening, so another full quiet spell after the question counts again
        session.reset_away_timer()

    async def goodbye() -> None:
        await session.say(SILENT_GOODBYE, allow_interruptions=False).wait_for_playout()
        await hang_up("No response", page)

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal away
        if ev.new_state == "speaking":
            away = 0
            return
        if ev.new_state != "away" or away >= 2:
            return
        away += 1
        task = asyncio.create_task(check_in() if away == 1 else goodbye())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
