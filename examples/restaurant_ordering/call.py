"""Phone-call helpers: the speaking rules, the voice's pacing, the audio input, and hanging up
or handing off without cutting the caller off."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterable
from typing import Any

from livekit import rtc
from livekit.agents import RunContext, get_job_context, room_io
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from livekit.plugins import noise_cancellation

logger = logging.getLogger("restaurant-ordering")

VOICE_RULES = """\
You're on a phone call. Speak in short, plain sentences: no markdown, lists, emoji, or symbols.
Sound like a real person who does this job every day, not a script: use contractions, everyday
words, and quick acknowledgments ("Sure.", "Got it.", "Okay, so..."), and vary your phrasing. When
you look something up, say so the way a person would ("Let me pull that up."). Don't
over-apologize, over-thank, or repeat the caller's words back unless you're confirming a detail.
Write numbers in digits, the way they're normally written ($79, $3,800, 7:30 a.m., Oct 14,
2023, 26,200 miles), never spelled out as words: the voice reads them naturally, and the
caller sees them on screen. Don't read out confirmation or reference numbers unless asked.
Take dates, times, and numbers however the caller says them, and convert them yourself for
your tools. Never tell the caller what format to use.
If they ask something you don't have the answer to (hours, parking, directions, policies), say
so plainly and offer what you can do; never make it up. Never read out internal IDs or codes.
Keep each turn to one or two sentences. Ask one thing at a time: a single question per turn,
never two requests bundled together ("your name and phone number", "the size and the sauce").
Wait for the answer, then ask the next. If they give two things at once anyway, take both.
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


async def hang_up(reason: str) -> None:
    logger.info("hanging up: %s", reason)
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


async def transfer(ctx: RunContext[Any], who: str, reason: str) -> None:
    """Hands the call to `who` ("someone at the restaurant"), saying so first.

    This example has no one to hand off to, so the call ends there. To reach a real person,
    move the caller with `ctx.api.sip.transfer_sip_participant` instead of hanging up.
    """
    logger.info("transferring to %s: %s", who, reason)
    await ctx.wait_for_playout()
    # never a silent drop: if the caller spoke last, say where they're going first
    if _caller_spoke_last(ctx):
        await ctx.session.say(f"One moment, I'm connecting you with {who}.").wait_for_playout()
    await hang_up(f"Transferred to {who}")


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
        task = asyncio.create_task(_hang_up_after(bye, reason))
        _goodbyes.add(task)
        task.add_done_callback(_goodbyes.discard)
        return
    await hang_up(reason)


async def _hang_up_after(speech: SpeechHandle, reason: str) -> None:
    await speech.wait_for_playout()
    await hang_up(reason)
