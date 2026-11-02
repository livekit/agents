"""Participant attributes for the livekit.io homepage frontend.

The homepage dev panel builds its "Agent configuration" section from the
``lk.agent.session`` byte stream, whose ``TTSModelUsage`` message carries no
voice field. The TTS voice therefore has to reach the frontend as a
participant attribute (``tts_voice``). Everything else is left to the byte
stream — attributes take precedence over it in the frontend, so publishing
more would freeze live-reported values at their startup snapshot.
"""

import asyncio

from livekit.agents import get_job_context

_in_flight: set[asyncio.Task] = set()


def frontend_attributes(*, tts_voice: str | None) -> dict[str, str]:
    """Config the session byte stream can't carry, keyed by frontend attribute name."""
    if not tts_voice:
        return {}
    return {"tts_voice": tts_voice}


def publish_frontend_attributes(*, tts_voice: str | None) -> None:
    attributes = frontend_attributes(tts_voice=tts_voice)
    if not attributes:
        return

    ctx = get_job_context()

    async def publish() -> None:
        await ctx.connect()
        await ctx.room.local_participant.set_attributes(attributes)

    task = asyncio.create_task(publish())
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)
