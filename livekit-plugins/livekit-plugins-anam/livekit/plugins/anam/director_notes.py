from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import get_args

from livekit import rtc
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .log import logger
from .types import PresetStyle

# Data-packet topic carrying director-note cues to the Anam engine. Must match the engine's LiveKit
# session component. Payload:
#   {"cue": {"tag": <tag>}, "char_offset": <int>}
# char_offset locates the cue within this segment's emitted text.
DIRECTOR_NOTE_CUE_TOPIC = "director_note_cue"

# Default cue allowlist — Anam's built-in styles (:data:`PresetStyle`). Only these `[tag]`s are
# stripped before TTS and forwarded; other brackets (e.g. ElevenLabs audio tags like [whispers])
# pass through to the TTS.
DIRECTOR_NOTE_TAGS: frozenset[str] = frozenset(get_args(PresetStyle))

# A bracket cue tag is a short word in square brackets, e.g. [happy] or [sad].
_TAG_BODY = r"[A-Za-z][A-Za-z0-9_-]{0,63}"
_TAG_RE = re.compile(rf"\[({_TAG_BODY})\]")
# An unterminated tag at the very end of a chunk (e.g. "[hap"), held back so a tag split across a
# streaming-chunk boundary isn't sent to TTS or miscounted.
_PARTIAL_TAG_RE = re.compile(rf"\[(?:{_TAG_BODY})?$")

_TextTransform = Callable[[AsyncIterable[str]], AsyncIterable[str]]


def director_note_cue_transform(
    room: rtc.Room,
    *,
    stripped_tags: frozenset[str] | None = DIRECTOR_NOTE_TAGS,
    forwarded_tags: frozenset[str] | None = DIRECTOR_NOTE_TAGS,
) -> _TextTransform:
    """Build a ``tts_text_transforms`` callable that handles inline ``[tag]`` director-note cues:
    optionally stripping them from the text before TTS and/or forwarding them to the Anam engine,
    keyed by the character offset where each applies in that turn's spoken text.

    Use this only when the TTS would otherwise *speak* the tags aloud. If instead the TTS accepts the
    tags as control tokens — not speaking them, yet still emitting them in its timed transcript with
    word timings (e.g. Cartesia ``sonic-3.5``) — you don't need this transform at all: the engine
    reads each cue and its timing straight from the forwarded transcript, so just forward it. (If you
    keep this in the pipeline anyway, pass both ``stripped_tags`` and ``forwarded_tags`` empty for a
    no-op.)

    ``stripped_tags`` and ``forwarded_tags`` are independent levers, so each tag falls into one of
    four behaviours: stripped + forwarded (a normal director cue), forwarded-only (kept for the TTS
    *and* sent to Anam, e.g. ``[laughter]``), stripped-only (removed but not forwarded), or neither
    (left for the TTS, e.g. an ElevenLabs ``[whispers]`` audio tag). Both default to
    :data:`DIRECTOR_NOTE_TAGS`, so by default the preset styles are cues and everything else passes
    through.

    The engine resolves the cue against the forwarded timed transcript, so keep
    ``use_tts_aligned_transcript=True`` and the JSON transcription sink enabled. Cue packets are
    published on ``DIRECTOR_NOTE_CUE_TOPIC`` as ``{"cue": {"tag": tag}, "char_offset": int}``:
    ``char_offset`` locates the cue within *this segment's* emitted text (each transform call is its
    own coordinate space). Note a forwarded-only tag that the TTS itself strips from its output will
    shift ``char_offset`` for that and later cues in the segment by the tag's length.

    Cues are sent only to the Anam engine — the participant publishing on behalf of this agent
    (``lk.publish_on_behalf``), discovered automatically from ``room``. Publishing waits for it to
    join (exactly as the audio output does); if it never joins there is no avatar at all, so a
    stalled cue is moot.

    Composes with other ``tts_text_transforms`` (they run in series, in list order). Place this one
    **last**: it must see the raw ``[tag]`` markers (so don't put a bracket-rewriting transform
    ahead of it), and its output must be exactly what the TTS speaks (so any transform that changes
    character content should run before it, or the cue offsets drift).

    Example — ElevenLabs, which would otherwise *speak* the tags. The default strips the preset
    styles and forwards them to Anam, while leaving native audio tags (e.g. ``[whispers]``) for the
    TTS::

        session = AgentSession(
            stt=..., llm=..., tts=elevenlabs.TTS(), vad=...,
            use_tts_aligned_transcript=True,
            tts_text_transforms=[anam.director_note_cue_transform(ctx.room)],
        )

    Other scenarios (just the ``tts_text_transforms`` entry shown)::

        # Cartesia sonic-3.5 — accepts the tags, doesn't speak them, and returns them in the timed
        # transcript: omit this transform entirely and forward the transcript.
        tts_text_transforms=[]

        # A tag both Anam and the TTS should act on (e.g. ElevenLabs [laughter] — face *and* voice
        # laugh): forward it but don't strip it, so the TTS still receives it.
        tts_text_transforms=[
            anam.director_note_cue_transform(
                ctx.room, stripped_tags=anam.DIRECTOR_NOTE_TAGS - {"laughter"}
            )
        ]

        # Strip every bracketed tag from the speech, forward none (e.g. to keep stage directions out
        # of the audio without driving Anam):
        tts_text_transforms=[
            anam.director_note_cue_transform(ctx.room, stripped_tags=None, forwarded_tags=frozenset())
        ]

    Args:
        room: The LiveKit room cues are published to (e.g. ``ctx.room``).
        stripped_tags: Tags removed from the text before it reaches the TTS. Defaults to
            :data:`DIRECTOR_NOTE_TAGS`; pass ``None`` to strip every well-formed ``[tag]``, or an
            empty set to strip none.
        forwarded_tags: Tags forwarded to the Anam engine as cues. Defaults to
            :data:`DIRECTOR_NOTE_TAGS`; pass ``None`` to forward every tag, or an empty set to
            forward none.
    """
    strip_set = None if stripped_tags is None else {t.strip().lower() for t in stripped_tags}
    forward_set = None if forwarded_tags is None else {t.strip().lower() for t in forwarded_tags}

    # The Anam engine is whoever publishes on behalf of us. Discover it once (waiting if it hasn't
    # joined yet) and target every cue at it, so cues aren't broadcast to other participants and a
    # cue published before the engine is in the room isn't lost.
    engine: asyncio.Task[str] | None = None

    async def _find_engine() -> str:
        me = room.local_participant.identity

        def _match(p: rtc.RemoteParticipant) -> bool:
            return p.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF) == me

        for p in room.remote_participants.values():
            if _match(p):
                return p.identity

        fut: asyncio.Future[str] = asyncio.Future()

        def _on_connected(p: rtc.RemoteParticipant) -> None:
            if _match(p) and not fut.done():
                fut.set_result(p.identity)

        def _on_attrs(_: list[str], p: rtc.Participant) -> None:
            if isinstance(p, rtc.RemoteParticipant) and _match(p) and not fut.done():
                fut.set_result(p.identity)

        room.on("participant_connected", _on_connected)
        room.on("participant_attributes_changed", _on_attrs)
        try:
            return await fut
        finally:
            room.off("participant_connected", _on_connected)
            room.off("participant_attributes_changed", _on_attrs)

    def _engine_identity() -> asyncio.Task[str]:  # one shared discovery reused by every cue
        nonlocal engine
        if engine is None:
            engine = asyncio.ensure_future(_find_engine())
        return engine

    def _strip(tag: str) -> bool:  # remove from the TTS input
        return strip_set is None or tag in strip_set

    def _forward(tag: str) -> bool:  # forward to Anam as a cue
        return forward_set is None or tag in forward_set

    def _publish(tag: str, char_offset: int) -> None:
        payload = json.dumps({"cue": {"tag": tag}, "char_offset": char_offset})

        async def _send() -> None:
            identity = await _engine_identity()
            try:
                await room.local_participant.publish_data(
                    payload,
                    topic=DIRECTOR_NOTE_CUE_TOPIC,
                    reliable=True,
                    destination_identities=[identity],
                )
            except Exception:
                logger.exception("failed to publish director-note cue")

        # Fire-and-forget so cue delivery never stalls the TTS text stream.
        asyncio.ensure_future(_send())
        logger.debug("director-note cue %r at char %d", tag, char_offset)

    async def _transform(text: AsyncIterable[str]) -> AsyncGenerator[str, None]:
        # char_offset counts emitted chars from the start of this segment (each call is its own
        # coordinate space); the engine locates the cue by that offset in the timed transcript.
        emitted = 0  # emitted-char count so far this segment

        # State resets per call, so a [tag] split across separate TTS generations isn't reassembled.
        # The transform wraps each generation's whole text stream once, so all chunk-boundary splits
        # within a generation are handled here; a tag can't span two independent generations, and its
        # body has no whitespace or sentence punctuation for an upstream tokenizer to split on.
        partial = ""
        async for chunk in text:
            buf = partial + chunk
            m = _PARTIAL_TAG_RE.search(buf)
            buf, partial = (buf[: m.start()], buf[m.start() :]) if m else (buf, "")

            out: list[str] = []
            pos = 0
            for tm in _TAG_RE.finditer(buf):
                piece = buf[pos : tm.start()]
                out.append(piece)
                emitted += len(piece)
                tag = tm.group(1).lower()
                if not _strip(tag):
                    # Kept for the TTS (passthrough, or forwarded-only so both react).
                    out.append(tm.group(0))
                    emitted += len(tm.group(0))
                if _forward(tag):
                    # Offset = next emitted char after the tag (a kept tag is already counted).
                    _publish(tag, emitted)
                pos = tm.end()
            tail = buf[pos:]
            out.append(tail)
            emitted += len(tail)
            yield "".join(out)

        if partial:  # unterminated "[..." left at end of segment — emit as plain text
            yield partial

    return _transform
