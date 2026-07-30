"""Rewrite "LiveKit" so the TTS pronounces it correctly.

Inworld TTS reads a single word's IPA transcription wrapped in slashes as a
custom pronunciation: https://docs.inworld.ai/tts/capabilities/custom-pronunciation

The rewrite only trusts complete words: LLM chunk boundaries land anywhere,
so a per-chunk regex would both miss a "Live"/"Kit" split across chunks and
falsely match a chunk ending mid-word ("LiveKit" + "ten"). Buffering to
space-terminated words is the smallest unit that is correct without holding
back time-to-first-audio.
"""

import re
from collections.abc import AsyncIterable

LIVEKIT_IPA = "/ˈlaɪvkɪt/"  # noqa: RUF001 - intentional IPA characters
_LIVEKIT_RE = re.compile(r"\blivekit\b", re.IGNORECASE)


async def _whole_words(chunks: AsyncIterable[str]) -> AsyncIterable[str]:
    """Regroup a stream of arbitrary chunks into space-terminated words."""
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        while " " in buffer:
            word, buffer = buffer.split(" ", 1)
            yield word + " "
    if buffer:
        yield buffer


async def pronounce_livekit(text: AsyncIterable[str]) -> AsyncIterable[str]:
    async for word in _whole_words(text):
        yield _LIVEKIT_RE.sub(LIVEKIT_IPA, word)
