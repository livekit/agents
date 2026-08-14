"""Rewrite "LiveKit" and "LiveKit's" with TTS-friendly spellings.

The rewrite only trusts complete words: LLM chunk boundaries land anywhere,
so a per-chunk regex would both miss a "Live"/"Kit" split across chunks and
falsely match a chunk ending mid-word ("LiveKit" + "ten"). Buffering to
space-terminated words is the smallest unit that is correct without holding
back time-to-first-audio.
"""

import re
from collections.abc import AsyncIterable

LIVEKIT_PRONUNCIATION = "Lyve Kit"
LIVEKITS_PRONUNCIATION = "Lyve Kit's"
_LIVEKIT_RE = re.compile(r"\blivekit(?P<possessive>['’]s)?\b", re.IGNORECASE)


def _replace_livekit(match: re.Match[str]) -> str:
    return LIVEKITS_PRONUNCIATION if match.group("possessive") else LIVEKIT_PRONUNCIATION


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
        yield _LIVEKIT_RE.sub(_replace_livekit, word)
