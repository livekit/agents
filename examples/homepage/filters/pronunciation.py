"""Rewrite "LiveKit" and "LiveKit's" so the TTS pronounces them correctly.

Fish Audio reads one word of CMU Arpabet per phoneme tag as a custom pronunciation:
https://docs.fish.audio/developer-guide/core-features/fine-grained-control#phoneme-control

The rewrite only trusts complete words: LLM chunk boundaries land anywhere,
so a per-chunk regex would both miss a "Live"/"Kit" split across chunks and
falsely match a chunk ending mid-word ("LiveKit" + "ten"). Buffering to
space-terminated words is the smallest unit that is correct without holding
back time-to-first-audio.
"""

import re
from collections.abc import AsyncIterable

LIVEKIT_PHONEMES = "<|phoneme_start|>L AY1 V<|phoneme_end|> <|phoneme_start|>K IH1 T<|phoneme_end|>"
LIVEKITS_PHONEMES = (
    "<|phoneme_start|>L AY1 V<|phoneme_end|> <|phoneme_start|>K IH1 T S<|phoneme_end|>"
)
_LIVEKIT_RE = re.compile(r"\blivekit(?P<possessive>['’]s)?\b", re.IGNORECASE)


def _replace_livekit(match: re.Match[str]) -> str:
    return LIVEKITS_PHONEMES if match.group("possessive") else LIVEKIT_PHONEMES


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
