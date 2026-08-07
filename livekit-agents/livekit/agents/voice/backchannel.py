"""Phrase-based backchannel classification for interruption handling.

When a user says "okay", "thank you", or "uh-huh" while the agent is
talking, they usually aren't asking the agent to stop — they're
acknowledging ("backchanneling"). The adaptive interruption detector
classifies overlapping speech acoustically; this module is the
transcript-content counterpart for the STT path: an overlapping utterance
whose words are entirely backchannel phrases (plus filler sounds) neither
interrupts the agent's speech nor commits a user turn.

Enabled with ``InterruptionOptions["backchannel_phrases"]``; disabled by
default. ``DEFAULT_BACKCHANNEL_PHRASES`` is a curated English starter list.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

DEFAULT_BACKCHANNEL_PHRASES: tuple[str, ...] = (
    "makes sense",
    "sounds good",
    "thank you",
    "got it",
    "i see",
    "mm hmm",
    "uh huh",
    "gotcha",
    "mhm",
    "mmhmm",
    "thanks",
    "okay",
    "ok",
    "yeah",
    "yep",
    "yes",
    "right",
    "sure",
    "alright",
    "cool",
)
"""A curated list of English acknowledgment phrases suitable for
``InterruptionOptions["backchannel_phrases"]``. Deliberately absent:
"no", "wait", "stop", "what", "hello", and a bare "huh" — those are
real barge-ins ("uh huh" the backchannel still matches as a phrase).
"""

# Filler sounds are skipped between phrases, and a filler-only utterance
# ("uh", "um hm") counts as a backchannel: with ``min_words`` low, a lone
# filler passes the word-count gate and would otherwise cut the agent.
_FILLERS = frozenset({"uh", "um", "hm", "hmm", "mm", "ah", "oh"})


def _normalize(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()


@lru_cache(maxsize=8)
def _split_phrases(phrases: tuple[str, ...]) -> list[list[str]]:
    # longest-first so "uh huh" wins over a lone filler "uh"
    return sorted((_normalize(p) for p in phrases), key=len, reverse=True)


def is_backchannel_only(text: str, phrases: Sequence[str], *, partial: bool = False) -> bool:
    """True when ``text`` consists solely of backchannel ``phrases`` and
    filler sounds — i.e. the user is acknowledging, not interrupting.

    Empty text returns ``False``: with no words yet the decision belongs to
    the ``min_duration``/``min_words`` gates, not to the phrase filter.

    ``partial=True`` is for judging a LIVE interim transcript (the
    interruption path): a trailing proper prefix of a known phrase
    ("thank" → "thank you") defers the cut instead of committing it —
    interims arrive word by word, so without this the first word of every
    multi-word backchannel would cut the agent before the phrase finished.
    Deferring is safe because every subsequent interim re-judges the full
    text: "thank god you called" cuts one interim later. Final transcripts
    are complete utterances and must use exact matching (``partial=False``).
    """
    words = _normalize(text)
    if not words:
        return False
    split = _split_phrases(tuple(phrases))
    i = 0
    while i < len(words):
        for p in split:
            if words[i : i + len(p)] == p:
                i += len(p)
                break
        else:
            if words[i] in _FILLERS:
                # fillers are skipped inline, AFTER phrase matching gets first
                # crack at the position — pre-stripping them would break
                # "uh huh" (the "uh" vanishes, and a bare "huh" is a real
                # "what?"-style barge-in, deliberately not a default phrase)
                i += 1
                continue
            if partial:
                tail = words[i:]
                if any(p[: len(tail)] == tail for p in split if len(p) > len(tail)):
                    return True
            return False
    return True
