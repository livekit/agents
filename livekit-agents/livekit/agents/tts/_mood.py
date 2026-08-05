"""Normalize a free-form delivery label into a small, fixed set of moods.

Expressive Mode asks the LLM for a delivery label, and the label space is open-ended: Fish
Audio emits single words from a closed set, Inworld emits free-form English ("soft, with
genuine care"), and models routinely drift outside whichever set they were given. Matching
here is best-effort, so frontends can drive UI off a fixed enum instead of each client SDK
carrying its own copy of the keyword table.
"""

from __future__ import annotations

from typing import Literal

from ._mood_data import MOOD_KEYWORDS

AgentMood = Literal[
    "excited",
    "happy",
    "playful",
    "curious",
    "surprised",
    "hopeful",
    "empathetic",
    "sad",
    "angry",
    "anxious",
    "calm",
]

# Tie-break order, most specific first. Two moods scoring equally on a compound label
# resolve to whichever appears earlier here.
MOOD_PRIORITY: list[AgentMood] = [
    "angry",
    "sad",
    "anxious",
    "surprised",
    "playful",
    "empathetic",
    "excited",
    "curious",
    "hopeful",
    "happy",
    "calm",
]

# The mood an unrecognized label falls back to. `calm` is the most recessive mood, so an
# unmatched label reads as "no strong signal" rather than asserting a feeling the agent
# never expressed.
DEFAULT_MOOD: AgentMood = "calm"


def _matches_word(text: str, keyword: str) -> bool:
    # keywords match at a word start, so the stem "cross" hits "cross" and "crossly" but not
    # "across". Matching mid-word read "like a pirate" as `angry`, via "irate".
    start = 0
    while (at := text.find(keyword, start)) != -1:
        if at == 0 or not text[at - 1].isalpha():
            return True
        start = at + 1
    return False


def match_mood(label: str, *, fallback: AgentMood | None = DEFAULT_MOOD) -> AgentMood | None:
    """Match a raw delivery label to a mood.

    Matching is keyword-based and deliberately lossy: the label space is open-ended, so an
    unrecognized label resolves to ``fallback`` rather than a wrong guess.

    Args:
        label: The raw delivery label, as the provider wrote it.
        fallback: Mood to return when nothing matches. Pass ``None`` to get ``None`` back and
            handle the miss yourself.

    Example:
        >>> match_mood("soft, with genuine care")
        'empathetic'
        >>> match_mood("bright, upbeat energy")
        'excited'
        >>> match_mood("like a pirate")
        'calm'
    """
    text = label.lower()
    best: AgentMood | None = None
    best_score = 0

    for mood in MOOD_PRIORITY:
        keywords = MOOD_KEYWORDS[mood]
        score = sum(weight for kw, weight in keywords.items() if _matches_word(text, kw))
        # strictly greater, so MOOD_PRIORITY breaks ties
        if score > best_score:
            best = mood
            best_score = score

    return best if best is not None else fallback
