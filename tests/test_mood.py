from __future__ import annotations

import json

import pytest

from livekit.agents.tts._mood import DEFAULT_MOOD, MOOD_KEYWORDS, MOOD_PRIORITY, match_mood
from livekit.agents.tts._provider_format import expression_attribute, split_all_markup

pytestmark = pytest.mark.unit


def test_every_mood_is_prioritized() -> None:
    assert set(MOOD_PRIORITY) == set(MOOD_KEYWORDS)
    assert len(MOOD_PRIORITY) == len(set(MOOD_PRIORITY))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("excited", "excited"),
        ("soft, with genuine care", "empathetic"),
        ("bright, upbeat energy", "excited"),
        # an explicit naming (weight 2) beats a supporting descriptor (weight 1)
        ("gently curious, welcoming", "curious"),
        ("furious", "angry"),
        ("Excited!!!", "excited"),
    ],
)
def test_matches_known_labels(label: str, expected: str) -> None:
    assert match_mood(label) == expected


@pytest.mark.parametrize("label", ["like a pirate", "across the room", "", "   "])
def test_unmatched_falls_back(label: str) -> None:
    assert match_mood(label) == DEFAULT_MOOD
    assert match_mood(label, fallback=None) is None


def test_keywords_match_at_word_start_only() -> None:
    # "irate" is the stem that made "pirate" read as angry before word-start matching
    assert match_mood("irate", fallback=None) == "angry"
    assert match_mood("pirate", fallback=None) is None
    assert match_mood("excited", fallback=None) == "excited"
    assert match_mood("unexcited", fallback=None) is None


@pytest.mark.parametrize(
    ("label", "mood"),
    [("soft, with genuine care", "empathetic"), ("like a pirate", DEFAULT_MOOD)],
)
def test_expression_attribute_carries_the_normalized_mood(label: str, mood: str) -> None:
    _, tags = split_all_markup(f'<expr type="expression" label="{label}"/>hey')
    attr = expression_attribute(tags)
    assert attr is not None

    # the provider's own words survive alongside the normalized mood, even unrecognized ones
    assert json.loads(attr["lk.expression"]) == {"expression": label, "mood": mood}
