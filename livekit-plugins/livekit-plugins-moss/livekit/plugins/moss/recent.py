# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Sequence

from moss import QueryResultDocumentInfo

ENGLISH_FILLERS = frozenset(
    {
        "ok",
        "okay",
        "alright",
        "all right",
        "yeah",
        "yes",
        "yep",
        "no",
        "nope",
        "mhm",
        "uh huh",
        "uh",
        "um",
        "hmm",
        "right",
        "sure",
        "got it",
        "i see",
        "thanks",
        "thank you",
    }
)
"""Utterances that skip the search by default, such as "okay" or "got it"."""

_SHORT_TURN_WORDS = 8
_CONTEXT_CHARS = 400  # keeps the latest turn inside the embedding model's input


def split_words(text: str) -> list[str]:
    """Split text into lowercase words in any script, dropping punctuation and symbols."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in folded).split()


class Fillers:
    """Utterances with no topic to search for, ignoring case and punctuation."""

    def __init__(self, phrases: Iterable[str]) -> None:
        self._phrases = {tuple(split_words(phrase)) for phrase in phrases} - {()}
        self._longest = max(map(len, self._phrases), default=0)

    def match(self, text: str) -> bool:
        """Whether the text holds only fillers, such as "okay thanks", or no words at all."""
        words = split_words(text)
        # split[end] is whether the first `end` words split into whole fillers.
        split = [True] + [False] * len(words)
        for end in range(1, len(words) + 1):
            split[end] = any(
                split[start] and tuple(words[start:end]) in self._phrases
                for start in range(max(0, end - self._longest), end)
            )
        return split[-1]


def build_queries(texts: Sequence[str], fillers: Fillers) -> list[str]:
    """Queries for the last of the user texts, none for a filler.

    A short turn such as "and the price?" also gets a query
    with the two user turns before it prepended, to keep their topic.
    """
    latest = texts[-1]
    if fillers.match(latest):
        return []
    queries = [latest]
    if len(split_words(latest)) <= _SHORT_TURN_WORDS:
        earlier = [text for text in texts[:-1] if not fillers.match(text)][-2:]
        if context := " ".join(earlier)[-_CONTEXT_CHARS:]:
            queries.append(f"{context} {latest}")
    return queries


class RecentPassages:
    """Passages found in the last few searched user turns, with older turns weighted down."""

    def __init__(self, *, turns: int = 3, max_chars: int = 3200, decay: float = 0.8) -> None:
        self._turns: OrderedDict[str, list[QueryResultDocumentInfo]] = OrderedDict()
        self._max_turns = turns
        self._max_chars = max_chars
        self._decay = decay

    def keep(self, turn: str, docs: Sequence[QueryResultDocumentInfo]) -> None:
        """Store the passages of a turn, replacing those of an earlier draft of it."""
        self._turns.pop(turn, None)
        self._turns[turn] = list(docs)
        while len(self._turns) > self._max_turns:
            self._turns.popitem(last=False)

    def forget(self, turn: str) -> None:
        self._turns.pop(turn, None)

    def clear(self) -> None:
        self._turns.clear()

    def render(self) -> list[str]:
        """Passage texts, each once and best first, within the size cap.

        A passage is ranked by its score, multiplied by the decay once for
        each searched turn since it was found. The best passage always fits.
        """
        best: dict[tuple[str | None, str], tuple[float, str]] = {}
        for age, docs in enumerate(reversed(self._turns.values())):
            for doc in docs:
                weight = doc.score * self._decay**age
                key = (doc.index_name, doc.id)
                if key not in best or weight > best[key][0]:
                    best[key] = (weight, doc.text)
        kept: list[str] = []
        used = 0
        for _, text in sorted(best.values(), key=lambda ranked: ranked[0], reverse=True):
            if kept and used + len(text) > self._max_chars:
                continue
            kept.append(text)
            used += len(text)
        return kept
