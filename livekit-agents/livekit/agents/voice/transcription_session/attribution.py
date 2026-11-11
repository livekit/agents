from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...stt import SpeechData
from ...types import TimedString
from ...utils import is_given
from .activity import ActivityTimeline

AttributionMethod = Literal["single", "word", "utterance", "diarization", "none"]
"""How a segment's speaker was chosen.

- ``single``: only one participant was audible over the whole utterance.
- ``word``: each word went to the participant loudest over its own timing.
- ``utterance``: no word timings; the participant loudest over the utterance.
- ``diarization``: energy was ambiguous; the STT's speaker label decided.
- ``none``: nobody was audible, the segment has no participant.
"""

# how far to look either side of a span that caught nobody: STT timings drift by a word or so
_SEARCH_PAD = 0.25


@dataclass
class _Run:
    identity: str | None
    text: str
    start_time: float
    end_time: float
    words: list[TimedString] | None
    overlap: bool
    method: AttributionMethod


class _SpeakerMap:
    """Learns which participant a diarization label belongs to.

    Only words energy attributed confidently are evidence, so a label is trusted where energy
    is ambiguous without ever overriding a clear energy decision.
    """

    def __init__(self, *, min_evidence: int = 3, min_share: float = 0.7) -> None:
        self._min_evidence = min_evidence
        self._min_share = min_share
        self._counts: dict[str, dict[str, int]] = {}

    def observe(self, speaker_id: str, identity: str) -> None:
        counts = self._counts.setdefault(speaker_id, {})
        counts[identity] = counts.get(identity, 0) + 1

    def resolve(self, speaker_id: str) -> str | None:
        counts = self._counts.get(speaker_id)
        if not counts:
            return None
        total = sum(counts.values())
        identity, best = max(counts.items(), key=lambda item: item[1])
        if total < self._min_evidence or best / total < self._min_share:
            return None
        return identity


class _Attributor:
    def __init__(self, timeline: ActivityTimeline, *, overlap_ratio: float) -> None:
        self._timeline = timeline
        self._overlap_ratio = overlap_ratio
        self._speakers = _SpeakerMap()

    def attribute(self, alt: SpeechData, *, fallback: tuple[float, float]) -> list[_Run]:
        """Split a transcript into runs of one speaker each.

        Args:
            alt: The transcript.
            fallback: The span to assume when the transcript carries no timing at all.
        """
        words = _timed_words(alt.words)
        if words:
            start, end = words[0].start, words[-1].end
        elif alt.end_time > alt.start_time:
            start, end = alt.start_time, alt.end_time
        else:
            start, end = fallback

        identity, overlap, scores = self._choose(start, end)
        if len(scores) == 1:
            if identity is not None:
                # the words' own labels when the STT gives them, else the utterance's
                labels = [w.text.speaker_id for w in words or [] if w.text.speaker_id is not None]
                if not labels and alt.speaker_id is not None:
                    labels = [alt.speaker_id]
                for speaker_id in labels:
                    self._speakers.observe(speaker_id, identity)
            return [_Run(identity, alt.text, start, end, alt.words, False, "single")]

        if words:
            return self._by_word(alt.text, words)

        method: AttributionMethod = "utterance" if identity else "none"
        if alt.speaker_id is not None:
            if (identity is None or overlap) and (mapped := self._speakers.resolve(alt.speaker_id)):
                identity, method = mapped, "diarization"
            elif identity is not None and not overlap:
                self._speakers.observe(alt.speaker_id, identity)
        return [_Run(identity, alt.text, start, end, alt.words, overlap, method)]

    def _choose(self, start: float, end: float) -> tuple[str | None, bool, dict[str, float]]:
        scores = self._timeline.scores(start, end)
        if not scores:
            scores = self._timeline.scores(start - _SEARCH_PAD, end + _SEARCH_PAD)
        if not scores:
            return None, False, scores

        ranked = sorted(scores.values(), reverse=True)
        overlap = len(ranked) > 1 and ranked[1] >= ranked[0] * self._overlap_ratio
        return max(scores, key=lambda identity: scores[identity]), overlap, scores

    def _by_word(self, text: str, words: list[_TimedWord]) -> list[_Run]:
        assigned: list[tuple[str | None, bool, AttributionMethod]] = []
        for word in words:
            identity, overlap, _ = self._choose(word.start, word.end)
            method: AttributionMethod = "word"
            if (speaker_id := word.text.speaker_id) is not None:
                if (identity is None or overlap) and (mapped := self._speakers.resolve(speaker_id)):
                    identity, method = mapped, "diarization"
                elif identity is not None and not overlap:
                    self._speakers.observe(speaker_id, identity)
            assigned.append((identity, overlap, method))

        # a word nobody was audible for is timing drift, not a new speaker: it joins a neighbour
        identities = [identity for identity, _, _ in assigned]
        for i in range(len(identities)):
            if identities[i] is None and i > 0:
                identities[i] = identities[i - 1]
        for i in reversed(range(len(identities))):
            if identities[i] is None and i + 1 < len(identities):
                identities[i] = identities[i + 1]

        starts = [0] + [i for i in range(1, len(words)) if identities[i] != identities[i - 1]]
        texts = _split_text(text, [word.text for word in words], starts)
        runs: list[_Run] = []
        for n, first in enumerate(starts):
            last = starts[n + 1] if n + 1 < len(starts) else len(words)
            run_words = words[first:last]
            methods = {assigned[i][2] for i in range(first, last)}
            identity = identities[first]
            runs.append(
                _Run(
                    identity=identity,
                    text=texts[n] if texts else " ".join(word.text for word in run_words),
                    start_time=run_words[0].start,
                    end_time=run_words[-1].end,
                    words=[word.text for word in run_words],
                    overlap=any(assigned[i][1] for i in range(first, last)),
                    method="none"
                    if identity is None
                    else ("diarization" if "diarization" in methods else "word"),
                )
            )
        return runs


@dataclass
class _TimedWord:
    text: TimedString
    start: float
    end: float


def _timed_words(words: list[TimedString] | None) -> list[_TimedWord] | None:
    """The words with their timings, when every word has them."""
    timed: list[_TimedWord] = []
    for word in words or []:
        if not is_given(word.start_time) or not is_given(word.end_time):
            return None
        timed.append(_TimedWord(word, word.start_time, word.end_time))
    if not timed or timed[-1].end <= 0:
        return None  # no words, or a provider that zero-fills timings it doesn't have
    return timed


def _split_text(text: str, words: list[TimedString], starts: list[int]) -> list[str] | None:
    """Cut the transcript at run boundaries, keeping its own spacing and punctuation.

    None when a word can't be found in the text (a provider whose words are normalized
    differently from its transcript); the caller joins the words instead.
    """
    lowered = text.lower()
    positions: list[int] = []
    cursor = 0
    for word in words:
        position = lowered.find(str(word).lower(), cursor)
        if position < 0:
            return None
        positions.append(position)
        cursor = position + len(word)

    cuts = [0] + [positions[i] for i in starts[1:]] + [len(text)]
    return [text[cuts[i] : cuts[i + 1]].strip() for i in range(len(starts))]
