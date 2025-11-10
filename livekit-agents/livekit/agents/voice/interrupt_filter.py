from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass


def _csv_env(name: str, default: Iterable[str]) -> set[str]:
    val = os.getenv(name)
    if not val:
        return {w.strip().lower() for w in default if w.strip()}
    return {w.strip().lower() for w in val.split(",") if w.strip()}


def _float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
class InterruptionDecision:
    kind: str  # "ignore_filler" | "real_interrupt" | "passive"
    reason: str


class InterruptionClassifier:
    """
    Classifies user transcripts during agent speech to decide interruption behavior.

    Configuration via env vars:
    - AGENTS_IGNORED_FILLERS: CSV of filler tokens to ignore when agent is speaking.
    - AGENTS_STOP_KEYWORDS: CSV of keywords that always trigger interruption (e.g., stop, wait).
    - AGENTS_MIN_CONFIDENCE: Float [0,1] below which short murmurs are ignored as noise when agent speaking.
    """

    def __init__(
        self,
        *,
        fillers: Iterable[str] | None = None,
        stop_keywords: Iterable[str] | None = None,
        min_confidence: float | None = None,
        fillers_by_lang: dict[str, set[str]] | None = None,
        stop_by_lang: dict[str, set[str]] | None = None,
    ) -> None:
        # default/global sets
        self._fillers_default: set[str] = {f.lower() for f in (fillers or [])}
        self._stop_default: set[str] = {s.lower() for s in (stop_keywords or [])}
        # language-specific overrides
        self._fillers_by_lang: dict[str, set[str]] = fillers_by_lang or {}
        self._stop_by_lang: dict[str, set[str]] = stop_by_lang or {}
        self._min_conf = 0.6 if min_confidence is None else float(min_confidence)

    @classmethod
    def from_env(cls) -> InterruptionClassifier:
        # default/global
        fillers = _csv_env("AGENTS_IGNORED_FILLERS", ["uh", "umm", "um", "hmm", "haan"])
        stops = _csv_env("AGENTS_STOP_KEYWORDS", ["stop", "wait", "hold", "hold on", "pause"])
        min_conf = _float_env("AGENTS_MIN_CONFIDENCE", 0.6)
        # language-specific from env: AGENTS_IGNORED_FILLERS_<lang>, AGENTS_STOP_KEYWORDS_<lang>
        fillers_by_lang: dict[str, set[str]] = {}
        stop_by_lang: dict[str, set[str]] = {}
        for k, v in os.environ.items():
            if k.startswith("AGENTS_IGNORED_FILLERS_") and v:
                lang = k.split("_", maxsplit=3)[-1].lower()
                fillers_by_lang[lang] = {w.strip().lower() for w in v.split(",") if w.strip()}
            elif k.startswith("AGENTS_STOP_KEYWORDS_") and v:
                lang = k.split("_", maxsplit=3)[-1].lower()
                stop_by_lang[lang] = {w.strip().lower() for w in v.split(",") if w.strip()}
        return cls(
            fillers=fillers,
            stop_keywords=stops,
            min_confidence=min_conf,
            fillers_by_lang=fillers_by_lang,
            stop_by_lang=stop_by_lang,
        )

    def update_fillers(self, fillers: Iterable[str], language: str | None = None) -> None:
        if language:
            self._fillers_by_lang[language.lower()] = {f.lower() for f in fillers}
        else:
            self._fillers_default = {f.lower() for f in fillers}

    def update_stop_keywords(
        self, stop_keywords: Iterable[str], language: str | None = None
    ) -> None:
        if language:
            self._stop_by_lang[language.lower()] = {s.lower() for s in stop_keywords}
        else:
            self._stop_default = {s.lower() for s in stop_keywords}

    def update_min_confidence(self, min_confidence: float) -> None:
        self._min_conf = float(min_confidence)

    def classify(
        self,
        *,
        transcript: str,
        confidence: float | None,
        agent_speaking: bool,
        language: str | None = None,
    ) -> InterruptionDecision:
        if not transcript:
            return InterruptionDecision(kind="passive", reason="empty_transcript")

        # Always allow when agent is not speaking
        if not agent_speaking:
            return InterruptionDecision(kind="passive", reason="agent_not_speaking")

        text = transcript.strip().lower()
        tokens = _tokenize(text)

        # Select language-specific sets
        lang = language.lower() if language else None
        fillers = self._fillers_by_lang.get(lang, self._fillers_default)
        stops = self._stop_by_lang.get(lang, self._stop_default)

        # Command override
        if any(cmd in text for cmd in stops):
            return InterruptionDecision(kind="real_interrupt", reason="stop_keyword_match")

        # Filler-only logic while agent is speaking
        if tokens and all(t in fillers for t in tokens):
            # Low-confidence murmurs get ignored; higher confidence fillers also ignored to avoid false cuts
            conf = confidence if confidence is not None else 0.0
            if conf < self._min_conf or True:
                return InterruptionDecision(kind="ignore_filler", reason="filler_only")

        # Very short, low-confidence murmurs like "hmm yeah" can be treated as filler if mostly fillers
        conf = confidence if confidence is not None else 0.0
        if len(tokens) <= 3 and conf < self._min_conf:
            filler_like = sum(1 for t in tokens if t in fillers)
            if filler_like >= max(1, len(tokens) - 1):
                return InterruptionDecision(kind="ignore_filler", reason="low_conf_murmur")

        # Otherwise treat as real interruption
        return InterruptionDecision(kind="real_interrupt", reason="contentful_speech")
