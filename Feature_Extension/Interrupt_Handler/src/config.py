from dataclasses import dataclass, field
from typing import List, Set, Dict
import os
import re

def _split_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [w.strip().lower() for w in re.split(r"[,\n;]+", raw) if w.strip()]

@dataclass
class IHConfig:
    fillers: Dict[str, Set[str]] = field(default_factory=dict)
    hard_phrases: Dict[str, List[str]] = field(default_factory=dict)
    active_langs: List[str] = field(default_factory=lambda: ["en"])
    min_confidence: float = 0.5
    min_content_tokens: int = 2
    min_duration_ms: int = 250
    debounce_ms: int = 200

    @classmethod
    def default_lang_packs(cls) -> Dict[str, Dict[str, List[str]]]:
        """Predefined English + Hindi/Hinglish lexicons."""
        return {
            "en": {
                "fillers": ["uh","umm","um","hmm","huh","er","eh","mmm","ah","oh"],
                "commands": ["stop","wait","pause","hold on","cancel","no","not that one"],
            },
            "hi": {
                "fillers": ["haan","arey","acha","hmmm"],
                "commands": ["ruk","ruko","thoda ruk","thodi der ruk","ek second","bas","sun","sunna"],
            },
        }

    @classmethod
    def from_env(cls) -> "IHConfig":
        packs = cls.default_lang_packs()
        langs = _split_env("IH_LANGS", "en,hi")
        fillers: Dict[str, Set[str]] = {}
        hard: Dict[str, List[str]] = {}

        for lang in langs:
            if lang in packs:
                fillers[lang] = set(packs[lang]["fillers"])
                hard[lang] = list(packs[lang]["commands"])

        # Optional overrides from env (comma separated)
        custom_fills = _split_env("IH_FILLERS", "")
        custom_cmds = _split_env("IH_HARD_PHRASES", "")
        if custom_fills:
            fillers["custom"] = set(custom_fills)
        if custom_cmds:
            hard["custom"] = custom_cmds

        return cls(
            fillers=fillers,
            hard_phrases=hard,
            active_langs=langs,
            min_confidence=float(os.getenv("IH_MIN_CONFIDENCE", "0.5")),
            min_content_tokens=int(os.getenv("IH_MIN_CONTENT_TOKENS", "2")),
            min_duration_ms=int(os.getenv("IH_MIN_DURATION_MS", "250")),
            debounce_ms=int(os.getenv("IH_DEBOUNCE_MS", "200")),
        )

    # === Bonus 1: runtime updates ===
    def add_fillers(self, words: List[str], lang: str = "custom"):
        self.fillers.setdefault(lang, set()).update(w.strip().lower() for w in words if w.strip())

    def remove_fillers(self, words: List[str], lang: str = "custom"):
        for w in words:
            self.fillers.get(lang, set()).discard(w.strip().lower())

    def add_commands(self, phrases: List[str], lang: str = "custom"):
        self.hard_phrases.setdefault(lang, [])
        for p in phrases:
            p = p.strip().lower()
            if p and p not in self.hard_phrases[lang]:
                self.hard_phrases[lang].append(p)

    def remove_commands(self, phrases: List[str], lang: str = "custom"):
        if lang not in self.hard_phrases:
            return
        bad = {p.strip().lower() for p in phrases}
        self.hard_phrases[lang] = [p for p in self.hard_phrases[lang] if p not in bad]

    def all_fillers(self) -> Set[str]:
        """Union of fillers across active languages."""
        return set().union(*(self.fillers.get(l, set()) for l in self.active_langs))

    def all_commands(self) -> List[str]:
        out = []
        for l in self.active_langs:
            out += self.hard_phrases.get(l, [])
        return out
