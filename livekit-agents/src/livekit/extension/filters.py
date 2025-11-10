import re
import unicodedata
from typing import Iterable

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize(text: str) -> str:
    # Unicode normalize + lower + strip punctuation (keep spaces)
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = _PUNCT_RE.sub("", t)
    return " ".join(t.split())

def tokenize(text: str) -> list[str]:
    return normalize(text).split()

def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    base = normalize(text)
    for p in phrases:
        p_norm = normalize(p)
        if p_norm and p_norm in base:
            return True
    return False
