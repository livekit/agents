import re
from typing import List

TOKEN_RE = re.compile(r"\w+['-]?\w*|\w+")

def tokenize(text: str):
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

def is_filler_only(tokens: List[str], ignored_words: List[str], strict=True):
    if not tokens:
        return True
    if strict:
        # all tokens must be exactly in ignored list
        return all(t in ignored_words for t in tokens)
    else:
        # allow matching if token contains an ignored fragment
        return all(any(ig in t for ig in ignored_words) for t in tokens)

def contains_interrupt_command(tokens: List[str], commands: List[str]):
    return any(t in commands for t in tokens)
