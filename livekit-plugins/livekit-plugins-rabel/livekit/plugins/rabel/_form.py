# Copyright 2026 Rabel Labs
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

"""Bell's guess written the way the speech-to-text writes its final transcript.

Why: under LiveKit's default turn detection (its turn detector, with preemptive generation on), when
the final transcript arrives after the turn has ended LiveKit restarts the early reply unless the
final's string is exactly the early start's string (livekit-agents 1.8.13,
voice/audio_recognition.py: `transcript_changed = self._audio_transcript !=
self._audio_preflight_transcript`). Bell's guesses come in lower case with no punctuation, while
speech-to-texts send cased, punctuated finals, so a right guess was never kept there. (With the
speech-to-text's own end of turn, LiveKit compares words, `_transcripts_equivalent`, and either form
is kept.)

So the guess is written from the speech-to-text's own text of the words so far, exactly as it last
showed them, and only the words Bell adds are formed here: "i" and the common contractions restored,
"AM"/"PM" after an hour, the first word capitalised when it opens a sentence; the turn closed with
"?" when its last sentence opens like a question and "." otherwise. A guess written this way is kept
when the final has that form; when the speech-to-text writes the end of the turn otherwise (a comma,
a name not yet heard, a number), it is not, and LiveKit starts on the final as it does without Bell.

Those rules are English. For any other language the guess is written from what the call itself has
shown (`CallForm`, `written`): each added word as the speech-to-text last wrote it in this call, and
the turn closed the way this call's final transcripts have closed after the same opening word, or
the way most of them have closed. Nothing about the language is assumed. Until the call has had a
final transcript there is nothing to copy, and the guess is written as above, except that added
words of a script written without spaces (kana, ideographs, Thai), after shown text that ends in such
a script or after none, are joined without them and the turn is not closed with ".": after kana or
an ideograph with "\u3002", else with no mark. Such words in a sentence of a spaced script (a name in
English) are written by the English rules, as before.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from livekit.agents.tokenize.basic import split_words

# the words Bell's guesses carry without their apostrophe (LiveKit's comparison drops it), restored;
# a word that is also an English word without one ("ill", "were", "well", "its", "hell", "shell",
# "wed", "id") is left out unless measured
_CONTRACTIONS = {
    w.replace("'", ""): w
    for w in (
        "i'm i'd i've i'll don't doesn't didn't can't couldn't won't wouldn't shouldn't isn't "
        "aren't wasn't weren't haven't hasn't hadn't that's what's there's here's where's who's "
        "how's let's you're you've you'll you'd they're they've they'll we're we've it's he's she's"
    ).split()
}
_QUESTION_OPENERS = frozenset(
    "can could would will do does did is are am was were should shall may might have has what "
    "whats how where when why who which whose whom isnt arent dont doesnt didnt wont wouldnt "
    "couldnt cant".split()
)
_ENDS = (".", "?", "!")
_HOURS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve thirty fifteen "
    "fortyfive".split()
)
_PUNCT = re.compile(r"[^\w]", re.UNICODE)
# the marks a sentence can end on, in any script: a class of punctuation, not a language's rules
_ANY_ENDS = (
    ".",
    "?",
    "!",
    "\u0964",
    "\u0965",
    "\u061f",
    "\u06d4",
    "\u3002",
    "\uff1f",
    "\uff01",
    "\u2026",
)


def norm_word(token: str) -> str:
    """A shown token in the form Bell's guesses and LiveKit's comparison use.

    Punctuation dropped, case-folded.
    """
    return _PUNCT.sub("", token).casefold()


def _form(word: str) -> str:
    if word == "i":
        return "I"
    if word in _CONTRACTIONS:
        f = _CONTRACTIONS[word]
        return "I" + f[1:] if f.startswith("i'") else f
    return word


def _cap(w: str) -> str:
    return w[:1].upper() + w[1:]


def _sentence_opener(toks: list[str]) -> str:
    start = 0
    for i, t in enumerate(toks[:-1]):
        if t.endswith(_ENDS):
            start = i + 1
    return norm_word(toks[start]) if toks[start:] else ""


def final_form(shown: str, added: tuple[str, ...]) -> str:
    """The guess in the final's form: `shown` (the speech-to-text's text of the words so far, as it
    showed them) and the words Bell adds after them. With no shown text, the added words are the
    whole turn."""
    shown = shown.strip()
    if not added:
        return shown
    if unspaced(shown, added):
        # a script written without spaces (kana, ideographs, Thai): no English rule applies, the
        # added words are joined without spaces, and the turn closed with the ideographic full stop
        # after kana or an ideograph, else with no mark (Thai writes none)
        toks = (shown.split() if shown else []) + list(added)
        return _join(toks, len(added)) + ("\u3002" if _ideographic(added[-1][-1:]) else "")
    words = [_form(w) for w in added]
    before = [norm_word(t) for t in shown.split()[-1:]] + list(added)
    for i, w in enumerate(added):  # "five pm" -> "five PM": the hour's mark, after a number
        prev = before[i]
        if w in ("am", "pm") and (prev in _HOURS or prev.isdigit()):
            words[i] = w.upper()
    if not shown or shown.endswith(_ENDS):
        words[0] = _cap(words[0])
    toks = (shown.split() if shown else []) + words
    return " ".join(toks) + ("?" if _sentence_opener(toks) in _QUESTION_OPENERS else ".")


# -------------------------------------------------- any other language: the call's own form


def _words(text: str) -> list[str]:
    """The words LiveKit compares: split on whitespace and per character for scripts without spaces,
    punctuation dropped, case-folded."""
    return [
        w.casefold()
        for w, _, _ in split_words(text, ignore_punctuation=True, split_character=True)
        if w
    ]


def _is_mark(c: str) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


def _opener(toks: list[str]) -> str:
    """The first word of the last sentence in `toks`, as LiveKit compares words.

    "" when it has none.
    """
    start = 0
    for i, t in enumerate(toks[:-1]):
        if t.endswith(_ANY_ENDS):
            start = i + 1
    for t in toks[start:]:
        w = _words(t)
        if w:
            return w[0]
    return ""


class CallForm:
    """How this call's speech-to-text has written its final transcripts: each word's written form,
    whether a sentence opens with a capital, how a turn closes (its last marks, by the sentence's
    opening word and over all), and whether a space stands where a script written without spaces
    meets a spaced one."""

    def __init__(self) -> None:
        self._surface: dict[str, str] = {}
        self._ends: Counter[str] = Counter()
        self._ends_after: dict[str, Counter[str]] = {}
        # final transcripts that opened with a capital, less those that opened without
        self._capital = 0
        # meetings of a script without spaces and a spaced one written with a space, less those
        # written without
        self._script_gap = 0

    def final(self, text: str) -> None:
        """A final transcript, as the speech-to-text wrote it."""
        text = text.strip()
        n = len(text)
        while n and _is_mark(text[n - 1]):
            n -= 1
        if not n:
            return
        end, toks = text[n:], text[:n].split()
        self._ends[end] += 1
        self._ends_after.setdefault(_opener(toks), Counter())[end] += 1
        first = next((c for c in text if c.isalpha()), "")
        if first and first.upper() != first.lower():
            self._capital += 1 if first == first.upper() else -1
        opens = True
        for tok in toks:
            w = _words(tok)
            if len(w) == 1 and not opens:
                a, b = 0, len(tok)
                while a < b and _is_mark(tok[a]):
                    a += 1
                while b > a and _is_mark(tok[b - 1]):
                    b -= 1
                self._surface[w[0]] = tok[a:b]
            opens = tok.endswith(_ANY_ENDS)
        prev, gap = "", False
        for c in text:
            if c.isspace():
                gap = True
                continue
            if not c.isalnum():  # a mark between them: no meeting of the two scripts
                prev, gap = "", False
                continue
            if prev and _unspaced(prev) != _unspaced(c):
                self._script_gap += 1 if gap else -1
            prev, gap = c, False

    def end_after(self, opener: str) -> str | None:
        """How a turn has closed after this opening word, else how most turns have closed.

        None before any final.
        """
        seen = self._ends_after.get(opener) or self._ends
        return max(seen, key=lambda e: seen[e]) if seen else None

    def word(self, w: str) -> str:
        return self._surface.get(w, w)

    @property
    def capitals(self) -> bool:
        return self._capital >= 0

    @property
    def spaces_scripts(self) -> bool:
        """A space where a script without spaces meets a spaced one, as most of this call's finals
        have written it; none before any final has shown one."""
        return self._script_gap > 0


def _unspaced(c: str) -> bool:
    """A character of a script written without spaces between words, which LiveKit's comparison
    takes one character at a time (split_words with split_character: kana, ideographs, Thai)."""
    return not c.isspace() and len(split_words(c + c, split_character=True)) == 2


def unspaced(shown: str, added: tuple[str, ...]) -> bool:
    """Bell's added words are all of a script written without spaces (an English word never is),
    and so is the end of the shown text they join, if there is one. A name of such a script in a
    sentence of a spaced one ("I'd like to speak to" and "王", "please") is not."""
    end = next((c for c in reversed(shown) if not _is_mark(c)), "")
    return (
        bool(added) and all(_unspaced(c) for w in added for c in w) and (not end or _unspaced(end))
    )


def _ideographic(c: str) -> bool:
    """Kana or an ideograph: Japanese and Chinese close a sentence with the ideographic full
    stop."""
    return unicodedata.name(c, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH", "HIRAGANA", "KATAKANA")
    )


def _join(toks: list[str], n_added: int, spaces_scripts: bool = False) -> str:
    """The tokens joined as the speech-to-text writes them: the shown ones with a space between,
    as before, and each of the last `n_added` (Bell's added words) after a space between two words
    of spaced scripts, none between two of a script written without spaces, and where the two
    meet as `spaces_scripts` says."""
    shown = " ".join(toks[: len(toks) - n_added])
    out = shown
    for w in toks[len(toks) - n_added :]:
        if out:
            before, after = _unspaced(out[-1]), _unspaced(w[:1])
            if spaces_scripts if before != after else not before:
                out += " "
        out += w
    return out


def written(shown: str, added: tuple[str, ...], form: CallForm) -> str:
    """The guess in the final's form for a language other than English: `shown` (the
    speech-to-text's text of the words so far) and the words Bell adds after them, written as this
    call's final transcripts have been written."""
    shown = shown.strip()
    if not added:
        return shown
    words = [form.word(w) for w in added]
    toks = (shown.split() if shown else []) + words
    end = form.end_after(_opener(toks))
    if end is None:
        return final_form(shown, added)  # no final transcript in this call yet: nothing to copy
    if form.capitals and (not shown or shown.endswith(_ANY_ENDS)):
        toks[len(toks) - len(words)] = _cap(words[0])
    return _join(toks, len(words), form.spaces_scripts) + end
