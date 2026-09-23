# Copyright 2025 LiveKit, Inc.
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

"""Sentence splitting for TTS that works in every script without a language setting.

The plugin sends one text frame per sentence, so the splitter decides when the
first frame leaves and where the provider hears a pause. livekit's default
sentence tokenizer (blingfire) is good at Latin-script disambiguation (``Dr.``,
``3.14``, ``example.com``) but does not recognise the sentence terminators of
several scripts (the Devanagari danda, the Myanmar section mark, the Khmer khan,
the Armenian full stop), and its 20-character minimum re-merges the short
sentences of Chinese and Japanese. A reply in those languages then leaves as a
single frame once the LLM has finished, so first audio waits for the whole reply.

This tokenizer keeps blingfire for what it is good at and adds two
script-agnostic rules on top:

1. Cut after any character Unicode classifies as ``Sentence_Terminal``, whatever
   the script, keeping closing quotes and brackets attached to the sentence.
2. Cut at the last space once a span exceeds ``max_chars``, so text with no
   terminator at all (Thai and Lao have none) still streams.

No minimum sentence length: a sentence boundary is a natural pause, and a
character-count minimum is exactly what starves scripts with short sentences.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from livekit import blingfire
from livekit.agents import tokenize

# Imported by name because the ``tokenize`` method below shadows the module
# inside the class body, where this annotation is resolved.
from livekit.agents.tokenize import SentenceStream, TokenData
from livekit.agents.utils import shortuuid

from .log import logger

# Every Sentence_Terminal code point above U+007F, taken from Unicode's PropList
# (https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt, PropList-18.0.0.txt
# dated 2026-08-07): 169 code points in 86 ranges. The table is vendored because
# ``unicodedata`` exposes no Sentence_Terminal property and ``\p{STerm}`` needs
# the third-party ``regex`` module, which livekit-agents does not depend on. To
# regenerate, parse the "; Sentence_Terminal" lines of that file, expand the
# ranges and drop everything below U+0080.
#
# ASCII ``. ! ?`` are left to blingfire, which knows when a full stop is not a
# sentence end. Nothing does that for the rest of the table, so a fullwidth or
# one-dot-leader full stop (U+FF0E, U+2024, U+FE52) ends a sentence wherever it
# appears, including inside a fullwidth number or abbreviation. Cutting a
# sentence in two costs a pause; not cutting costs the whole reply's streaming.
_STERM_RANGES: tuple[tuple[int, int], ...] = (
    (0x0589, 0x0589),
    (0x061D, 0x061F),
    (0x06D4, 0x06D4),
    (0x0700, 0x0702),
    (0x07F9, 0x07F9),
    (0x0837, 0x0837),
    (0x0839, 0x0839),
    (0x083D, 0x083E),
    (0x0964, 0x0965),
    (0x104A, 0x104B),
    (0x1362, 0x1362),
    (0x1367, 0x1368),
    (0x166E, 0x166E),
    (0x1735, 0x1736),
    (0x17D4, 0x17D5),
    (0x1803, 0x1803),
    (0x1809, 0x1809),
    (0x1944, 0x1945),
    (0x1AA8, 0x1AAB),
    (0x1B4E, 0x1B4F),
    (0x1B5A, 0x1B5B),
    (0x1B5E, 0x1B5F),
    (0x1B7D, 0x1B7F),
    (0x1C3B, 0x1C3C),
    (0x1C7E, 0x1C7F),
    (0x2024, 0x2024),
    (0x203C, 0x203D),
    (0x2047, 0x2049),
    (0x2CF9, 0x2CFB),
    (0x2E2E, 0x2E2E),
    (0x2E3C, 0x2E3C),
    (0x2E53, 0x2E54),
    (0x2E60, 0x2E61),
    (0x3002, 0x3002),
    (0xA4FF, 0xA4FF),
    (0xA60E, 0xA60F),
    (0xA6F3, 0xA6F3),
    (0xA6F7, 0xA6F7),
    (0xA876, 0xA877),
    (0xA8CE, 0xA8CF),
    (0xA92F, 0xA92F),
    (0xA9C8, 0xA9C9),
    (0xAA5D, 0xAA5F),
    (0xAAF0, 0xAAF1),
    (0xABEB, 0xABEB),
    (0xFE12, 0xFE12),
    (0xFE15, 0xFE16),
    (0xFE52, 0xFE52),
    (0xFE56, 0xFE57),
    (0xFF01, 0xFF01),
    (0xFF0E, 0xFF0E),
    (0xFF1F, 0xFF1F),
    (0xFF61, 0xFF61),
    (0x10A56, 0x10A57),
    (0x10F55, 0x10F59),
    (0x10F86, 0x10F89),
    (0x11047, 0x11048),
    (0x110BE, 0x110C1),
    (0x11141, 0x11143),
    (0x111C5, 0x111C6),
    (0x111CD, 0x111CD),
    (0x111DE, 0x111DF),
    (0x11238, 0x11239),
    (0x1123B, 0x1123C),
    (0x112A9, 0x112A9),
    (0x113D4, 0x113D5),
    (0x1144B, 0x1144C),
    (0x115C2, 0x115C3),
    (0x115C9, 0x115D7),
    (0x11641, 0x11642),
    (0x1173C, 0x1173E),
    (0x11944, 0x11944),
    (0x11946, 0x11946),
    (0x11A42, 0x11A43),
    (0x11A9B, 0x11A9C),
    (0x11C41, 0x11C42),
    (0x11EF7, 0x11EF8),
    (0x11F43, 0x11F44),
    (0x16A6E, 0x16A6F),
    (0x16AF5, 0x16AF5),
    (0x16B37, 0x16B38),
    (0x16B44, 0x16B44),
    (0x16D6E, 0x16D6F),
    (0x16E98, 0x16E98),
    (0x1BC9F, 0x1BC9F),
    (0x1DA88, 0x1DA88),
)
_STERM = frozenset(chr(cp) for lo, hi in _STERM_RANGES for cp in range(lo, hi + 1))
_ASCII_TERMINATORS = frozenset(".!?")
# Closing quotes and brackets stay attached to the sentence they close.
_CLOSERS = frozenset("\"'”’»›」』）)]}〉》】〕〗〙〛")
_INNER_NEWLINES = re.compile(r"\s*\n+\s*")
# Zero-width non-joiner and joiner. Both are format controls that govern how
# their two neighbours are shaped (the joiner asks for a ligature, the
# non-joiner forbids one), so a cut on either side of one changes rendering.
_JOINERS = frozenset("\u200c\u200d")
# Canonical combining class of a virama. Most viramas ask for a conjunct with
# the consonant that follows, so a chunk must not end on one.
_VIRAMA_CLASS = 9
# Thai and Lao spacing vowels. Category Lo rather than Mn, so the general
# category does not catch them, but they are written after their consonant and
# cannot begin a cluster.
_SPACING_VOWELS = frozenset("\u0e32\u0e33\u0eb2\u0eb3")
# Unicode's whole Logical_Order_Exception set, all Thai and Lao: vowels stored
# before the consonant they are pronounced after, so a chunk must not end on one.
_LEADING_VOWELS = frozenset(chr(cp) for cp in (*range(0x0E40, 0x0E45), *range(0x0EC0, 0x0EC5)))
# Emoji skin-tone modifiers (category Sk) attach to the emoji before them.
_EMOJI_MODIFIERS = frozenset(chr(cp) for cp in range(0x1F3FB, 0x1F400))
# A flag is a pair of regional indicators; cutting between them shows letters.
_REGIONAL_INDICATORS = frozenset(chr(cp) for cp in range(0x1F1E6, 0x1F200))

# Every clause mark above U+007F: what Unicode's Terminal_Punctuation adds to
# Sentence_Terminal, which is the comma, semicolon and colon of each script (the
# ideographic and fullwidth comma, the Arabic comma and semicolon, the Ethiopic
# comma and colons, and so on). Taken from the same PropList-18.0.0.txt as
# _STERM_RANGES: 118 code points in 59 ranges. To regenerate, parse the
# "; Terminal_Punctuation" lines of that file, expand the ranges, remove every
# Sentence_Terminal code point and drop everything below U+0080. The property
# also holds a few word dividers, such as the Ethiopic wordspace, where an
# opening then ends at a word.
_CLAUSE_MARK_RANGES: tuple[tuple[int, int], ...] = (
    (0x037E, 0x037E),
    (0x0387, 0x0387),
    (0x05C3, 0x05C3),
    (0x060C, 0x060C),
    (0x061B, 0x061B),
    (0x0703, 0x070A),
    (0x070C, 0x070C),
    (0x07F8, 0x07F8),
    (0x0830, 0x0835),
    (0x0838, 0x0838),
    (0x083A, 0x083C),
    (0x085E, 0x085E),
    (0x0E5A, 0x0E5B),
    (0x0F08, 0x0F08),
    (0x0F0D, 0x0F12),
    (0x1361, 0x1361),
    (0x1363, 0x1366),
    (0x16EB, 0x16ED),
    (0x17D6, 0x17D6),
    (0x17DA, 0x17DA),
    (0x1802, 0x1802),
    (0x1804, 0x1805),
    (0x1808, 0x1808),
    (0x1B5D, 0x1B5D),
    (0x1C3D, 0x1C3F),
    (0x2E41, 0x2E41),
    (0x2E4C, 0x2E4C),
    (0x2E4E, 0x2E4F),
    (0x3001, 0x3001),
    (0xA4FE, 0xA4FE),
    (0xA60D, 0xA60D),
    (0xA6F4, 0xA6F6),
    (0xA9C7, 0xA9C7),
    (0xAADF, 0xAADF),
    (0xFE50, 0xFE51),
    (0xFE54, 0xFE55),
    (0xFF0C, 0xFF0C),
    (0xFF1A, 0xFF1B),
    (0xFF64, 0xFF64),
    (0x1039F, 0x1039F),
    (0x103D0, 0x103D0),
    (0x10857, 0x10857),
    (0x1091F, 0x1091F),
    (0x10AF0, 0x10AF5),
    (0x10B3A, 0x10B3F),
    (0x10B99, 0x10B9C),
    (0x11049, 0x1104D),
    (0x1123A, 0x1123A),
    (0x1144D, 0x1144D),
    (0x1145A, 0x1145B),
    (0x115C4, 0x115C5),
    (0x11AA1, 0x11AA2),
    (0x11C43, 0x11C43),
    (0x11C71, 0x11C71),
    (0x12470, 0x12474),
    (0x16B39, 0x16B39),
    (0x16E97, 0x16E97),
    (0x1DA87, 0x1DA87),
    (0x1DA89, 0x1DA8A),
)
_CLAUSE_MARKS = frozenset(chr(cp) for lo, hi in _CLAUSE_MARK_RANGES for cp in range(lo, hi + 1))
# The ASCII part of the same set. These only count with whitespace after them,
# which keeps a decimal comma ("2,5") and a clock ("10:30") whole; the other
# scripts' marks are not used inside numbers.
_ASCII_CLAUSE_MARKS = frozenset(",;:")

# Where an early sentence opening may end. On a model that starts on part of a
# sentence, releasing the opening clause as soon as it exists brings first audio
# forward by the time the rest of the sentence takes to generate: around 450 ms
# for a 120-character opening sentence fed at 200 characters a second. A clause
# boundary is the natural place to stop; past this much text with none in sight,
# a word boundary is the next best thing.
_HEAD_LOOKAHEAD_CHARS = 80
_HEAD_MIN_CHARS = 60
# No opening is shorter than this, so a clause mark nearer the start is passed
# over. Two reasons. A model that starts on part of a sentence voices the opening
# as it arrives, and has only been shown to sound right on clause-sized openings:
# a word-sized one can come out word by word. And a turn whose opening reaches
# the model on its own is neither served from the TTS cache nor stored in it,
# because its full text is not known when the lookup runs, so a greeting split at
# its first comma would never be cached. Kept whole, it is.
_HEAD_MIN_CLAUSE_CHARS = 25

_Span = tuple[int, int]

# Set the first time blingfire refuses a string. Blingfire is still asked on
# every later call; this only keeps the warning to one line per process rather
# than one per reply.
_blingfire_warned = False


def _blingfire_ends(text: str) -> frozenset[int] | None:
    """Offsets where blingfire ends a sentence, used to judge ASCII terminators only.

    Blingfire knows that the stop in ``Dr.``, ``3.14`` or ``example.com`` is not
    a sentence end. It has no opinion on the danda or the ideographic full stop,
    so those are decided by the terminator scan instead.

    ``None`` means it could not be asked, and every ASCII terminator is then
    taken at face value.
    """
    global _blingfire_warned
    try:
        _, offsets = blingfire.text_to_sentences_with_offsets(text)
    except Exception:
        # A native extension, so text it cannot encode to UTF-8 raises instead
        # of returning: one unpaired surrogate, which a streaming LLM produces
        # by splitting an emoji across two deltas, would otherwise lose the
        # whole reply and report it as a connection error. Splitting without
        # abbreviation handling is the better failure.
        if _blingfire_warned:
            logger.debug("[TTS] sentence tokenizer could not use blingfire", exc_info=True)
        else:
            _blingfire_warned = True
            logger.warning(
                "[TTS] sentence tokenizer could not use blingfire on this text, so it "
                "splits it on terminators only; later failures are logged at debug",
                exc_info=True,
            )
        return None
    return frozenset(end for _start, end in offsets)


def _boundaries(text: str) -> list[int]:
    """Offsets to cut at, one after each sentence-ending run of punctuation."""
    ends = _blingfire_ends(text)
    cuts: list[int] = []
    length = len(text)
    i = 0
    while i < length:
        char = text[i]
        if char not in _STERM and char not in _ASCII_TERMINATORS:
            i += 1
            continue
        # Take the whole run, so "？！" or a stop before a closing quote is one
        # boundary rather than several.
        non_ascii = char in _STERM
        saw_closer = False
        j = i + 1
        while j < length:
            nxt = text[j]
            if nxt in _CLOSERS:
                saw_closer = True
            elif nxt in _STERM:
                non_ascii = True
            elif nxt not in _ASCII_TERMINATORS:
                break
            j += 1
        # A mark or joiner after the run belongs to the run's last character.
        while j < length and _binds_to_previous(text[j]):
            j += 1
        if j < length:
            if saw_closer and not text[j].isspace():
                # A quotation closing mid-sentence, as in 「はい。」と言った。
                # Breaking here would put a pause inside one sentence.
                pass
            elif non_ascii or ends is None or j in ends:
                cuts.append(j)
        i = j
    return cuts


def _binds_to_previous(char: str) -> bool:
    """Whether ``char`` attaches to the character before it.

    ``unicodedata.combining`` is the wrong test for a mark: it returns the
    canonical combining class, which is 0 for most Indic and Thai vowel signs
    (Devanagari vowel sign I, Thai sara i, mai han-akat). The general category
    is what identifies a mark. The rest are characters that are not marks but
    still cannot start a cluster.
    """
    return (
        unicodedata.category(char).startswith("M")
        or char in _JOINERS
        or char in _SPACING_VOWELS
        or char in _EMOJI_MODIFIERS
    )


def _binds_to_next(char: str) -> bool:
    """Whether ``char`` attaches to the character after it.

    A virama asks for a conjunct with the following consonant, a joiner binds
    forward by definition, and a Thai or Lao leading vowel is pronounced after
    the consonant stored behind it. Ending a chunk on any of them leaves a
    fragment that renders and reads wrong.
    """
    return (
        char in _JOINERS or char in _LEADING_VOWELS or unicodedata.combining(char) == _VIRAMA_CLASS
    )


def _splits_a_flag(text: str, cut: int) -> bool:
    """Whether cutting here would land between the two halves of a flag."""
    if cut <= 0 or cut >= len(text) or text[cut] not in _REGIONAL_INDICATORS:
        return False
    run = 0
    index = cut - 1
    while index >= 0 and text[index] in _REGIONAL_INDICATORS:
        run += 1
        index -= 1
    return run % 2 == 1


def _safe_cut(text: str, cut: int, end: int) -> int:
    """Move ``cut`` forward until it is not inside a cluster of joined characters.

    Grapheme-cluster safety is the bar. A script written without spaces can
    still be cut mid-word or mid-syllable: deciding where a Thai word ends
    needs a dictionary, which is exactly the per-language dependency this
    tokenizer exists to avoid.
    """
    while cut < end:
        if _binds_to_previous(text[cut]):
            cut += 1
            continue
        if cut > 0 and _binds_to_next(text[cut - 1]):
            # A joiner, virama or leading vowel ends the chunk: take the
            # character it binds to as well.
            cut += 1
            continue
        if _splits_a_flag(text, cut):
            cut += 1
            continue
        break
    return cut


def _apply_budget(text: str, spans: list[_Span], max_chars: int) -> list[_Span]:
    """Cut spans over the budget, at the last space inside it when there is one.

    Applies to every span, not only to text with no terminator: a sentence
    longer than the budget is cut as well. The budget is soft, because a cut
    that would split a character cluster gives way instead.
    """
    out: list[_Span] = []
    for start, end in spans:
        while end - start > max_chars:
            cut = start + max_chars
            at_space = cut
            while at_space > start and not text[at_space - 1].isspace():
                at_space -= 1
            if at_space > start:
                cut = _safe_cut(text, at_space, end)
            else:
                # Scripts that do not use spaces: cut at the budget, but never
                # inside a cluster, which would strand a vowel sign or a tone
                # mark at the head of the next frame.
                cut = _safe_cut(text, cut, end)
            # _safe_cut having run past the end means the only cut inside the
            # budget would have split a cluster: leave the span whole instead.
            if cut >= end:
                break
            out.append((start, cut))
            start = cut
        out.append((start, end))
    return out


def _normalize(text: str) -> str:
    """One emitted piece: inner newlines become spaces and the edges are trimmed."""
    return _INNER_NEWLINES.sub(" ", text).strip()


def _ends_a_sentence(text: str) -> bool:
    """Whether the pending text already reads as a finished sentence.

    Used to leave a short opening alone: it is already as fast as it can be, and
    releasing part of it early would only add a frame.
    """
    tail = text.rstrip()
    while tail and (tail[-1] in _CLOSERS or _binds_to_previous(tail[-1])):
        tail = tail[:-1]
    return bool(tail) and (tail[-1] in _ASCII_TERMINATORS or tail[-1] in _STERM)


def _clause_end(text: str, index: int) -> int | None:
    """Where an opening would end if ``text[index]`` closes a clause, or None.

    A clause mark of any script counts, and so does a dash with whitespace on
    both sides, which is how an LLM writes a clause break with an em dash or an
    en dash. A dash inside a word or a range ("well-known", "10-20") has none.
    """
    char = text[index]
    after = index + 1
    if char in _ASCII_CLAUSE_MARKS:
        return after + 1 if text[after : after + 1].isspace() else None
    if char in _CLAUSE_MARKS:
        # A closing quote or a combining mark after it belongs to the opening,
        # and until the next character arrives it is not known whether one
        # follows, so the mark has to wait for it.
        while after < len(text) and text[after] in _CLOSERS:
            after += 1
        after = _safe_cut(text, after, len(text))
        return after if after < len(text) else None
    if unicodedata.category(char) == "Pd":
        # Not the first thing on its line either, where a dash is a list bullet.
        line = text[:index].rsplit("\n", 1)[-1]
        spaced = line[-1:].isspace() and bool(line.strip())
        return after + 1 if spaced and text[after : after + 1].isspace() else None
    return None


def _head_cut(text: str) -> int | None:
    """How much of an unfinished sentence can be released now, or None to wait.

    The first clause break that leaves at least ``_HEAD_MIN_CLAUSE_CHARS`` of
    text, else, once ``_HEAD_MIN_CHARS`` have accumulated, the last word
    boundary inside the lookahead.
    """
    limit = min(len(text), _HEAD_LOOKAHEAD_CHARS)
    for index in range(limit):
        cut = _clause_end(text, index)
        if cut is not None and len(_normalize(text[:cut])) >= _HEAD_MIN_CLAUSE_CHARS:
            return cut
    if len(text) < _HEAD_MIN_CHARS:
        return None
    # The last word boundary inside the lookahead. A single word running through
    # the whole window leaves nothing safe to cut, so nothing is released.
    index = limit
    while index > _HEAD_MIN_CHARS and not text[index - 1].isspace():
        index -= 1
    return index if index > _HEAD_MIN_CHARS else None


@dataclass
class _SentencePiece(TokenData):
    """A token from this plugin's stream. ``partial`` marks an early opening.

    ``separator`` is what followed the piece in the text: for an opening, a
    space or nothing, exactly as written, so the opening and the rest of its
    sentence join back into the sentence the LLM wrote. Between whole
    sentences it is always a space.
    """

    partial: bool = False
    separator: str = " "


def is_partial_head(token: TokenData) -> bool:
    """Whether this token is the opening of a sentence that is still being written."""
    return isinstance(token, _SentencePiece) and token.partial


def separator_after(token: TokenData) -> str:
    """What to put between this token and the next piece of text.

    A space, except after an opening that the text continued without one, as
    after a Chinese or Japanese comma.
    """
    return token.separator if isinstance(token, _SentencePiece) else " "


def _split(text: str, *, max_chars: int) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(sentence, start, end)`` offsets into ``text``.

    ``BufferedSentenceStream`` consumes its buffer up to each ``end``, so every
    ``end`` advances and is the next span's ``start``, save where a span held
    nothing but whitespace and was dropped.
    """
    if not text.strip():
        return []
    cuts = _boundaries(text)
    spans: list[_Span] = []
    head = 0
    for cut in cuts:
        spans.append((head, cut))
        head = cut
    spans.append((head, len(text)))
    result: list[tuple[str, int, int]] = []
    for start, end in _apply_budget(text, spans, max_chars):
        piece = _normalize(text[start:end])
        if piece:
            result.append((piece, start, end))
    return result


class _SentenceStream(SentenceStream):
    """Emits whole sentences, and the opening of a long one as soon as it exists.

    The buffer holds raw text plus an offset saying how much of it has already
    been handed out, and every piece is a slice of that raw text. That offset is
    what keeps an early opening and the sentence it belongs to consistent:
    ``_split`` collapses inner newlines and trims each piece, so a released
    opening is not a prefix of the sentence as the tokenizer would later render
    it, and subtracting one string from the other would corrupt the text.

    The last sentence is held until the next one starts or the input ends, as
    every livekit tokenizer does. Splitting at a run of whitespace trims it, as
    at any other piece edge, so a doubled space there reaches the gateway as
    one; no word is ever repeated or lost.
    """

    def __init__(
        self,
        *,
        split: Callable[[str], list[tuple[str, int, int]]],
        min_ctx_len: int,
        partial_head: bool,
    ) -> None:
        super().__init__()
        self._split_fnc = split
        self._min_ctx_len = min_ctx_len
        self._partial_head = partial_head
        self._segment_id = shortuuid()
        self._buf = ""
        self._handed_out = 0
        # Cleared once this segment has had its opening, or once a sentence has
        # gone out without needing one. Only the first chunk of a reply is worth
        # releasing early: by the second, audio is already playing.
        self._head_allowed = partial_head

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if not text:
            return
        self._buf += text
        if len(self._buf) - self._handed_out < self._min_ctx_len:
            return
        self._emit_sentences()
        self._maybe_emit_head()

    def flush(self) -> None:
        self._check_not_closed()
        self._emit_sentences()
        self._emit(self._buf[self._handed_out :], partial=False)
        self._buf = ""
        self._handed_out = 0
        self._head_allowed = self._partial_head
        self._segment_id = shortuuid()

    def end_input(self) -> None:
        self.flush()
        self._do_close()

    async def aclose(self) -> None:
        self._do_close()

    def _emit_sentences(self) -> None:
        """Emit every sentence that is certainly finished, holding the last."""
        while True:
            spans = self._split_fnc(self._buf)
            if len(spans) <= 1:
                return
            end = spans[0][2]
            # min() and max() carry an opening that reached past this boundary:
            # nothing already handed out is sent twice, and nothing is dropped.
            self._emit(self._buf[min(self._handed_out, end) : end], partial=False)
            self._buf = self._buf[end:]
            self._handed_out = max(0, self._handed_out - end)
            self._head_allowed = False

    def _maybe_emit_head(self) -> None:
        """Release the opening of the sentence being written, once per segment."""
        if not self._head_allowed:
            return
        pending = self._buf[self._handed_out :]
        if _ends_a_sentence(pending):
            return
        cut = _head_cut(pending)
        if cut is None:
            return
        # A cut after an ASCII mark or at a word takes the space with it, and a
        # cut after another script's clause mark leaves any space in front of
        # the rest; either way, emitting trims it, so say whether there was one.
        spaced = pending[cut - 1].isspace() or pending[cut : cut + 1].isspace()
        self._emit(pending[:cut], partial=True, separator=" " if spaced else "")
        self._handed_out += cut
        self._head_allowed = False

    def _emit(self, text: str, *, partial: bool, separator: str = " ") -> None:
        piece = _normalize(text)
        if piece:
            self._event_ch.send_nowait(
                _SentencePiece(
                    token=piece,
                    segment_id=self._segment_id,
                    partial=partial,
                    separator=separator,
                )
            )


class SentenceTokenizer(tokenize.SentenceTokenizer):
    """Sentence tokenizer that splits on the sentence terminators of every script.

    The plugin's default for ``text_chunking="sentence"``. It needs no language
    setting: blingfire handles Latin punctuation and its abbreviations, and any
    character Unicode classifies as a sentence terminator ends a sentence in
    every other script. Text with no terminator, such as Thai, is cut at a space
    once it passes ``max_chars``.

    Args:
        max_chars: Length at which a piece is cut, at the last space inside
            that length. It applies to every piece, not only to text with no
            terminator, so a sentence longer than this is cut as well. Raise it
            to keep long sentences in one frame, lower it to make a
            terminator-free script stream in smaller ones. A cut that would
            split a character cluster gives way, so a piece can run over.
            Defaults to 200.
        stream_context_len: Minimum buffered text before the stream looks for
            a boundary. Defaults to 10.
        partial_head: Allow a stream to release the opening of a long sentence
            before the sentence is finished, so that a model able to start on
            part of a sentence begins speaking sooner. A stream still does so
            only when ``stream(partial_head=True)`` asks for it, which the
            plugin does when the server supports it. Defaults to True.
    """

    def __init__(
        self,
        *,
        max_chars: int = 200,
        stream_context_len: int = 10,
        partial_head: bool = True,
    ) -> None:
        super().__init__()
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._stream_context_len = stream_context_len
        self._partial_head = partial_head

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return _split(text, max_chars=self._max_chars)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split ``text`` into sentences. ``language`` is accepted and ignored."""
        return [piece for piece, _start, _end in self._split(text)]

    def stream(self, *, language: str | None = None, partial_head: bool = False) -> SentenceStream:
        """Incremental splitter. ``language`` is accepted and ignored.

        The stream holds its last sentence until the next one begins or input
        ends, like every livekit tokenizer; the plugin's sender relies on that
        to know which frame the reply's terminating flush follows.

        ``partial_head=True`` lets the stream release a long opening early, as
        a token marked partial that only this plugin's sender understands. So
        it is off by default, and any other consumer, such as
        ``tts.StreamAdapter``, only ever receives whole sentences. The
        tokenizer's own setting still has to allow it.
        """
        return _SentenceStream(
            split=self._split,
            min_ctx_len=self._stream_context_len,
            partial_head=self._partial_head and partial_head,
        )
