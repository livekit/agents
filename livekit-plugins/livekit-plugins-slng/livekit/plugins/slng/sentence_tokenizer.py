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
2. For text with no terminator at all (Thai and Lao have none), cut at the last
   space once a span exceeds ``max_chars``, so a reply still streams.

No minimum sentence length: a sentence boundary is a natural pause, and a
character-count minimum is exactly what starves scripts with short sentences.
"""

from __future__ import annotations

import re
import unicodedata

from livekit import blingfire
from livekit.agents import tokenize

# Imported by name because the ``tokenize`` method below shadows the module
# inside the class body, where this annotation is resolved.
from livekit.agents.tokenize import SentenceStream

# Every Sentence_Terminal code point above U+007F, taken from Unicode's PropList
# (https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt, PropList-18.0.0.txt
# dated 2026-08-07): 169 code points in 86 ranges. ASCII ``. ! ?`` are left to
# blingfire, which knows when a full stop is not a sentence end. To regenerate,
# parse the "; Sentence_Terminal" lines of that file, expand the ranges and drop
# everything below U+0080.
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

_Span = tuple[int, int]


def _blingfire_ends(text: str) -> frozenset[int]:
    """Offsets where blingfire ends a sentence, used to judge ASCII terminators only.

    Blingfire knows that the stop in ``Dr.``, ``3.14`` or ``example.com`` is not
    a sentence end. It has no opinion on the danda or the ideographic full stop,
    so those are decided by the terminator scan instead.
    """
    _, offsets = blingfire.text_to_sentences_with_offsets(text)
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
        if j < length:
            if saw_closer and not text[j].isspace():
                # A quotation closing mid-sentence, as in 「はい。」と言った。
                # Breaking here would put a pause inside one sentence.
                pass
            elif non_ascii or j in ends:
                cuts.append(j)
        i = j
    return cuts


def _apply_budget(text: str, spans: list[_Span], max_chars: int) -> list[_Span]:
    """Cut spans over the budget, at the last space inside it when there is one."""
    out: list[_Span] = []
    for start, end in spans:
        while end - start > max_chars:
            cut = start + max_chars
            at_space = cut
            while at_space > start and not text[at_space - 1].isspace():
                at_space -= 1
            if at_space > start:
                cut = at_space
            else:
                # Scripts that do not use spaces: cut at the budget, but never
                # in front of a combining mark, which belongs to the letter
                # before it.
                while cut < end and unicodedata.combining(text[cut]):
                    cut += 1
            if cut >= end:
                break
            out.append((start, cut))
            start = cut
        out.append((start, end))
    return out


def _split(text: str, *, max_chars: int) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(sentence, start, end)`` offsets into ``text``.

    ``BufferedSentenceStream`` consumes its buffer up to each ``end``, so the
    spans cover the text contiguously and every ``end`` advances.
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
        piece = _INNER_NEWLINES.sub(" ", text[start:end]).strip()
        if piece:
            result.append((piece, start, end))
    return result


class SentenceTokenizer(tokenize.SentenceTokenizer):
    """Sentence tokenizer that splits on the sentence terminators of every script.

    The plugin's default for ``text_chunking="sentence"``. It needs no language
    setting: blingfire handles Latin punctuation and its abbreviations, and any
    character Unicode classifies as a sentence terminator ends a sentence in
    every other script. Text with no terminator, such as Thai, is cut at a space
    once it passes ``max_chars``.

    Args:
        max_chars: Length at which text with no sentence terminator is cut, at
            the last space inside that length. Long enough that a long
            punctuated sentence is never cut, short enough that a reply in a
            script without terminators still streams. Defaults to 200.
        stream_context_len: Minimum buffered text before the stream looks for
            a boundary. Defaults to 10.
    """

    def __init__(self, *, max_chars: int = 200, stream_context_len: int = 10) -> None:
        super().__init__()
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._stream_context_len = stream_context_len

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return _split(text, max_chars=self._max_chars)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split ``text`` into sentences. ``language`` is accepted and ignored."""
        return [piece for piece, _start, _end in self._split(text)]

    def stream(self, *, language: str | None = None) -> SentenceStream:
        """Incremental splitter. ``language`` is accepted and ignored.

        The stream holds its last sentence until the next one begins or input
        ends, like every livekit tokenizer; the plugin's sender relies on that
        to put the flush flag on the reply's final frame.
        """
        return tokenize.BufferedSentenceStream(
            tokenizer=self._split,
            min_token_len=1,
            min_ctx_len=self._stream_context_len,
        )
