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

"""Text normalisation before TTS synthesis.

Strips punctuation and symbol characters (minus a small allowlist) and expands
currency amounts, so the model only ever receives speakable text.
"""

import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal

from num2words import num2words  # type: ignore[import-untyped]

_CURRENCY_MAP = {
    "$": ("dollar", "cent"),
    "£": ("pound", "penny"),
    "€": ("euro", "cent"),
}


def _expand_currency(m: re.Match) -> str:
    symbol = m.group(1)
    number = m.group(2).replace(",", "")
    name, frac_name = _CURRENCY_MAP.get(symbol, ("dollar", "cent"))
    amount = Decimal(number)
    whole = int(amount)
    cents = int((amount - whole).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)
    parts = [f"{num2words(whole)} {name}{'s' if whole != 1 else ''}"]
    if cents:
        parts.append(f"{num2words(cents)} {frac_name}{'s' if cents != 1 else ''}")
    return " and ".join(parts)


_CURRENCY_RE = re.compile(r"([$£€])([\d,]+(?:\.\d{1,2})?)")

_KEEP = set(",'?.-%$!:/£⁇¿")

_REPLACE = {
    "—": ",",
    "–": "-",
    "’": "'",
    "@": " at ",
    "&": "and",
    "×": " times ",
    "÷": " divided by ",
    "±": " plus or minus ",
    "°": " degrees ",
}


def _strip_punctuation_and_symbols(text: str) -> str:
    """Drop punctuation/symbol characters, keeping the _KEEP allowlist.

    Only the characters actually present in ``text`` are inspected, so there is
    no import-time cost from scanning the whole Unicode range.
    """
    out: list[str] = []
    for char in text:
        replacement = _REPLACE.get(char)
        if replacement is not None:
            out.append(replacement)
        elif char in _KEEP:
            out.append(char)
        elif unicodedata.category(char)[0] in ("P", "S"):
            continue  # punctuation or symbol — drop it
        else:
            out.append(char)
    return "".join(out)


_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_SPACE_BEFORE_COMMA = re.compile(r" ,")
_MULTI_PUNCT = re.compile(r"[,'?.%$!:/£⁇¿-]{2,}")


def clean_text(text: str) -> str:
    text = _CURRENCY_RE.sub(_expand_currency, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _EMAIL.sub(lambda m: m.group().replace(".", " dot ").replace("@", " at "), text)
    text = _strip_punctuation_and_symbols(text)
    text = re.sub(r" {2,}", " ", text)
    text = _SPACE_BEFORE_COMMA.sub(",", text)
    text = _MULTI_PUNCT.sub(lambda m: m.group()[-1], text)
    return text
