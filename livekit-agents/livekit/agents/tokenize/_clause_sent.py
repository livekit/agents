"""
Clause-level sentence segmentation with multi-language support.

Splits text at clause boundaries (commas, conjunctions, sentence endings)
using language-specific rules from LanguageProfile.  Produces finer-grained
segments than sentence-level tokenizers, which reduces perceived TTS latency
without sacrificing naturalness.

Architecture:
    _find_protected_positions()  ->  positions NOT to split (abbreviations,
                                     numbers, quoted text, URLs, emails)
    _find_split_positions()      ->  valid clause boundaries
    split_clauses()              ->  final clause list with offset tuples
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Language profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageProfile:
    """Language-specific rules for clause boundary detection.

    Attributes:
        name: Short language code (e.g. "en", "tr").
        conjunctions: Words that mark clause boundaries (split BEFORE these).
            Must be lowercase -- matching is case-insensitive.
        abbreviations: Strings ending with "." that should NOT trigger a split
            (e.g. "Mr.", "Dr.").
        decimal_sep: Decimal separator character for the language.
        thousands_sep: Thousands separator character for the language.
        min_clause_len: Minimum clause length (characters) before emitting.
        min_ctx_len: Minimum buffer context for streaming tokenization.
    """

    name: str

    conjunctions: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()

    decimal_sep: str = ","
    thousands_sep: str = "."

    min_clause_len: int = 10
    min_ctx_len: int = 5


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

TURKISH = LanguageProfile(
    name="tr",
    conjunctions=(
        "ve",
        "ama",
        "fakat",
        "ancak",
        "çünkü",
        "veya",
        "ya da",
        "hem",
        "ise",
        "oysa",
        "ayrıca",
        "dolayısıyla",
        "üstelik",
        "halbuki",
        "yoksa",
        "nitekim",
        "zira",
        "yani",
        "hatta",
        "lakin",
        "dahası",
        "böylece",
        "mademki",
        "madem",
        "yalnız",
        "ki",
        "bu yüzden",
        "bu nedenle",
        "ne var ki",
        "buna rağmen",
        "bununla birlikte",
        "öte yandan",
    ),
    abbreviations=(
        "Dr.",
        "Prof.",
        "Doç.",
        "Yrd.",
        "Av.",
        "Uzm.",
        "Alb.",
        "Yzb.",
        "Müh.",
        "No.",
        "vb.",
        "vs.",
    ),
    decimal_sep=",",
    thousands_sep=".",
    min_clause_len=15,
)

GERMAN = LanguageProfile(
    name="de",
    conjunctions=(
        "und",
        "aber",
        "oder",
        "denn",
        "weil",
        "dass",
        "obwohl",
        "jedoch",
        "sondern",
        "deshalb",
        "daher",
        "trotzdem",
        "außerdem",
        "allerdings",
        "dennoch",
        "sofern",
        "sobald",
        "solange",
        "bevor",
        "nachdem",
        "damit",
        "also",
    ),
    abbreviations=(
        "Dr.",
        "Prof.",
        "Hr.",
        "Fr.",
        "Nr.",
        "Str.",
        "bzw.",
        "usw.",
        "z.B.",
        "d.h.",
        "u.a.",
        "o.g.",
        "s.o.",
        "s.u.",
        "Mio.",
        "Mrd.",
        "ca.",
        "evtl.",
        "ggf.",
        "inkl.",
    ),
    decimal_sep=",",
    thousands_sep=".",
    min_clause_len=15,
)

FRENCH = LanguageProfile(
    name="fr",
    conjunctions=(
        "et",
        "mais",
        "ou",
        "car",
        "donc",
        "or",
        "ni",
        "puis",
        "parce que",
        "puisque",
        "cependant",
        "pourtant",
        "toutefois",
        "néanmoins",
        "alors",
        "ensuite",
        "lorsque",
        "quand",
        "tandis que",
        "bien que",
        "afin que",
        "c'est pourquoi",
        "en revanche",
        "par conséquent",
        "de plus",
        "en effet",
        "d'ailleurs",
    ),
    abbreviations=(
        "M.",
        "Mme.",
        "Mlle.",
        "Dr.",
        "Prof.",
        "Jr.",
        "Sr.",
        "av.",
        "apr.",
        "env.",
        "éd.",
        "ex.",
        "fig.",
        "vol.",
        "no.",
        "p.",
        "pp.",
        "etc.",
        "c.-à-d.",
    ),
    decimal_sep=",",
    thousands_sep=".",
    min_clause_len=15,
)

ITALIAN = LanguageProfile(
    name="it",
    conjunctions=(
        "e",
        "ma",
        "o",
        "oppure",
        "però",
        "perché",
        "poiché",
        "quindi",
        "dunque",
        "inoltre",
        "tuttavia",
        "eppure",
        "anzi",
        "ovvero",
        "pertanto",
        "nonostante",
        "sebbene",
        "affinché",
        "anche se",
        "di conseguenza",
        "in quanto",
        "dal momento che",
    ),
    abbreviations=(
        "Sig.",
        "Sig.ra",
        "Dott.",
        "Dott.ssa",
        "Prof.",
        "Prof.ssa",
        "Avv.",
        "Ing.",
        "Arch.",
        "ecc.",
        "es.",
        "pag.",
        "vol.",
        "cap.",
        "fig.",
        "n.",
        "p.",
    ),
    decimal_sep=",",
    thousands_sep=".",
    min_clause_len=15,
)

ENGLISH = LanguageProfile(
    name="en",
    conjunctions=(
        "and",
        "but",
        "or",
        "because",
        "since",
        "although",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "nevertheless",
        "whereas",
        "while",
        "yet",
        "so",
    ),
    abbreviations=(
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "Inc.",
        "Ltd.",
        "Co.",
        "Corp.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "a.m.",
        "p.m.",
        "U.S.",
        "U.K.",
    ),
    decimal_sep=".",
    thousands_sep=",",
    min_clause_len=15,
)


# ---------------------------------------------------------------------------
# Precompiled patterns for protected ranges
# ---------------------------------------------------------------------------

# Quoted text: "...", \u201c...\u201d, \u2018...\u2019, \xab...\xbb
_QUOTED_RE = re.compile(
    r'"[^"]*"'
    r"|\u201c[^\u201d]*\u201d"
    r"|\u2018[^\u2019]*\u2019"
    r"|\xab[^\xbb]*\xbb",
    re.DOTALL,
)

# URLs: http(s)://... until whitespace
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Email addresses: word@word.word
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Time colons: digit:digit
_TIME_RE = re.compile(r"\d:\d")

# Bare domain TLDs: word.com, word.net, etc. (without https://)
_BARE_DOMAIN_RE = re.compile(
    r"[a-zA-Z0-9-]+[.](?:com|net|org|io|gov|edu|me|co|info|dev|app|tr|uk|de|fr|it)\b",
    re.IGNORECASE,
)

# Single-letter abbreviations: J. K. Rowling, Y. A.
_SINGLE_LETTER_ABBR_RE = re.compile(
    r"(?<!\w)[A-Za-z\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc][.]"
)


# ---------------------------------------------------------------------------
# Protected position detection
# ---------------------------------------------------------------------------


def _find_protected_positions(text: str, profile: LanguageProfile) -> set[int]:
    """Find character positions that must NOT be used as split points.

    Protects:
    - Periods inside abbreviations  (Dr. Prof. vs.)
    - Decimal separators between digits  (3,14  or  3.14)
    - Thousands separators between digits  (1.250  or  1,250)
    - Colons in time formats  (14:30)
    - All punctuation inside quoted text  ("no, not here")
    - All punctuation inside URLs  (https://example.com/path?q=1)
    - All punctuation inside email addresses  (user@example.com)
    - Periods in bare domain names  (google.com, example.org)
    - Periods in single-letter abbreviations  (J. K. Rowling)
    """
    protected: set[int] = set()

    # --- Range-based protection (quotes, URLs, emails, bare domains) ---
    for pattern in (_QUOTED_RE, _URL_RE, _EMAIL_RE, _BARE_DOMAIN_RE):
        for m in pattern.finditer(text):
            protected.update(range(m.start(), m.end()))

    # --- Point-based protection ---

    # Abbreviation periods
    for abbr in profile.abbreviations:
        for m in re.finditer(re.escape(abbr), text, re.IGNORECASE):
            for i in range(m.start(), m.end()):
                if text[i] == ".":
                    protected.add(i)

    # Number separators: digit<sep>digit
    for sep in {profile.decimal_sep, profile.thousands_sep}:
        for m in re.finditer(r"\d" + re.escape(sep) + r"\d", text):
            protected.add(m.start() + 1)

    # Time colons: 14:30
    for m in _TIME_RE.finditer(text):
        protected.add(m.start() + 1)

    # Single-letter abbreviations: J. K. Rowling, Y. A.
    for m in _SINGLE_LETTER_ABBR_RE.finditer(text):
        protected.add(m.end() - 1)

    return protected


# ---------------------------------------------------------------------------
# Split position detection
# ---------------------------------------------------------------------------


def _find_split_positions(
    text: str,
    profile: LanguageProfile,
    protected: set[int],
) -> list[int]:
    """Find valid clause boundary positions in *text*.

    Returns a sorted list of character offsets where the text should be split.
    Each position marks the start of a new clause.
    """
    positions: list[int] = []

    # Punctuation: , ; : . ! ? followed by whitespace
    # Split AFTER the punctuation (at the whitespace boundary).
    for m in re.finditer(r"[,;:.!?](?=\s)", text):
        if m.start() not in protected:
            positions.append(m.end())

    # Conjunctions: split BEFORE the conjunction word.
    # Longest-first to avoid partial matches ("ya" vs "ya da").
    if profile.conjunctions:
        sorted_conjs = sorted(profile.conjunctions, key=len, reverse=True)
        conj_alts = "|".join(re.escape(c) for c in sorted_conjs)
        conj_pattern = r"(?<=\s)(" + conj_alts + r")(?=\s)"
        for m in re.finditer(conj_pattern, text, re.IGNORECASE):
            if m.start() not in protected:
                positions.append(m.start())

    return sorted(set(positions))


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------


def split_clauses(
    text: str,
    *,
    profile: LanguageProfile = ENGLISH,
    min_clause_len: int | None = None,
    retain_format: bool = False,
) -> list[tuple[str, int, int]]:
    """Split *text* at clause boundaries.

    Returns a list of ``(clause_text, start, end)`` tuples compatible with
    LiveKit's ``BufferedSentenceStream``.  Offsets are relative to the input
    text and cover it contiguously (no gaps).

    Args:
        text: Input text to split.
        profile: Language-specific rules.
        min_clause_len: Override ``profile.min_clause_len``.  Clauses shorter
            than this are merged with the next clause.
        retain_format: If True, preserve original whitespace/newlines in
            clause text.  If False (default), strip whitespace from each
            clause.
    """
    if not text:
        return []
    if not text.strip():
        return [(text, 0, len(text))]

    protected = _find_protected_positions(text, profile)
    positions = _find_split_positions(text, profile, protected)

    def _fmt(s: str) -> str:
        return s if retain_format else s.strip()

    if not positions:
        return [(_fmt(text), 0, len(text))]

    min_len = min_clause_len if min_clause_len is not None else profile.min_clause_len
    clauses: list[tuple[str, int, int]] = []
    start = 0

    for pos in positions:
        chunk = _fmt(text[start:pos])
        if len(chunk) >= min_len:
            clauses.append((chunk, start, pos))
            start = pos

    # Remaining text
    if start < len(text):
        remainder = _fmt(text[start:])
        if clauses and len(remainder) < min_len:
            # Merge short tail with last clause
            _, prev_start, _ = clauses.pop()
            clauses.append((_fmt(text[prev_start:]), prev_start, len(text)))
        else:
            clauses.append((remainder, start, len(text)))

    return clauses if clauses else [(_fmt(text), 0, len(text))]
