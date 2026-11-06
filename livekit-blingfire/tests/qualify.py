"""Require exact native API outcomes from the independently frozen platform oracles."""

from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from importlib.machinery import ExtensionFileLoader
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Callable


Method: TypeAlias = Literal[
    "text_to_sentences",
    "text_to_words",
    "text_to_sentences_with_offsets",
    "text_to_words_with_offsets",
]
Value: TypeAlias = str | tuple[str, list[tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class Case:
    identity: str
    text: str


class EncodedCase(TypedDict):
    identity: str
    codepoints: list[int]


@dataclass(frozen=True, slots=True)
class Outcome:
    case: str
    method: Method
    value: Value | None
    error_type: str | None
    error_text: str | None


def cases() -> list[Case]:
    texts = [
        "",
        " ",
        "\n\t",
        "\u3000",
        "\u00a0",
        "\u200b",
        "\ufeff",
        "\u200d",
        "\x00",
        "one\x00two.",
        "\ud800",
        "\udfff",
        "a\ud800b",
        "Hello world. Next sentence!",
        "Dr. Smith paid $3.50. Then he left.",
        "Café 👩🏽‍💻. 你好。",
        "汉字。かな\uff01",
        "السَّلَامُ عَلَيْكُمْ. مرحبًا!",
        "यह पहली पंक्ति है। यह दूसरी है॥",
        "ภาษาไทยทดสอบ. ข้อความใหม่!",
        "Καλημέρα. До свидания!",
        "한글 문장입니다. 다음 문장입니다.",
        "Email a@example.org, then visit https://example.com/a.b?q=c.",
        "<speak>Hello <emphasis>world.</emphasis> Next!</speak>",
        "a\r\nb\rc\nd\u2028e\u2029f",
        "foo—bar… baz?! (one.) [two!] «three?»",
        "e\u0301 é ﬃ ① ㍿ Ⅷ ﷽ \U0001d518\U0001d52b\U0001d526\U0001d520\U0001d52c\U0001d521\U0001d522",
        "👨‍👩‍👧‍👦 🇨🇦 🏳️‍🌈 👍🏿",
    ]
    result = [Case(f"fixed-{index:03}", text) for index, text in enumerate(texts)]
    rng = random.Random(3152)  # noqa: S311 -- Frozen test inputs, not cryptography.
    alphabet = "Abc xyz.?!,;:'\"()[]—…\n\t\x00éößΩЖשعअก漢かな한글👩🏽‍💻\u0301\u200b\ufeff\u3000"
    for index in range(600):
        size = rng.randrange(1, 300)
        result.append(Case(f"mixed-{index:04}", "".join(rng.choice(alphabet) for _ in range(size))))
    for index in range(200):
        points: list[str] = []
        for _ in range(rng.randrange(1, 90)):
            codepoint = rng.randrange(0x110000)
            if not 0xD800 <= codepoint <= 0xDFFF:
                points.append(chr(codepoint))
        result.append(Case(f"scalar-{index:04}", "".join(points)))
    for index, text in enumerate(texts[13:]):
        result.append(Case(f"long-{index:03}", (text + " \n") * 256))
    return result


def functions() -> dict[Method, Callable[[str], Value]]:
    from livekit import blingfire

    return {
        "text_to_sentences": blingfire.text_to_sentences,
        "text_to_words": blingfire.text_to_words,
        "text_to_sentences_with_offsets": blingfire.text_to_sentences_with_offsets,
        "text_to_words_with_offsets": blingfire.text_to_words_with_offsets,
    }


def main() -> None:
    if len(sys.argv) != 2:
        message = "expected output JSON path"
        raise ValueError(message)
    output = Path(sys.argv[1])
    if sys.version_info[:3] != (3, 15, 2):
        message = "Qualification requires CPython3.15.2"
        raise RuntimeError(message)
    if metadata.version("livekit-blingfire") != "1.1.0":
        message = "Unexpected native distribution"
        raise RuntimeError(message)
    spec = find_spec("lk_blingfire")
    if spec is None or spec.origin is None or not isinstance(spec.loader, ExtensionFileLoader):
        message = "Qualification requires a genuine native extension"
        raise RuntimeError(message)
    native = Path(spec.origin).resolve()
    if not native.is_relative_to(Path(sys.prefix).resolve()):
        message = "Native producer is outside the owned qualification environment"
        raise RuntimeError(message)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        expected = "b963ea191ac018cf1f72e76db2d8ab9c8a5d769f926e7982b199dfde59f0d76c"
    elif sys.platform == "linux" and platform.machine() in {"aarch64", "x86_64"}:
        expected = "5a088672040dd9fd024dd90fbd1f5120fceaff6210ae91f4dd48a58000644529"
    else:
        message = "Unqualified platform"
        raise RuntimeError(message)
    corpus = cases()
    encoded: list[EncodedCase] = [
        {"identity": case.identity, "codepoints": [ord(char) for char in case.text]}
        for case in corpus
    ]
    corpus_bytes = TypeAdapter(list[EncodedCase]).dump_json(encoded)
    if (
        sha256(corpus_bytes).hexdigest()
        != "e53b6619f0aa271a651b37409008bbd301f6806b22537e8084e369c232fe7208"
    ):
        message = "The frozen corpus changed"
        raise ValueError(message)
    adapter = TypeAdapter[Value](Value)
    rows: list[Outcome] = []
    for case in corpus:
        for method, function in functions().items():
            try:
                value = adapter.validate_python(function(case.text), strict=True)
                row = Outcome(case.identity, method, value, None, None)
            except Exception as error:
                row = Outcome(case.identity, method, None, type(error).__name__, str(error))
            rows.append(row)
    output.write_bytes(TypeAdapter(list[Outcome]).dump_json(rows, indent=2))
    receipt = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "native": str(native),
        "native_sha256": sha256(native.read_bytes()).hexdigest(),
        "distribution": metadata.version("livekit-blingfire"),
        "expected_output_sha256": expected,
        "exact": sha256(output.read_bytes()).hexdigest() == expected,
        "cases": len(corpus),
        "outcomes": len(rows),
        "corpus_sha256": sha256(corpus_bytes).hexdigest(),
        "output_sha256": sha256(output.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if sha256(output.read_bytes()).hexdigest() != expected:
        message = "Native outcomes differ from the independent platform oracle"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
