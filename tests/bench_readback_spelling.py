"""Bench for the spelled read-back added in livekit/agents#6990.

Two questions raised in review:

1. Does the model hallucinate letters when it spells a value itself (the prompt hands it
   ``Shayne Cole``) versus when the prompt already carries the spaced form
   (``S h a y n e C o l e``)? ``GetEmailTask`` and ``GetPhoneNumberTask`` already pass the
   spaced form; ``GetNameTask`` and ``GetAddressTask`` do not.
2. Would a NATO ("S as in Sierra") read-back survive the audio channel better than plain
   letters, so a third attempt should escalate to it?

Part A drives the real tasks through ``AgentSession`` in audio modality: the caller states a
value, refuses the natural read-back, restates it, and the spelled read-back is scored
against the known letters. Arms rewrite only the spelled instruction the task returns.

Part B synthesizes a letter-by-letter and a NATO read-back with TTS, transcribes each with
STT, and scores how many letters come back. STT stands in for the caller's ear.

    uv run python tests/bench_readback_spelling.py
    uv run python tests/bench_readback_spelling.py --models openai/gpt-4.1 openai/gpt-4.1-mini
    uv run python tests/bench_readback_spelling.py --only audio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import aiohttp
from dotenv import load_dotenv

from livekit.agents import AgentSession, beta, inference
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.run_result import (
    ChatMessageEvent,
    FunctionCallOutputEvent,
    RunResult,
)

NAMES: list[tuple[str, str]] = [
    ("Shayne", "Cole"),
    ("Siobhan", "Keohane"),
    ("Krzysztof", "Brzezinski"),
    ("Nguyen", "Pham"),
    ("Xiomara", "Okonkwo"),
    ("Bartholomew", "Featherstonehaugh"),
    ("Dhruv", "Venkataraghavan"),
    ("Ptolemy", "Schwarzschild"),
    ("Mikaela", "Thibodeaux"),
    ("Rhiannon", "Llewellyn"),
    ("Jaxon", "Mbeki"),
    ("Yevgeniya", "Kolesnikova"),
    ("Oluwaseun", "Adebayo"),
    ("Wyatt", "Pfeiffer"),
    ("Cillian", "Gallagher"),
    ("Phoebe", "Nietzsche"),
    ("Leighton", "Cholmondeley"),
    ("Tomasz", "Wojciechowski"),
    ("Aoife", "Ruaidhri"),
    ("Deshawn", "Ugochukwu"),
]

# (house number, street name, suffix, unit, locality)
STREETS: list[tuple[str, str, str, str, str]] = [
    ("4821", "Wyckoff", "Avenue", "apartment 3B", "Ridgewood, New York 11385"),
    ("112", "Schermerhorn", "Street", "", "Brooklyn, New York 11201"),
    ("7", "Kosciuszko", "Street", "unit 2", "Brooklyn, New York 11221"),
    ("3300", "Tchoupitoulas", "Street", "", "New Orleans, Louisiana 70115"),
    ("15", "Rhinelander", "Avenue", "", "Bronx, New York 10461"),
    ("905", "Nostrand", "Avenue", "floor 2", "Brooklyn, New York 11225"),
    ("60", "Bleecker", "Street", "", "New York, New York 10012"),
    ("214", "Massapequa", "Avenue", "", "Massapequa, New York 11758"),
    ("1400", "Sepulveda", "Boulevard", "suite 410", "Los Angeles, California 90025"),
    ("88", "Featherstone", "Road", "", "Ealing, London W5 3PY"),
    ("29", "Ponsonby", "Terrace", "", "Auckland 1011"),
    ("501", "Gowanus", "Way", "", "Brooklyn, New York 11217"),
]

NATO: dict[str, str] = {
    "a": "alfa",
    "b": "bravo",
    "c": "charlie",
    "d": "delta",
    "e": "echo",
    "f": "foxtrot",
    "g": "golf",
    "h": "hotel",
    "i": "india",
    "j": "juliett",
    "k": "kilo",
    "l": "lima",
    "m": "mike",
    "n": "november",
    "o": "oscar",
    "p": "papa",
    "q": "quebec",
    "r": "romeo",
    "s": "sierra",
    "t": "tango",
    "u": "uniform",
    "v": "victor",
    "w": "whiskey",
    "x": "xray",
    "y": "yankee",
    "z": "zulu",
}
_NATO_ALIASES = {"alpha": "alfa", "juliet": "juliett", "x-ray": "xray", "whisky": "whiskey"}
_NATO_TO_LETTER = {w: letter for letter, w in NATO.items()}

Arm = Literal["raw", "spaced", "nato_raw", "nato_spaced"]
Scenario = Literal["verify_spelling", "respell", "restate"]


def spaced(value: str) -> str:
    return " ".join(c for c in value if c.isalpha())


def letters_of(value: str) -> str:
    return "".join(c for c in value if c.isalpha()).lower()


def nato_of(value: str) -> str:
    return ", ".join(
        f"{c.upper()} as in {NATO[c.lower()].capitalize()}" for c in value if c.isalpha()
    )


_PAIR_RE = re.compile(r"\b([A-Za-z])\b[\s,.\-]*(?:as in|like in)\s+([A-Za-z\-]+)", re.I)


def extract_letters(text: str) -> tuple[str, list[str]]:
    """Recover the spelled sequence from a spoken read-back.

    Returns the letters and the NATO code words the speaker used that are not canonical.
    Plain runs need at least two consecutive single-letter tokens so prose "a"/"I" are
    ignored. A NATO pair contributes the letter, falling back to the code word when the two
    disagree because the word is what carries across a bad line.
    """
    pairs = _PAIR_RE.findall(text)
    if len(pairs) >= 2:
        out, bad_words = [], []
        for letter, word in pairs:
            word_n = _NATO_ALIASES.get(word.lower(), word.lower())
            canonical = NATO[letter.lower()]
            if word_n != canonical:
                bad_words.append(f"{letter}={word}")
            out.append(_NATO_TO_LETTER.get(word_n, letter.lower()))
        return "".join(out), bad_words

    # "That's S H A Y N E" must not contribute the "s" of "That's" to the run
    tokens = re.sub(r"[^A-Za-z]+", " ", text.replace("'", "").replace("\u2019", "")).split()
    out, run = [], []
    for tok in tokens + [""]:
        if len(tok) == 1:
            run.append(tok.lower())
            continue
        if len(run) >= 2:
            out.extend(run)
        run = []
    return "".join(out), []


@dataclass
class Case:
    part: Literal["llm", "audio"]
    task: str
    model: str
    arm: str
    scenario: str
    value: str
    truth: str
    got: str
    exact: bool
    contains: bool
    recalled: bool
    bad_nato_words: list[str]
    reply: str
    error: str = ""


# ---------------------------------------------------------------- Part A: LLM spelling


def _name_task(arm: Arm, *, verify_spelling: bool) -> beta.workflows.GetNameTask:
    class _Task(beta.workflows.GetNameTask):
        async def _update_name_impl(  # type: ignore[override]
            self,
            ctx: RunContext,
            first_name: str | None = None,
            middle_name: str | None = None,
            last_name: str | None = None,
        ) -> str | None:
            out = await super()._update_name_impl(ctx, first_name, middle_name, last_name)
            if out is None or arm == "raw":
                return out
            full = f"{first_name or ''} {last_name or ''}".strip()
            needle = f"Spell out the name letter by letter for verification: {full}"
            if needle not in out:
                return out
            value = spaced(full) if arm.endswith("spaced") else full
            if arm.startswith("nato"):
                line = (
                    "Spell out the name with the NATO phonetic alphabet, "
                    f"'S as in Sierra', for verification: {value}"
                )
            else:
                line = f"Spell out the name letter by letter for verification: {value}"
            return out.replace(needle, line)

    return _Task(first_name=True, last_name=True, verify_spelling=verify_spelling)


def _address_task(arm: Arm, street_name: str) -> beta.workflows.GetAddressTask:
    class _Task(beta.workflows.GetAddressTask):
        async def _update_address_impl(  # type: ignore[override]
            self,
            street_address: str,
            unit_number: str,
            locality: str,
            country: str,
            ctx: RunContext,
        ) -> str | None:
            out = await super()._update_address_impl(
                street_address, unit_number, locality, country, ctx
            )
            if out is None or arm == "raw":
                return out
            needle = "spelling the street name letter by letter: "
            if needle not in out:
                return out
            value = spaced(street_name) if arm.endswith("spaced") else street_name
            if arm.startswith("nato"):
                line = f"spelling the street name with the NATO phonetic alphabet, {value}: "
            else:
                line = f"spelling the street name letter by letter, {value}: "
            return out.replace(needle, line)

    return _Task()


def _reply_after_recall(result: RunResult[None], tool: str) -> tuple[bool, str]:
    recalled = False
    parts: list[str] = []
    for ev in result.events:
        if isinstance(ev, FunctionCallOutputEvent) and ev.item.name == tool:
            recalled = True
            parts.clear()
        elif isinstance(ev, ChatMessageEvent) and ev.item.role == "assistant":
            parts.append(ev.item.text_content or "")
    return recalled, " ".join(p for p in parts if p)


async def run_name_case(model: str, arm: Arm, scenario: Scenario, first: str, last: str) -> Case:
    full = f"{first} {last}"
    restate = f"No, that's not right. It's {first} {last}."
    respell = (
        f"No, that's not right. It's {first}, {spaced(first).upper()}, "
        f"{last}, {spaced(last).upper()}."
    )
    async with inference.LLM(model=model) as llm_v, AgentSession(llm=llm_v) as sess:
        await sess.start(_name_task(arm, verify_spelling=scenario == "verify_spelling"))
        result = await sess.run(
            user_input=f"Hi, my name is {first} {last}.", input_modality="audio"
        )
        if scenario != "verify_spelling":
            result = await sess.run(
                user_input=restate if scenario == "restate" else respell, input_modality="audio"
            )
    recalled, reply = _reply_after_recall(result, "update_name")
    got, bad = extract_letters(reply)
    truth = letters_of(full)
    return Case(
        part="llm",
        task="name",
        model=model,
        arm=arm,
        scenario=scenario,
        value=full,
        truth=truth,
        got=got,
        exact=got == truth,
        contains=truth in got,
        recalled=recalled,
        bad_nato_words=bad,
        reply=reply,
    )


async def run_address_case(
    model: str, arm: Arm, scenario: Scenario, street: tuple[str, str, str, str, str]
) -> Case:
    number, name, suffix, unit, locality = street
    unit_part = f", {unit}" if unit else ""
    said = f"{number} {name} {suffix}{unit_part}, {locality}, United States"
    spelled = (
        f"{number} {name}, {spaced(name).upper()}, {suffix}{unit_part}, {locality}, United States"
    )
    async with inference.LLM(model=model) as llm_v, AgentSession(llm=llm_v) as sess:
        await sess.start(_address_task(arm, name))
        await sess.run(user_input=f"I live at {said}.", input_modality="audio")
        result = await sess.run(
            user_input=f"No, that's not right. It's {spelled if scenario == 'respell' else said}.",
            input_modality="audio",
        )
    recalled, reply = _reply_after_recall(result, "update_address")
    got, bad = extract_letters(reply)
    truth = letters_of(name)
    return Case(
        part="llm",
        task="address",
        model=model,
        arm=arm,
        scenario=scenario,
        value=said,
        truth=truth,
        got=got,
        exact=got == truth,
        contains=truth in got,
        recalled=recalled,
        bad_nato_words=bad,
        reply=reply,
    )


# ---------------------------------------------------------------- Part B: audio round trip


async def run_audio_case(
    tts_v: inference.TTS, stt_v: object, style: Literal["letters", "nato"], first: str, last: str
) -> Case:
    full = f"{first} {last}"
    if style == "letters":
        said = f"That's {', '.join(spaced(first).upper().split())}. {', '.join(spaced(last).upper().split())}. Is this correct?"  # noqa: E501
    else:
        said = f"That's {nato_of(first)}. {nato_of(last)}. Is this correct?"

    frames: AudioBuffer = [ev.frame async for ev in tts_v.synthesize(said)]
    event = await stt_v.recognize(buffer=frames)  # type: ignore[attr-defined]
    transcript = event.alternatives[0].text if event.alternatives else ""
    got, bad = extract_letters(transcript)
    truth = letters_of(full)
    return Case(
        part="audio",
        task="name",
        model=f"{tts_v.model}->deepgram/nova-3",
        arm=style,
        scenario="tts_stt",
        value=said,
        truth=truth,
        got=got,
        exact=got == truth,
        contains=truth in got,
        recalled=True,
        bad_nato_words=bad,
        reply=transcript,
    )


# ---------------------------------------------------------------- driver


def _rescored(raw: dict) -> Case:
    c = Case(**raw)
    if c.error:
        return c
    c.got, c.bad_nato_words = extract_letters(c.reply)
    c.exact, c.contains = c.got == c.truth, c.truth in c.got
    return c


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "   n/a"


def report(cases: list[Case]) -> None:
    groups: dict[tuple[str, ...], list[Case]] = defaultdict(list)
    for c in cases:
        groups[(c.part, c.task, c.model, c.arm, c.scenario)].append(c)

    print()
    print(
        f"{'part':6} {'task':8} {'model':34} {'arm':12} {'scenario':9} {'n':>3} "
        f"{'recalled':>8} {'exact':>7} {'contains':>9} {'bad NATO':>9} {'errors':>6}"
    )
    for key in sorted(groups):
        cs = groups[key]
        ok = [c for c in cs if not c.error]
        scored = [c for c in ok if c.recalled]
        print(
            f"{key[0]:6} {key[1]:8} {key[2]:34} {key[3]:12} {key[4]:9} {len(cs):3d} "
            f"{_pct(len(scored), len(ok)):>8} "
            f"{_pct(sum(c.exact for c in scored), len(scored)):>7} "
            f"{_pct(sum(c.contains for c in scored), len(scored)):>9} "
            f"{sum(len(c.bad_nato_words) for c in scored):9d} "
            f"{len(cs) - len(ok):6d}"
        )

    misses = [c for c in cases if not c.error and c.recalled and not c.contains]
    if misses:
        print("\nmisses:")
        for c in misses:
            print(f"- [{c.part}/{c.task}/{c.arm}/{c.scenario}] {c.model} {c.value!r}")
            print(f"    truth={c.truth} got={c.got}")
            print(f"    reply={c.reply!r}")
    errors = [c for c in cases if c.error]
    if errors:
        print("\nerrors:")
        for c in errors:
            print(f"- [{c.part}/{c.task}/{c.arm}/{c.scenario}] {c.model} {c.value!r}: {c.error}")


async def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=["openai/gpt-4.1"])
    p.add_argument("--arms", nargs="+", default=["raw", "spaced", "nato_raw", "nato_spaced"])
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["verify_spelling", "respell"],
        choices=["verify_spelling", "respell", "restate"],
        help="verify_spelling: spelled on the first turn, nothing spelled in context yet; "
        "respell: the caller refuses and spells it; restate: the caller refuses with the word only",
    )
    p.add_argument("--only", choices=["llm", "audio"], default=None)
    p.add_argument("--limit", type=int, default=None, help="use only the first N names/streets")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--tts", default="cartesia/sonic-3")
    p.add_argument("--dump", type=Path, default=None, help="write every case as JSON lines")
    p.add_argument(
        "--rescore",
        nargs="+",
        type=Path,
        default=None,
        help="re-score and report the cases in these dumps instead of running anything",
    )
    args = p.parse_args()

    if args.rescore:
        report([_rescored(json.loads(line)) for f in args.rescore for line in f.open()])
        return

    load_dotenv()
    names = NAMES[: args.limit] if args.limit else NAMES
    streets = STREETS[: args.limit] if args.limit else STREETS
    sem = asyncio.Semaphore(args.concurrency)
    cases: list[Case] = []

    async def guarded(coro, fallback: Case) -> Case:
        async with sem:
            try:
                c = await asyncio.wait_for(coro, timeout=120)
            except Exception as e:  # noqa: BLE001
                c = fallback
                c.error = f"{type(e).__name__}: {e}"
            cases.append(c)
            sys.stderr.write(".")
            sys.stderr.flush()
            return c

    def blank(part: str, task: str, model: str, arm: str, scenario: str, value: str) -> Case:
        return Case(part, task, model, arm, scenario, value, "", "", False, False, False, [], "")  # type: ignore[arg-type]

    jobs = []
    if args.only != "audio":
        for _ in range(args.trials):
            for model in args.models:
                for arm in args.arms:
                    for scenario in args.scenarios:
                        for first, last in names:
                            jobs.append(
                                guarded(
                                    run_name_case(model, arm, scenario, first, last),
                                    blank("llm", "name", model, arm, scenario, f"{first} {last}"),
                                )
                            )
                    for scenario in args.scenarios:
                        if arm.startswith("nato") or scenario == "verify_spelling":
                            continue
                        for street in streets:
                            jobs.append(
                                guarded(
                                    run_address_case(model, arm, scenario, street),
                                    blank("llm", "address", model, arm, scenario, street[1]),
                                )
                            )

    if args.only != "llm":
        from livekit.plugins import deepgram

        # the plugins reach for the job's shared http session when none is given
        http = aiohttp.ClientSession()
        tts_v = inference.TTS(model=args.tts, http_session=http)
        stt_v = deepgram.STT(model="nova-3", http_session=http)
        for _ in range(args.trials):
            for style in ("letters", "nato"):
                for first, last in names:
                    jobs.append(
                        guarded(
                            run_audio_case(tts_v, stt_v, style, first, last),
                            blank("audio", "name", args.tts, style, "tts_stt", f"{first} {last}"),
                        )
                    )

    await asyncio.gather(*jobs)
    if args.only != "llm":
        await http.close()
    sys.stderr.write("\n")
    report(cases)

    if args.dump:
        with args.dump.open("w") as f:
            for c in cases:
                f.write(json.dumps(asdict(c)) + "\n")
        print(f"\nwrote {len(cases)} cases to {args.dump}")


if __name__ == "__main__":
    asyncio.run(main())
