#!/usr/bin/env python3
"""A/B test: the special-occasion instruction (run SR_VJtN4xDxPPVe, job SRJ_AYWWLkNJLYCQ).

The judge failed the anniversary scenario because the agent noticed the
occasion but never offered to SET ANYTHING UP - it pitched a suite and moved
on. This harness replays that transcript's first bad turn (the caller has just
volunteered the anniversary) against bare gemma-4-31b with the agent's real
system prompt and tool schemas, per the prompting-gemma skill method: baseline
must reproduce, one wording idea per variant, n>=10 per cell, client
concurrency 2 with paced request starts, flagged samples read manually.

Measured 2026-07-23 (offer-to-arrange rate, n=10 per cell):
  no-bullet   0/10  reproduces production: verbatim suite pitch, no offer
  overfit    10/10  but every reply is the bullet's scripted dinner-table line
  general    7/10   nominal - manual read: all 10 are still suite upsells,
                    the abstract action ("arrange something that fits") only
                    bolted "shall I move you?" onto the same room pitch
  menu       10/10  situation + concrete action menu; zero suite pitches,
                    also 10/10 on an adjacent ask (mom's 70th birthday
                    instead of the anniversary), correctly adapted
The menu wording shipped to instructions.py. Confirms the skill: the named
concrete action is load-bearing for gemma; abstract discipline is not.

Run:
  uv run --no-sync python examples/hotel_receptionist/ab_occasion_offer_test.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / ".claude/skills/prompting-gemma"))

from gemma_cluster_health import install

install()  # lk default-project credentials + gemma cluster health watcher

from agent import HotelReceptionistAgent  # noqa: E402
from instructions import build_instructions  # noqa: E402

from livekit.agents import inference  # noqa: E402
from livekit.agents.llm import ChatContext  # noqa: E402

N_PER_CELL = 10
CONCURRENCY = 2
STAGGER_SECONDS = 0.5

# The shipped wording: situation + a menu of concrete bookable actions.
MENU_BULLET = (
    "- Caller volunteers a special occasion in passing: after handling what they "
    "called about, offer to set up ONE thing on this call that fits the occasion - "
    "a dinner table at the on-site restaurant, flowers to the room, or a spa visit - "
    "and book it when they take it. Offer to arrange it, not just describe it. "
    "Pressure-free; drop it the moment they decline."
)

# The first fix attempt: scenario-specific (scripted dinner-table line,
# occasion list, upgrade prohibition). Worked 10/10 but overfitted to the eval.
OVERFIT_BULLET = (
    "- Caller volunteers a special occasion in passing (an anniversary, birthday, "
    "honeymoon, celebration): after handling what they actually called about, make "
    "ONE warm offer of something concrete you can set up on this call that fits the "
    'occasion - most naturally a dinner table at the on-site restaurant ("would you '
    'like me to reserve a table for the two of you tonight?") - and offer to book it '
    "right then. Offer to arrange, don't just suggest: naming a nice option without "
    "offering to set it up is the miss. Keep it benefit-first and pressure-free, drop "
    "it gracefully the moment they decline, and don't pitch room changes or upgrades "
    "to a guest already staying here."
)

# The fully-generalized rewrite: same situation trigger, abstract action.
# Failed the manual read - gemma kept the suite pitch and only added "shall I?".
GENERAL_BULLET = (
    "- Caller volunteers a special occasion in passing: after handling what they "
    "actually called about, make ONE warm, pressure-free offer to arrange something "
    "on this call that genuinely fits what they shared - and offer to set it up "
    "right then, not just mention that it exists. The offer has to grow out of what "
    "the caller said, not out of what you'd like to sell; drop it gracefully the "
    "moment they decline."
)

_BASE = build_instructions()
assert MENU_BULLET in _BASE, "instructions.py no longer contains the menu bullet"

CELLS = [
    ("no-bullet", _BASE.replace(MENU_BULLET + "\n", "")),
    ("overfit", _BASE.replace(MENU_BULLET, OVERFIT_BULLET)),
    ("general", _BASE.replace(MENU_BULLET, GENERAL_BULLET)),
    ("menu", _BASE),
]

# Transcript prefix at the first bad turn, verbatim from the failing job. For
# the adjacent-ask check, swap the last user line for e.g. "It's my mom's
# seventieth birthday this weekend, that's the whole reason we came".
HISTORY: list[tuple[str, str]] = [
    ("assistant", "Hello, thanks for calling The LiveKit Hotel; how can I help you?"),
    ("user", "Hi, quick question — what time does breakfast run until?"),
    ("assistant", "Breakfast is served from six thirty to ten thirty in the morning."),
    (
        "user",
        "Great, thank you — that helps. We’re actually here celebrating our wedding "
        "anniversary, so I was just checking timing before we make plans for the morning.",
    ),
]

TOOLS = HotelReceptionistAgent().tools

# An "offer to arrange": first-person offer language attached to an action the
# agent can take on this call. Regex is a flag, not a verdict - flagged rows
# get read manually (the "general" cell scores 7/10 here but fails the read).
_OFFER_RE = re.compile(
    r"shall i|would you like (?:me|us) to|want me to|like me to "
    r"(?:book|reserve|arrange|set)|i can (?:book|reserve|arrange|set)|"
    r"happy to (?:book|reserve|arrange|set)|can i (?:book|reserve|arrange|set)",
    re.I,
)


def classify(text: str, tool_calls: list[str]) -> str:
    if tool_calls:
        return "TOOL"
    if not text.strip():
        return "EMPTY"
    return "offer" if _OFFER_RE.search(text) else "NO-OFFER"


async def sample(llm: inference.LLM, instructions_text: str) -> tuple[str, list[str]]:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=instructions_text)
    for role, text in HISTORY:
        chat_ctx.add_message(role=role, content=text)
    reply, tool_calls = "", []
    async with llm.chat(chat_ctx=chat_ctx, tools=TOOLS) as stream:
        async for chunk in stream:
            delta = chunk.delta
            if delta is None:
                continue
            if delta.content:
                reply += delta.content
            for call in delta.tool_calls or []:
                if call.name:
                    tool_calls.append(call.name)
    return reply.strip(), tool_calls


async def run_cell(llm: inference.LLM, name: str, instructions_text: str) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: list[tuple[int, str, str]] = []

    async def one(i: int) -> None:
        await asyncio.sleep(i * STAGGER_SECONDS)  # pace request starts
        async with semaphore:
            reply, tool_calls = await sample(llm, instructions_text)
        verdict = classify(reply, tool_calls)
        label = reply if not tool_calls else f"{reply!r} + tools {tool_calls}"
        results.append((i, label, verdict))

    await asyncio.gather(*(one(i) for i in range(N_PER_CELL)))
    offers = 0
    for i, label, verdict in sorted(results):
        print(f"  [{i}] {verdict:8s} {label!r}", flush=True)
        offers += verdict == "offer"
    return offers, len(results)


async def main() -> None:
    llm = inference.LLM("google/gemma-4-31b-it")
    summary = []
    try:
        for name, instructions_text in CELLS:
            print(f"\n=== cell: {name} (n={N_PER_CELL}) ===", flush=True)
            offers, n = await run_cell(llm, name, instructions_text)
            summary.append((name, offers, n))
    finally:
        await llm.aclose()

    print("\n=== summary (offer-to-arrange rate) ===")
    for name, offers, n in summary:
        print(f"  {name:10s} {offers}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
