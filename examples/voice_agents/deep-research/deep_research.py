"""The deep-research background session.

A Python port of the Claude Code ``deep-research`` dynamic workflow
packaged as a reusable
``@background`` definition:

    Clarify -> Scope -> Search -> Fetch+Extract -> 3-vote adversarial Verify -> Synthesize

The exported ``deep_research`` definition is long-lived and multi-round: it
grills the user with clarifying questions before starting, accepts mid-run
steering that reshapes every later pipeline stage, writes a dated markdown
report next to this file after each round, and then keeps listening for
follow-up questions that start another round with the previous report as
context.

Every LLM call goes through LiveKit's ``llm.chat()`` API with
``tool_choice="required"`` and a save-tool, the same structured-output pattern
used by the AMD classifier.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from pydantic import BaseModel

from livekit.agents import BackgroundContext, background, inference
from livekit.agents.llm import (
    LLM,
    ChatContext,
    ChatMessage,
    Tool,
    ToolContext,
    function_tool,
)
from livekit.agents.llm.utils import execute_function_call

logger = logging.getLogger("deep-research")

RESEARCH_MODEL = "openai/gpt-5.5"

MAX_CLARIFY_ROUNDS = 3
CLARIFY_ANSWER_TIMEOUT = 180.0  # proceed with what we have if the user stays silent
MAX_ANGLES = 5
MAX_FETCH = 8
MAX_VERIFY_CLAIMS = 10
VOTES_PER_CLAIM = 3
REFUTATIONS_REQUIRED = 2
LLM_CONCURRENCY = 4
FETCH_TEXT_LIMIT = 8_000

# final report lands next to this file: <this dir>/YYYY-MM-DD-<slug>.md
REPORT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Structured output helper — the classifier.py pattern: force a tool call and
# capture its arguments through a closure.
# ---------------------------------------------------------------------------


async def _tool_call(llm_client: LLM, *, system: str, user: str, tools: list[Tool]) -> None:
    stream = llm_client.chat(
        chat_ctx=ChatContext(
            items=[
                ChatMessage(role="system", content=[system]),
                ChatMessage(role="user", content=[user]),
            ]
        ),
        tools=tools,
        tool_choice="required",
    )
    response = await stream.collect()
    for tool_call in response.tool_calls:
        await execute_function_call(tool_call, ToolContext(stream.tools))


# ---------------------------------------------------------------------------
# Web search + fetch. DuckDuckGo's HTML endpoint keeps the example key-free;
# swap in a real search API (Tavily, Brave, Exa, ...) for production use.
# ---------------------------------------------------------------------------


async def _search_web(http: aiohttp.ClientSession, query: str) -> list[dict[str, str]]:
    async with http.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (research example)"},
    ) as resp:
        page = await resp.text()

    results: list[dict[str, str]] = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page):
        url = unescape(m.group(1))
        if "uddg=" in url:  # unwrap DDG's redirect
            url = unquote(parse_qs(urlparse(url).query).get("uddg", [url])[0])
        title = unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        if url.startswith("http"):
            results.append({"url": url, "title": title})
        if len(results) >= 5:
            break
    return results


async def _fetch_page(http: aiohttp.ClientSession, url: str) -> str:
    async with http.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (research example)"},
        timeout=aiohttp.ClientTimeout(total=20),
    ) as resp:
        page = await resp.text(errors="replace")

    page = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", page)
    text = re.sub(r"<[^>]+>", " ", page)
    text = re.sub(r"\s+", " ", unescape(text)).strip()
    return text[:FETCH_TEXT_LIMIT]


def _norm_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or url
    host = host.removeprefix("www.")
    return (host + parsed.path.rstrip("/")).lower()


# ---------------------------------------------------------------------------
# Structured payloads
# ---------------------------------------------------------------------------


class SearchAngle(BaseModel):
    label: str
    query: str
    rationale: str


class ExtractedClaim(BaseModel):
    claim: str
    quote: str
    importance: Literal["central", "supporting", "tangential"]


class Finding(BaseModel):
    claim: str
    confidence: Literal["high", "medium", "low"]
    sources: list[str]
    evidence: str


@dataclass
class _SourcedClaim:
    claim: ExtractedClaim
    url: str
    quality: str
    refuted_votes: int = 0
    valid_votes: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def survives(self) -> bool:
        return (
            self.valid_votes >= REFUTATIONS_REQUIRED and self.refuted_votes < REFUTATIONS_REQUIRED
        )


@dataclass
class ResearchState:
    """Everything the user has told us. Re-read by every pipeline stage, so
    steering added mid-run shapes all subsequent LLM calls."""

    question: str = ""
    clarifications: list[str] = field(default_factory=list)  # "Q -> A" transcript
    steering: list[str] = field(default_factory=list)  # mid-run focus changes

    def context_block(self) -> str:
        parts = [f"## Research question\n{self.question}"]
        if self.clarifications:
            parts.append("## Clarifications from the user\n" + "\n".join(self.clarifications))
        if self.steering:
            parts.append(
                "## Mid-research direction changes from the user (most recent last; "
                "these OVERRIDE earlier scope where they conflict)\n"
                + "\n".join(f"- {s}" for s in self.steering)
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline stages — prompts adapted from the deep-research workflow script.
# ---------------------------------------------------------------------------


async def _clarify(llm_client: LLM, state: ResearchState) -> list[str] | None:
    """Grill-me step: returns clarifying questions, or None when ready to start."""
    questions: list[list[str]] = []
    ready: list[str] = []

    @function_tool
    async def ask_clarifying_questions(items: list[str]) -> None:
        """Ask the user clarifying questions before starting the research.

        Args:
            items: 2-4 sharp questions about scope, constraints, intended use,
                region/time bounds, or what a complete answer looks like.
        """
        questions.append(items)

    @function_tool
    async def start_research(refined_question: str) -> None:
        """Declare the question specific enough to research and start.

        Args:
            refined_question: The research question rewritten to fold in every
                clarification the user has provided.
        """
        ready.append(refined_question)

    await _tool_call(
        llm_client,
        system=(
            "You are the intake step of a deep research pipeline. Interrogate the request "
            "like a demanding reviewer: is it specific enough to research well? If key "
            "details are missing (audience, constraints, budget, region, timeframe, what "
            "decision this informs), ask. If it is already specific — or the user told you "
            "to just start — begin. Never ask about something the user already answered."
        ),
        user=state.context_block(),
        tools=[ask_clarifying_questions, start_research],
    )
    if ready:
        state.question = ready[0]
        return None
    return questions[0] if questions else None


async def _plan_angles(llm_client: LLM, state: ResearchState) -> list[SearchAngle]:
    plans: list[list[SearchAngle]] = []

    @function_tool
    async def save_research_plan(angles: list[SearchAngle]) -> None:
        """Save the research plan.

        Args:
            angles: 3-5 complementary web search angles that together cover the
                question (e.g. broad/primary, academic, recent news,
                contrarian/skeptical, practitioner). Queries must be specific
                enough to surface high-signal results; avoid redundancy.
        """
        plans.append(angles)

    await _tool_call(
        llm_client,
        system="Decompose the research question into complementary search angles.",
        user=state.context_block(),
        tools=[save_research_plan],
    )
    if not plans:
        raise RuntimeError("planner returned no angles")
    return plans[0][:MAX_ANGLES]


async def _extract_claims(
    llm_client: LLM, state: ResearchState, url: str, title: str, page_text: str
) -> tuple[str, list[ExtractedClaim]]:
    extractions: list[tuple[str, list[ExtractedClaim]]] = []

    @function_tool
    async def save_extraction(
        source_quality: Literal["primary", "secondary", "blog", "forum", "unreliable"],
        claims: list[ExtractedClaim],
    ) -> None:
        """Save claims extracted from a fetched source.

        Args:
            source_quality: primary research/institution, secondary reporting,
                blog/opinion, forum, or unreliable.
            claims: 0-4 FALSIFIABLE claims bearing on the research question.
                Each needs a direct supporting quote from the page and an
                importance rating. Empty if the page is irrelevant or paywalled.
        """
        extractions.append((source_quality, claims))

    await _tool_call(
        llm_client,
        system="Extract falsifiable, checkable claims from this source. No vague generalities.",
        user=(
            state.context_block()
            + f"\n\n## Source\nURL: {url}\nTitle: {title}\n\n## Page text\n{page_text}"
        ),
        tools=[save_extraction],
    )
    return extractions[0] if extractions else ("unreliable", [])


async def _verify_vote(llm_client: LLM, state: ResearchState, sc: _SourcedClaim) -> None:
    verdicts: list[tuple[bool, str]] = []

    @function_tool
    async def save_verdict(refuted: bool, evidence: str) -> None:
        """Save the adversarial verdict for the claim under review.

        Args:
            refuted: True if the claim is unsupported by its quote, contradicted,
                sourced too weakly for its strength, outdated, or marketing fluff.
                Default to True when uncertain.
            evidence: Specific reasoning for the verdict.
        """
        verdicts.append((refuted, evidence))

    await _tool_call(
        llm_client,
        system=(
            "You are an adversarial claim verifier. Be SKEPTICAL — try to REFUTE the claim. "
            "Check: does the quote actually support it? Is the source quality sufficient "
            "(extraordinary claims need primary sources)? Could it be outdated or cherry-picked?"
        ),
        user=(
            state.context_block()
            + f'\n\n## Claim under review\n"{sc.claim.claim}"\n\n'
            + f"Source: {sc.url} ({sc.quality})\n"
            + f'Supporting quote: "{sc.claim.quote}"'
        ),
        tools=[save_verdict],
    )
    if verdicts:
        refuted, evidence = verdicts[0]
        sc.valid_votes += 1
        if refuted:
            sc.refuted_votes += 1
        else:
            sc.evidence.append(evidence)


@dataclass
class ResearchReport:
    slug: str
    summary: str
    findings: list[Finding]
    caveats: str
    open_questions: list[str]


async def _synthesize(
    llm_client: LLM, state: ResearchState, confirmed: list[_SourcedClaim]
) -> ResearchReport:
    reports: list[ResearchReport] = []

    @function_tool
    async def save_report(
        slug: str,
        summary: str,
        findings: list[Finding],
        caveats: str,
        open_questions: list[str],
    ) -> None:
        """Save the final research report.

        Args:
            slug: Short kebab-case name for the report file, derived from the
                question (e.g. "eu-ev-charging-market").
            summary: 3-5 sentence executive summary answering the question.
            findings: Confirmed claims merged into coherent findings, each with
                confidence, source URLs, and evidence.
            caveats: What is uncertain, weakly sourced, or time-sensitive.
            open_questions: 2-4 questions that emerged but were not answered.
        """
        reports.append(ResearchReport(slug, summary, findings, caveats, open_questions))

    block = "\n\n".join(
        f"### [{i}] {sc.claim.claim}\n"
        f"Vote: {sc.valid_votes - sc.refuted_votes}-{sc.refuted_votes} · "
        f"Source: {sc.url} ({sc.quality})\n"
        f'Quote: "{sc.claim.quote}"\n'
        f"Verifier evidence: {sc.evidence[0] if sc.evidence else 'n/a'}"
        for i, sc in enumerate(confirmed)
    )
    await _tool_call(
        llm_client,
        system=(
            "Synthesize a research report from adversarially verified claims. Merge claims "
            "that say the same thing (combining their sources), group related claims into "
            "findings that directly address the question, and assign confidence per finding."
        ),
        user=state.context_block() + "\n\n## Confirmed claims\n" + block,
        tools=[save_report],
    )
    if not reports:
        raise RuntimeError("synthesis returned no report")
    return reports[0]


def _write_report_file(
    state: ResearchState,
    report: ResearchReport,
    confirmed: list[_SourcedClaim],
    refuted: list[_SourcedClaim],
) -> Path:
    """Render the full markdown report and write it next to this file."""
    today = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "-", report.slug.lower()).strip("-") or "research-report"
    path = REPORT_DIR / f"{today}-{slug}.md"

    lines = [
        f"# Research Report: {state.question}",
        f"**Date:** {today}",
        "",
        "## Executive Summary",
        report.summary,
        "",
        "## Key Findings",
    ]
    for i, f in enumerate(report.findings, 1):
        lines += [
            f"### {i}. {f.claim} [confidence: {f.confidence}]",
            f"**Evidence:** {f.evidence}",
            "**Sources:**",
            *(f"- {s}" for s in f.sources),
            "",
        ]
    if report.caveats:
        lines += ["## Caveats", report.caveats, ""]
    if report.open_questions:
        lines += ["## Open Questions", *(f"- {q}" for q in report.open_questions), ""]
    if state.steering:
        lines += [
            "## Mid-research direction changes",
            *(f"- {s}" for s in state.steering),
            "",
        ]
    if refuted:
        lines += [
            "## Refuted claims (for transparency)",
            *(
                f'- "{sc.claim.claim}" ({sc.url}, '
                f"vote {sc.valid_votes - sc.refuted_votes}-{sc.refuted_votes})"
                for sc in refuted
            ),
            "",
        ]
    lines += [
        "## Verified claim ledger",
        *(
            f"- [{sc.quality}] {sc.claim.claim} "
            f"(vote {sc.valid_votes - sc.refuted_votes}-{sc.refuted_votes}, {sc.url})"
            for sc in confirmed
        ),
    ]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# The background session
# ---------------------------------------------------------------------------


@background(name="deep_research")
async def deep_research(ctx: BackgroundContext) -> None:
    """Runs deep, multi-source, fact-checked web research. Send it the research
    question first; while it works you can send follow-up context, narrow or
    redirect the focus, and it will fold that into the remaining research. It
    may ask clarifying questions before starting — relay them to the user and
    send the answers back. It stays available after delivering a report: send
    follow-up questions or a narrower focus to start another research round."""
    llm_client = inference.LLM(RESEARCH_MODEL)
    inbox: asyncio.Queue[str] = asyncio.Queue()
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def _read_inbox() -> None:
        async for message in ctx.message_stream():
            inbox.put_nowait(message)

    reader = asyncio.create_task(_read_inbox())
    http = aiohttp.ClientSession()
    prior_context: str | None = None
    # the state shown while blocked on the inbox; end-of-round outcomes update it
    # so "done, last report at ..." stays visible until the next question arrives
    idle_state: dict[str, Any] = {"phase": "waiting for a research question"}
    try:
        while True:
            ctx.set_state(idle_state)
            state = ResearchState(question=await inbox.get())
            ctx.set_state({"phase": "clarifying scope", "topic": state.question})
            if prior_context:
                # follow-up round: give the clarify/plan LLM the previous result so
                # "narrow it to X" style requests need no re-clarification
                state.clarifications.append(
                    "Context: an earlier research round this session concluded: " + prior_context
                )

            # ── Clarify (grill-me): ask, wait for the voice answer, repeat ──
            for _ in range(MAX_CLARIFY_ROUNDS):
                questions = await _clarify(llm_client, state)
                if questions is None:
                    break
                await ctx.send(
                    "Before starting, I need a few details:\n"
                    + "\n".join(f"- {q}" for q in questions)
                )
                try:
                    answer = await asyncio.wait_for(inbox.get(), timeout=CLARIFY_ANSWER_TIMEOUT)
                except asyncio.TimeoutError:
                    break  # user went quiet — research with what we have
                state.clarifications.append("Q: " + " / ".join(questions) + f"\nA: {answer}")

            # ── From here on, inbox messages are steering, drained concurrently ──
            async def _drain_steering(state: ResearchState = state) -> None:
                while True:
                    note = await inbox.get()
                    state.steering.append(note)
                    await ctx.send(f"Noted — I'll fold this into the remaining research: {note}")

            steering_task = asyncio.create_task(_drain_steering())
            try:
                # ── Scope ──
                angles = await _plan_angles(llm_client, state)
                ctx.set_state(
                    {
                        "phase": "searching the web",
                        "topic": state.question,
                        "angles": [a.label for a in angles],
                    }
                )
                await ctx.send(
                    f"Starting deep research on: {state.question}\n"
                    f"Covering {len(angles)} angles: {', '.join(a.label for a in angles)}."
                )

                # ── Search (parallel, one searcher per angle) ──
                search_lists = await asyncio.gather(
                    *(_search_web(http, a.query) for a in angles),
                    return_exceptions=True,
                )
                seen: set[str] = set()
                sources: list[dict[str, str]] = []
                for result in search_lists:
                    if isinstance(result, BaseException):
                        logger.warning("search failed: %s", result)
                        continue
                    for r in result:
                        key = _norm_url(r["url"])
                        if key not in seen and len(sources) < MAX_FETCH:
                            seen.add(key)
                            sources.append(r)

                # ── Fetch + extract (parallel) ──
                async def _fetch_extract(
                    src: dict[str, str], state: ResearchState = state
                ) -> list[_SourcedClaim]:
                    try:
                        text = await _fetch_page(http, src["url"])
                        async with sem:
                            quality, claims = await _extract_claims(
                                llm_client, state, src["url"], src["title"], text
                            )
                        return [
                            _SourcedClaim(claim=c, url=src["url"], quality=quality) for c in claims
                        ]
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("fetch/extract failed for %s: %s", src["url"], exc)
                        return []

                extracted = await asyncio.gather(*(_fetch_extract(s) for s in sources))
                imp_rank = {"central": 0, "supporting": 1, "tangential": 2}
                qual_rank = {
                    "primary": 0,
                    "secondary": 1,
                    "blog": 2,
                    "forum": 3,
                    "unreliable": 4,
                }
                all_claims = sorted(
                    (sc for group in extracted for sc in group),
                    key=lambda sc: (
                        imp_rank[sc.claim.importance],
                        qual_rank[sc.quality],
                    ),
                )[:MAX_VERIFY_CLAIMS]
                ctx.set_state(
                    {
                        "phase": "adversarially verifying claims",
                        "topic": state.question,
                        "sources_read": len(sources),
                        "claims_under_review": len(all_claims),
                    }
                )
                await ctx.send(
                    f"Read {len(sources)} sources and extracted "
                    f"{sum(len(g) for g in extracted)} claims. "
                    f"Adversarially verifying the top {len(all_claims)} now."
                )

                if not all_claims:
                    await ctx.send(
                        "I couldn't extract any usable claims — the sources were empty, "
                        "paywalled, or irrelevant. Try rephrasing the question."
                    )
                    idle_state = {
                        "phase": "waiting — last research found no usable sources",
                        "last_topic": state.question,
                    }
                    continue  # keep listening for a rephrased question

                # ── Verify: N adversarial votes per claim ──
                async def _vote(sc: _SourcedClaim, state: ResearchState = state) -> None:
                    async with sem:
                        await _verify_vote(llm_client, state, sc)

                await asyncio.gather(
                    *(_vote(sc) for sc in all_claims for _ in range(VOTES_PER_CLAIM)),
                    return_exceptions=True,
                )
                confirmed = [sc for sc in all_claims if sc.survives]
                refuted = [sc for sc in all_claims if sc.refuted_votes >= REFUTATIONS_REQUIRED]
                await ctx.send(
                    f"Verification done: {len(confirmed)} claims confirmed, "
                    f"{len(refuted)} refuted, "
                    f"{len(all_claims) - len(confirmed) - len(refuted)} inconclusive."
                )

                if not confirmed:
                    await ctx.send(
                        "No claims survived adversarial verification — the research is "
                        "inconclusive. The sources may be low quality or the claims overstated."
                    )
                    idle_state = {
                        "phase": "waiting — last research was inconclusive",
                        "last_topic": state.question,
                    }
                    continue  # keep listening for a refined question

                # ── Synthesize + write the report to disk ──
                ctx.set_state({"phase": "writing the report", "topic": state.question})
                report = await _synthesize(llm_client, state, confirmed)
                report_path = await asyncio.to_thread(
                    _write_report_file, state, report, confirmed, refuted
                )
                lines = [f"Research complete. {report.summary}", "", "Key findings:"]
                lines += [
                    f"- ({f.confidence} confidence) {f.claim} [{', '.join(f.sources)}]"
                    for f in report.findings
                ]
                if report.caveats:
                    lines += ["", f"Caveats: {report.caveats}"]
                if report.open_questions:
                    lines += ["", "Open questions: " + "; ".join(report.open_questions)]
                lines += [
                    "",
                    f"The full report was saved to {report_path}.",
                    "I'm still listening — send a follow-up question or a narrower "
                    "focus to start another round.",
                ]
                await ctx.send("\n".join(lines))
                prior_context = report.summary
                idle_state = {
                    "phase": "done, listening for follow-ups",
                    "last_topic": state.question,
                    "last_report": str(report_path),
                }
            finally:
                steering_task.cancel()
    finally:
        reader.cancel()
        await http.close()
        await llm_client.aclose()
