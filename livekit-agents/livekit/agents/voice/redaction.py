from __future__ import annotations

import asyncio
import enum
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..llm import ChatContext
from ..log import logger

REDACTION_FAILURE_MARKER = "[REDACTED]"
"""Replacement applied to a text item when the redactor fails or times out (fail-closed)."""


class RedactionSink(enum.Enum):
    """A trust-boundary egress where text leaves the agent process."""

    LLM = "llm"
    """Chat context sent to a (typically third-party) LLM."""
    TELEMETRY = "telemetry"
    """Chat content written to telemetry spans/logs. Not wired yet."""
    TRANSCRIPT = "transcript"
    """Chat history serialized or uploaded at session end. Not wired yet."""


@dataclass
class RedactedEntity:
    """Audit metadata for a single redacted span.

    Carries only the entity type and the span in the *original* text —
    never the raw value.
    """

    type: str
    start: int
    end: int


@dataclass
class RedactionContext:
    """Context passed to a redactor so it can apply per-boundary policy."""

    sink: RedactionSink
    role: str | None = None
    """Role of the chat item being redacted ("user", "assistant", "system", "tool")."""


@dataclass
class RedactionResult:
    text: str
    entities: list[RedactedEntity] = field(default_factory=list)


@runtime_checkable
class Redactor(Protocol):
    async def redact(self, text: str, ctx: RedactionContext) -> RedactionResult: ...


@dataclass
class RedactionOptions:
    """Opt-in redaction configuration for an ``AgentSession``.

    Redaction is applied to a copy of the content at each enabled sink
    (redact-on-egress): the in-memory chat history keeps the raw values.
    """

    redactor: Redactor
    sinks: set[RedactionSink] = field(default_factory=lambda: {RedactionSink.LLM})
    timeout: float = 3.0
    """Max seconds a single redact() call may take before failing closed."""


class NoopRedactor:
    """Redactor that returns text unchanged."""

    async def redact(self, text: str, ctx: RedactionContext) -> RedactionResult:
        return RedactionResult(text=text)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_valid(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _ssn_valid(area: str, group: str, serial: str) -> bool:
    if area in ("000", "666") or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


class RegexRedactor:
    """Best-effort redaction of structured PII using validated pattern matching.

    Detects credit card numbers (Luhn-checked), US SSNs (format-checked), and
    IBANs (mod-97-checked), replacing each with a ``[TYPE]`` placeholder. The
    validators keep false positives low on what it claims to catch, but this is
    pattern matching, not a compliance guarantee: it makes no attempt to detect
    free-form PII such as names, addresses, or organizations. Layer a stronger
    detector (e.g. Presidio) or vendor-side redaction for higher recall.
    """

    _CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
    _SSN_RE = re.compile(r"\b(\d{3})-(\d{2})-(\d{4})\b")
    _IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

    async def redact(self, text: str, ctx: RedactionContext) -> RedactionResult:
        spans: list[RedactedEntity] = []

        for m in self._CREDIT_CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                spans.append(RedactedEntity(type="CREDIT_CARD", start=m.start(), end=m.end()))

        for m in self._SSN_RE.finditer(text):
            if _ssn_valid(m.group(1), m.group(2), m.group(3)):
                spans.append(RedactedEntity(type="US_SSN", start=m.start(), end=m.end()))

        for m in self._IBAN_RE.finditer(text):
            if _iban_valid(m.group()):
                spans.append(RedactedEntity(type="IBAN", start=m.start(), end=m.end()))

        spans.sort(key=lambda e: e.start)
        entities: list[RedactedEntity] = []
        pieces: list[str] = []
        cursor = 0
        for entity in spans:
            if entity.start < cursor:  # overlaps a previous match
                continue
            pieces.append(text[cursor : entity.start])
            pieces.append(f"[{entity.type}]")
            cursor = entity.end
            entities.append(entity)
        pieces.append(text[cursor:])

        return RedactionResult(text="".join(pieces), entities=entities)


async def _redact_text(text: str, opts: RedactionOptions, ctx: RedactionContext) -> str:
    try:
        result = await asyncio.wait_for(opts.redactor.redact(text, ctx), timeout=opts.timeout)
        return result.text
    except Exception:
        # fail closed: never let raw text through on a redactor failure
        logger.exception(
            "redactor failed, replacing text with redaction marker",
            extra={"sink": ctx.sink.value},
        )
        return REDACTION_FAILURE_MARKER


async def redact_chat_ctx(
    chat_ctx: ChatContext, opts: RedactionOptions, *, sink: RedactionSink
) -> ChatContext:
    """Return a redacted copy of ``chat_ctx``; the input context and its items are untouched.

    Redacts the text content of messages and function call outputs. If the
    redactor raises or exceeds ``opts.timeout``, the affected text is replaced
    with :data:`REDACTION_FAILURE_MARKER`.
    """
    items = []
    for item in chat_ctx.items:
        if item.type == "message":
            ctx = RedactionContext(sink=sink, role=item.role)
            new_content = [
                await _redact_text(c, opts, ctx) if isinstance(c, str) else c for c in item.content
            ]
            if new_content != item.content:
                item = item.model_copy(update={"content": new_content})
        elif item.type == "function_call_output":
            ctx = RedactionContext(sink=sink, role="tool")
            new_output = await _redact_text(item.output, opts, ctx)
            if new_output != item.output:
                item = item.model_copy(update={"output": new_output})

        items.append(item)

    return ChatContext(items)
