"""PII redaction between STT and the LLM (issue #6204).

The transcribed user turn passes through ``Agent.on_user_turn_completed``
before it reaches the LLM, so sensitive data can be scrubbed there — a
deterministic, code-level hook rather than a system-prompt instruction. The
redacted text is what the LLM receives *and* what is stored in the chat
history, so the PII never leaves your servers toward the model provider.

That guarantee has a precondition: **preemptive generation must be off**
(it is on by default). Preemptive generation issues a speculative LLM
request built from the raw STT transcript *before* ``on_user_turn_completed``
runs, which would transmit the unredacted text to the provider — and since
redaction then changes the committed message, the framework discards the
speculative reply and regenerates anyway, so speculation buys nothing for
redacted turns. The session below disables it explicitly. The same applies
to any other feature that consumes the raw transcript before the hook (e.g.
LLM-based keyterm detection): leave them off, or redact at the STT layer
instead if you need them.

The pattern engine below is intentionally simple regexes so it reads as a
pattern, not a dependency: swap ``redact_pii`` for Microsoft Presidio, GLiNER,
or your own rules without touching the wiring.

The rules are deliberately safety-first: any sequence of 13 or more digits is
redacted (as ``<CARD_NUMBER>`` when it passes the Luhn checksum, ``<NUMBER>``
otherwise), with no upper length cap — a card number merged with neighboring
digits fails the checksum and would slip past a length-capped pattern while
still containing the card. Over-redacting an order id is the acceptable
failure mode for a privacy filter; leaking a card is not. Tune the rules if
your domain needs long non-sensitive numbers to survive.

Scope note: this redacts the LLM path. The raw transcript still exists inside
the process (e.g. ``user_input_transcribed`` events for UI display); redact at
those sinks too if your compliance boundary includes logs and storage.
"""

import logging
import re
from collections.abc import Callable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
)

logger = logging.getLogger("pii-redaction")

load_dotenv()


def _luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_long_number(match: re.Match) -> str:
    digits = re.sub(r"[ .-]", "", match.group())
    # Safety-first: every sequence of 13 or more digits is redacted — no upper
    # bound, so a card concatenated with adjacent digits (an expiry, another
    # number) is caught as one blob instead of slipping past a length cap. The
    # Luhn check only refines the label; it must NOT gate redaction, because a
    # merged sequence fails the checksum as a whole while still containing the
    # real card. Over-redacting an order id is the acceptable failure mode for
    # a privacy filter; leaking a card is not.
    return "<CARD_NUMBER>" if _luhn_valid(digits) else "<NUMBER>"


# (pattern, replacement) pairs — replacement may be a string or a callable
_PII_RULES: list[tuple[re.Pattern, str | Callable[[re.Match], str]]] = [
    (re.compile(r"\b\d(?:[ .-]?\d){12,}\b"), _redact_long_number),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    # phone numbers must contain separators (or a + prefix) so short plain
    # digit runs are left alone; the word boundary sits inside the optional
    # "(" because \b cannot assert between a space and a parenthesis
    (re.compile(r"(?:\+\d{1,3}[ .-]?)?\(?\b\d{2,4}\b\)?[ .-]\d{3,4}[ .-]\d{2,4}\b"), "<PHONE>"),
    (re.compile(r"\+\d{7,15}\b"), "<PHONE>"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return the redacted text and the list of rule replacements applied.

    All rules match against the original text in a single pass over claimed
    spans (earlier rules take priority on overlap), then replacements are
    applied together — so a later rule can never half-match inside a region an
    earlier rule already handled.
    """
    claimed: list[tuple[int, int, str]] = []  # (start, end, replacement)
    for pattern, replacement in _PII_RULES:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end, _ in claimed):
                continue  # an earlier (higher-priority) rule already claimed this span
            out = replacement(match) if callable(replacement) else replacement
            claimed.append((match.start(), match.end(), out))

    applied: list[str] = []
    for start, end, out in sorted(claimed, reverse=True):
        if out == text[start:end]:
            continue  # claimed but deliberately left as-is (e.g. Luhn-invalid number)
        applied.insert(0, out)
        text = text[:start] + out + text[end:]
    return text, applied


class RedactingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Redacted placeholders like "
            "<CARD_NUMBER>, <NUMBER>, <SSN>, <EMAIL>, or <PHONE> in the user's message "
            "mean sensitive data was removed for privacy; never ask the user to repeat it."
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        # runs after STT finalizes the turn and before the LLM sees it
        applied_rules: list[str] = []
        content: list = []
        for item in new_message.content:
            if isinstance(item, str):
                redacted, applied = redact_pii(item)
                content.append(redacted)
                applied_rules.extend(applied)
            else:
                content.append(item)
        new_message.content = content
        if applied_rules:
            # log categories only — never the original values
            logger.info("redacted PII from user turn", extra={"rules": applied_rules})


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        # REQUIRED for the privacy boundary: preemptive generation (on by
        # default) sends a speculative LLM request built from the raw STT
        # transcript before on_user_turn_completed runs - the unredacted
        # text would reach the model provider despite the redaction hook.
        turn_handling=TurnHandlingOptions(preemptive_generation={"enabled": False}),
    )
    await session.start(agent=RedactingAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
