"""PII redaction between STT and the LLM (issue #6204).

The transcribed user turn passes through ``Agent.on_user_turn_completed``
before it reaches the LLM, so sensitive data can be scrubbed there — a
deterministic, code-level hook rather than a system-prompt instruction. The
redacted text is what the LLM receives *and* what is stored in the chat
history, so the PII never leaves your servers toward the model provider.

The pattern engine below is intentionally simple regexes so it reads as a
pattern, not a dependency: swap ``redact_pii`` for Microsoft Presidio, GLiNER,
or your own rules without touching the wiring.

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


def _redact_card(match: re.Match) -> str:
    digits = re.sub(r"[ -]", "", match.group())
    # Luhn-check to avoid eating arbitrary long numbers (order ids, tracking numbers)
    return "<CARD_NUMBER>" if _luhn_valid(digits) else match.group()


# (pattern, replacement) pairs — replacement may be a string or a callable
_PII_RULES: list[tuple[re.Pattern, str | Callable[[re.Match], str]]] = [
    (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), _redact_card),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    # phone numbers must contain separators (or a + prefix) so plain digit
    # runs like order ids are left alone
    (re.compile(r"(?:\+\d{1,3}[ .-])?\b\(?\d{2,4}\)?[ .-]\d{3,4}[ .-]\d{2,4}\b"), "<PHONE>"),
    (re.compile(r"\+\d{7,15}\b"), "<PHONE>"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return the redacted text and the list of rule replacements applied.

    Single pass over claimed spans: earlier rules claim the regions they
    matched even when they decline to change them, so a card-like number that
    fails the Luhn check is left fully intact instead of being partially
    re-matched (and half-redacted) by the phone rule that runs later.
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
            "<CARD_NUMBER> in the user's message mean sensitive data was removed for "
            "privacy; never ask the user to repeat it."
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
    )
    await session.start(agent=RedactingAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
