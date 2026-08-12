from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from livekit.agents import llm
from livekit.agents.llm import function_tool

if TYPE_CHECKING:
    from common import Userdata

logger = logging.getLogger("hotel-receptionist.end-call-check")

# The pre-hangup policy audit: one LLM call that re-reads the receptionist's own
# standing policy against the transcript and names at most ONE concrete action the
# policy still requires before the line closes. The verdict comes back as a forced
# tool call (may_end / missing), not free text, so there's nothing to parse. High
# precision by construction - no verdict call, an empty action, errors, and slow
# responses all fail OPEN (the call may end): a caller stuck on a line that won't
# hang up is worse than a missed offer.
_AUDIT_SYSTEM_PROMPT = """\
You audit a hotel receptionist's phone call at the moment the receptionist is about \
to say goodbye. You are given the receptionist's standing policy and the call \
transcript. Decide whether the policy EXPLICITLY requires one more concrete action \
or offer that has not yet happened on this call.

Respond with exactly one tool call: missing, only when the policy clearly requires \
the action for THIS call and the transcript shows it never happened; otherwise \
may_end. When in doubt, call may_end."""

_NUDGE_TEMPLATE = (
    "Do NOT say goodbye or close the call yet - your standing policy still requires "
    "one thing on this call: {missing} Handle that with the caller now, in your own "
    "words, then call this tool again once they're done."
)


def _render_transcript(chat_ctx: llm.ChatContext) -> str:
    """Caller/Agent turns plus tool names - no system prompt, no tool outputs."""
    lines: list[str] = []
    for item in chat_ctx.items:
        if item.type == "message" and item.role in ("user", "assistant"):
            text = item.text_content
            if text:
                speaker = "Caller" if item.role == "user" else "Agent"
                lines.append(f"{speaker}: {text}")
        elif item.type == "function_call":
            lines.append(f"[tool] {item.name}")
    return "\n".join(lines)


async def find_missing_action(
    llm_v: llm.LLM,
    *,
    instructions: str,
    chat_ctx: llm.ChatContext,
    timeout: float = 10.0,
) -> str | None:
    """Audit the call against the policy; return the one missing action, or None."""
    audit_ctx = llm.ChatContext.empty()
    audit_ctx.add_message(role="system", content=_AUDIT_SYSTEM_PROMPT)
    audit_ctx.add_message(
        role="user",
        content=(
            f"RECEPTIONIST POLICY:\n{instructions}\n\n"
            f"CALL TRANSCRIPT:\n{_render_transcript(chat_ctx)}"
        ),
    )

    verdict: str | None = None

    async def may_end() -> None:
        """The policy requires nothing further; the call may end now."""

    async def missing(action: str) -> None:
        """The policy still requires one concrete action or offer on this call.

        Args:
            action: One short imperative instruction for the receptionist.
        """
        nonlocal verdict
        verdict = action.strip() or None

    stream = llm_v.chat(
        chat_ctx=audit_ctx,
        tools=[function_tool(may_end), function_tool(missing)],
        tool_choice="required",
    )
    try:
        response = await asyncio.wait_for(stream.collect(), timeout=timeout)
        # the prompt asks for exactly one call, so extras are ignored;
        # execute_function_call never raises, so an unusable verdict leaves
        # `verdict` unset and the call ends
        if response.tool_calls:
            await llm.execute_function_call(response.tool_calls[0], llm.ToolContext(stream.tools))
    except Exception:
        logger.exception("end-call policy audit failed; allowing the call to end")
        return None

    return verdict


async def run_goodbye_gate(
    userdata: Userdata,
    llm_v: llm.LLM | llm.RealtimeModel | None,
    *,
    instructions: str,
    chat_ctx: llm.ChatContext,
) -> str | None:
    """One-shot gate in front of the goodbye: the nudge instruction, or None to close.

    At most one nudge per call - once given, every later attempt closes
    unconditionally, so the agent can never get stuck unable to hang up.
    """
    if userdata.end_call_nudged:
        return None
    if not isinstance(llm_v, llm.LLM):
        # realtime model or no LLM: skip the audit rather than block the goodbye
        return None
    missing = await find_missing_action(llm_v, instructions=instructions, chat_ctx=chat_ctx)
    if missing is None:
        return None
    userdata.end_call_nudged = True
    logger.info("end-call audit nudge: %s", missing)
    return _NUDGE_TEMPLATE.format(missing=missing)
