from __future__ import annotations

from ....log import logger
from ....voice import AgentSession


async def announce_execution(tool_name: str, message: str, session: AgentSession) -> None:
    """Announce execution message via agent session."""
    try:
        await session.generate_reply(
            instructions=(
                f"You are running {tool_name} tool (do not announce the tool name) for user "
                f"and should announce it using this instruction: {message}"
            ),
            allow_interruptions=False,
        )
    except Exception as e:  # pragma: no cover - logging fallback
        logger.debug(
            "failed to announce execution for tool",
            extra={"tool_name": tool_name, "error": str(e)},
        )
