from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from .. import inference
from ..llm import LLM, ChatContext
from ..log import logger

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .events import ConversationItemAddedEvent

InjectRole = Literal["system", "assistant"]


@dataclass
class Guardrail:
    """Conversation guardrail configuration.

    A guardrail monitors conversations between an AI agent and a user,
    detecting specific situations and providing guidance to the agent
    without adding latency to the main conversation.

    Example:
        ```python
        session = AgentSession(
            llm="openai/gpt-4o",
            guardrails=[
                Guardrail(
                    name="compliance",
                    llm="openai/gpt-4o-mini",
                    instructions="Watch for pricing claims without disclaimers",
                ),
                Guardrail(
                    name="sentiment",
                    llm="openai/gpt-4o-mini",
                    instructions="Flag frustrated customers",
                    eval_interval=1,
                ),
            ],
        )
        ```
    """

    instructions: str
    """What to watch for and what advice to give to the agent."""

    llm: str | LLM
    """LLM for evaluation."""

    name: str | None = None
    """Optional name for logging. Example: "compliance", "sentiment"."""

    eval_interval: int = 3
    """Evaluate every N user turns. Default: 3"""

    max_interventions: int = 5
    """Maximum advice messages per session. Default: 5"""

    cooldown: float = 10.0
    """Minimum seconds between advice messages. Default: 10.0"""

    inject_role: InjectRole = "system"
    """Role for injected advice messages. Default: "system" """

    inject_prefix: str = "[GUARDRAIL ADVISOR]:"
    """Prefix for injected advice messages. Default: "[GUARDRAIL ADVISOR]:" """

    max_history: int | None = None
    """Max messages to include in evaluation. None = all history. Default: None"""


@dataclass
class _GuardrailState:
    turn_count: int = 0
    interventions_count: int = 0
    last_intervention_time: float = 0.0


GUARDRAIL_SYSTEM_PROMPT = """You are a conversation guardrail - a silent background monitor that observes conversations between an AI agent and a user.

YOUR ROLE:
- You watch conversations for specific issues
- When you detect a problem, you send advice to the agent
- The agent receives your advice as a system message
- You are invisible to the user - they don't know you exist
- You are NOT the agent - you don't talk to the user directly

WHEN TO INTERVENE:
- Only when you detect something from your instructions below
- Don't intervene if everything is fine
- When in doubt, intervene - false positives are better than missed issues

RESPONSE FORMAT:
If intervention needed, return JSON:
{{"intervene": true, "advice": "what agent should say or ask next (1-2 sentences)"}}

If no intervention needed:
{{"intervene": false}}

Return ONLY valid JSON. No markdown, no explanation.

YOUR MONITORING INSTRUCTIONS:
{user_instructions}
"""


class _GuardrailRunner:
    """Runs guardrail evaluation in the background."""

    def __init__(self, config: Guardrail, session: AgentSession) -> None:
        self._config = config
        self._session = session
        self._state = _GuardrailState()
        self._eval_lock = asyncio.Lock()
        self._started = False
        self._event_handler: Callable | None = None
        self._pending_tasks: set[asyncio.Task] = set()

        self._llm: LLM
        if isinstance(config.llm, str):
            self._llm = inference.LLM.from_model_string(config.llm)
        else:
            self._llm = config.llm

    @property
    def _log_prefix(self) -> str:
        if self._config.name:
            return f"guardrail[{self._config.name}]"
        return "guardrail"

    def start(self) -> None:
        if self._started:
            return

        def _on_conversation_item(event: ConversationItemAddedEvent) -> None:
            self._on_item(event)

        self._event_handler = _on_conversation_item
        self._session.on("conversation_item_added", _on_conversation_item)
        self._started = True
        logger.debug(f"{self._log_prefix}: started")

    def stop(self) -> None:
        if not self._started:
            return

        if self._event_handler is not None:
            self._session.off("conversation_item_added", self._event_handler)
            self._event_handler = None

        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()

        self._started = False
        logger.debug(f"{self._log_prefix}: stopped")

    def _on_item(self, event: ConversationItemAddedEvent) -> None:
        if not self._started:
            return

        item = event.item
        if not hasattr(item, "role"):
            return

        if item.role == "assistant":
            self._state.turn_count += 1
            if self._state.turn_count % self._config.eval_interval == 0:
                task = asyncio.create_task(self._evaluate(), name="guardrail_evaluate")
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)

    def _build_transcript(self) -> str:
        agent = self._session._agent
        if agent is None:
            return ""

        items = agent.chat_ctx.items
        messages = []
        first_system_skipped = False

        for item in items:
            if item.type != "message":
                continue
            content = item.text_content
            if not content:
                continue

            if item.role == "system":
                if not first_system_skipped:
                    first_system_skipped = True
                    continue
                if self._config.inject_prefix and content.startswith(self._config.inject_prefix):
                    messages.append(f"[PREVIOUS ADVICE]: {content}")
            elif item.role in ("user", "assistant"):
                messages.append(f"[{item.role.upper()}]: {content}")

        if self._config.max_history and len(messages) > self._config.max_history:
            messages = messages[-self._config.max_history :]

        return "\n".join(messages)

    async def _evaluate(self) -> None:
        if self._eval_lock.locked():
            return

        transcript = self._build_transcript()
        if not transcript:
            return

        async with self._eval_lock:
            try:
                system_prompt = GUARDRAIL_SYSTEM_PROMPT.format(
                    user_instructions=self._config.instructions
                )

                ctx = ChatContext()
                ctx.add_message(role="system", content=system_prompt)
                ctx.add_message(role="user", content=f"Analyze this conversation:\n\n{transcript}")

                response = ""
                async with self._llm.chat(chat_ctx=ctx) as stream:
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            response += chunk.delta.content

                result = self._parse_json(response)
                if result and result.get("intervene"):
                    advice = result.get("advice", "").strip()
                    if advice:
                        await self._maybe_inject(advice)

            except Exception:
                logger.exception(f"{self._log_prefix}: evaluation failed")

    async def _maybe_inject(self, advice: str) -> None:
        now = time.time()

        if self._state.interventions_count >= self._config.max_interventions:
            logger.debug(f"{self._log_prefix}: max interventions reached, skipping")
            return

        time_since_last = now - self._state.last_intervention_time
        if self._state.last_intervention_time > 0 and time_since_last < self._config.cooldown:
            logger.debug(f"{self._log_prefix}: cooldown active, skipping")
            return

        await self._inject(advice)
        self._state.interventions_count += 1
        self._state.last_intervention_time = now
        logger.info(
            f"{self._log_prefix}: intervention #{self._state.interventions_count} - {advice}"
        )

    async def _inject(self, advice: str) -> None:
        agent = self._session._agent
        if agent is None:
            logger.warning(f"{self._log_prefix}: no active agent")
            return

        prefix = self._config.inject_prefix
        content = f"{prefix} {advice}" if prefix else advice

        ctx = agent.chat_ctx.copy()
        ctx.add_message(role=self._config.inject_role, content=content)
        await agent.update_chat_ctx(ctx)

    def _parse_json(self, text: str) -> dict | None:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                if isinstance(result, dict):
                    return result
        except json.JSONDecodeError:
            logger.warning(f"{self._log_prefix}: failed to parse JSON: {text[:100]}")
        return None
