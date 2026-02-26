"""Example: Summarizing context during agent handoffs.

This example demonstrates three strategies for passing context between agents
during a handoff:

1. **Structured userdata** - Store key facts in a typed dataclass and serialize
   it (e.g. as YAML) so the next agent can read a compact snapshot.
2. **Chat context copy / truncate** - Carry the previous agent's recent
   conversation history into the new agent so it has conversational continuity.
3. **LLM-powered summarization** - Use the LLM to compress older conversation
   turns into a short summary before handing off, keeping token usage low
   while preserving important details.

Run with:
    python handoff_context_summarization.py dev
"""

import logging
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("handoff-context-summarization")
logger.setLevel(logging.INFO)

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Structured userdata - a typed container for facts gathered so far
# ---------------------------------------------------------------------------
@dataclass
class ConversationData:
    """Shared state passed between agents via RunContext.userdata."""

    topic: str | None = None
    customer_name: str | None = None
    customer_email: str | None = None
    sentiment: str | None = None
    key_requirements: list[str] | None = None

    def summarize(self) -> str:
        """Serialize collected data into a compact YAML string.

        YAML tends to produce fewer tokens than JSON while remaining easy for
        LLMs to parse.  This summary is injected as a system message when the
        next agent starts so it immediately has the full picture.
        """
        data = {
            "topic": self.topic or "unknown",
            "customer_name": self.customer_name or "unknown",
            "customer_email": self.customer_email or "unknown",
            "sentiment": self.sentiment or "unknown",
            "key_requirements": self.key_requirements or [],
        }
        return yaml.dump(data)


# ---------------------------------------------------------------------------
# 2. Base agent with chat context merging on handoff
# ---------------------------------------------------------------------------
class BaseAgent(Agent):
    """Base agent that merges the previous agent's recent chat history.

    On enter, it:
    - copies a truncated view of the previous agent's chat context
      (excluding system instructions and handoff markers)
    - appends a system message with the serialized userdata summary
    - triggers an initial reply so the new agent smoothly picks up
    """

    async def on_enter(self) -> None:
        userdata: ConversationData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # Merge last few turns from the previous agent so conversational
        # continuity is preserved without blowing up token usage.
        prev_agent = getattr(self.session, "_prev_agent", None)
        if isinstance(prev_agent, Agent):
            prev_ctx = prev_agent.chat_ctx.copy(
                exclude_instructions=True,  # don't carry over old system prompt
                exclude_function_call=False,  # keep tool calls for context
                exclude_handoff=True,  # strip handoff markers
                exclude_config_update=True,
            ).truncate(max_items=6)  # keep only the last ~3 turns (user+assistant)

            # de-duplicate by item id to avoid repeating messages already present
            existing_ids = {item.id for item in chat_ctx.items}
            for item in prev_ctx.items:
                if item.id not in existing_ids:
                    chat_ctx.items.append(item)

        # Inject a system message with the structured data summary so
        # the agent knows everything collected so far.
        chat_ctx.add_message(
            role="system",
            content=(
                f"You are the {self.__class__.__name__} agent. "
                f"Here is the current state of the conversation:\n{userdata.summarize()}"
            ),
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")


# ---------------------------------------------------------------------------
# 3. LLM-powered summarization before handoff
# ---------------------------------------------------------------------------
class TriageAgent(BaseAgent):
    """First agent the user talks to. Gathers initial info, then hands off."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly triage agent. Your job is to:\n"
                "1. Greet the user and learn their name.\n"
                "2. Understand what topic they need help with.\n"
                "3. Gauge their sentiment (happy, neutral, frustrated).\n"
                "Once you have this info, call the `transfer_to_specialist` tool."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def update_customer_info(
        self,
        context: RunContext[ConversationData],
        name: str,
        email: str,
    ):
        """Store the customer's name and email.

        Args:
            name: The customer's name.
            email: The customer's email address.
        """
        context.userdata.customer_name = name
        context.userdata.customer_email = email
        return f"Stored customer info: {name} <{email}>"

    @function_tool
    async def transfer_to_specialist(
        self,
        context: RunContext[ConversationData],
        topic: str,
        sentiment: str,
    ):
        """Hand the conversation to a specialist once triage is complete.

        Args:
            topic: The topic the user needs help with (e.g. billing, technical, general).
            sentiment: The user's current sentiment (happy, neutral, frustrated).
        """
        context.userdata.topic = topic
        context.userdata.sentiment = sentiment

        # --- Strategy 3: LLM-powered summarization ---
        # Before handing off, compress the chat history so the specialist
        # gets a concise summary rather than the full transcript.
        # _summarize keeps the last `keep_last_turns` user/assistant pairs
        # verbatim and compresses everything older into a short paragraph.
        llm_instance = self.session.llm
        if llm_instance is not None:
            logger.info("Summarizing conversation before handoff...")
            chat_ctx = self.chat_ctx.copy()
            await chat_ctx._summarize(llm_instance, keep_last_turns=2)
            await self.update_chat_ctx(chat_ctx)
            logger.info("Summarization complete.")

        # Create the specialist and hand off.
        # The specialist's on_enter will merge the (now-summarized) context.
        specialist = SpecialistAgent(topic)
        # Store a reference so the next agent's on_enter can access our context.
        self.session._prev_agent = self  # type: ignore[attr-defined]
        return specialist


class SpecialistAgent(BaseAgent):
    """Specialist that picks up after triage with full context."""

    def __init__(self, topic: str) -> None:
        super().__init__(
            instructions=(
                f"You are a specialist in {topic}. "
                "The user has already been triaged. You have their collected info "
                "and a summary of the prior conversation in your context. "
                "Help them resolve their issue. When done, call `wrap_up`."
            ),
        )

    @function_tool
    async def record_requirements(
        self,
        context: RunContext[ConversationData],
        requirements: list[str],
    ):
        """Record the specific requirements the user mentioned.

        Args:
            requirements: A list of specific requirements or issues.
        """
        context.userdata.key_requirements = requirements
        return f"Recorded {len(requirements)} requirement(s)."

    @function_tool
    async def wrap_up(self, context: RunContext[ConversationData]):
        """Wrap up the conversation when the user's issue is resolved."""
        self.session.interrupt()
        name = context.userdata.customer_name or "there"
        await self.session.generate_reply(
            instructions=f"Say goodbye to {name} and let them know their issue is resolved.",
            allow_interruptions=False,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession[ConversationData](
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(voice="echo"),
        userdata=ConversationData(),
    )

    await session.start(
        agent=TriageAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
