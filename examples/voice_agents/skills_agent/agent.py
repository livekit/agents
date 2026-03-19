"""Example: SkillSelector for search-driven skill discovery.

The agent starts with just a ``tool_search`` function. When the user asks about
weather or calendar topics, the LLM searches for matching skills. Matched
skills' tools are loaded and their instructions are provided inline.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.beta.skills import SkillRegistry, SkillSelector
from livekit.plugins import silero

logger = logging.getLogger("skills-agent")
logger.setLevel(logging.INFO)

load_dotenv()

SKILLS_DIR = Path(__file__).parent / "skills"


class SkillsAgent(Agent):
    def __init__(self, selector: SkillSelector) -> None:
        super().__init__(
            instructions=_INSTRUCTIONS,
            tools=[selector],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and let them know you can help with "
            "weather and calendar management. Tell them to just ask naturally."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    registry = SkillRegistry.from_directory(SKILLS_DIR)
    logger.info(
        "Loaded %d skills: %s",
        len(registry.available_skills),
        list(registry.available_skills.keys()),
    )

    selector = SkillSelector(skills=registry)

    session = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        stt=inference.STT("deepgram/nova-3"),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=SkillsAgent(selector),
        room=ctx.room,
    )


_INSTRUCTIONS = """
You are a helpful voice assistant with access to specialized skills. You can
discover and load the right skills by searching for them with tool_search.

When a user asks about something, use tool_search to find relevant tools.
For example, if they ask about the weather, search for "weather". If they
want to schedule something, search for "calendar".

Keep your responses short and natural — this is a voice conversation.
"""

if __name__ == "__main__":
    cli.run_app(server)
