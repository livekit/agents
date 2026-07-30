"""Voice agent fronting the deep-research background session.

Message flow (bdr = background deep research):

    bdr  -- ctx.send(clarifying questions / progress / findings) --> voice agent --> user
    user -- voice answer --> voice agent -- lk_background_send(id, content) --> bdr

The voice agent is a thin overlay: it relays the background session's
clarifying questions and progress updates to the user, and forwards the
user's answers, follow-up context, and mid-research direction changes back
through the generated ``lk_background_send`` tool. The research pipeline
itself lives in ``deep_research.py``.

Run with: uv run examples/voice_agents/deep-research/agent.py dev
"""

import logging

from deep_research import deep_research
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)

logger = logging.getLogger("deep-research-agent")

load_dotenv()


class ResearchAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice research assistant. You are the overlay between the user "
                "and a deep-research pipeline running in the background.\n"
                "- When the user asks a research question, send it to the deep_research "
                "background session and tell them the research is starting.\n"
                "- The background session may ask clarifying questions before starting. "
                "Relay them naturally, one at a time if there are several, collect the "
                "user's answers, and send them back to the deep_research session.\n"
                "- Relay progress updates conversationally and briefly.\n"
                "- If the user adds context, narrows scope, or changes direction while the "
                "research is running, forward it to the deep_research session immediately.\n"
                "- Keep responses concise; no markdown or special characters."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and ask what they'd like researched."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        background=[deep_research],
    )
    await session.start(agent=ResearchAssistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
