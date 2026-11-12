"""Classify the caller's mood while GPT-Live handles the conversation.

Run with LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET set:
    python gpt_live_decisions.py console
The serving gateway must support Decisions inside GPT-Live sessions.
"""

import logging
from typing import cast

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.inference.realtime import DecisionEventEnvelope, DuplexSession

load_dotenv()
logger = logging.getLogger("decisions-agent")
server = AgentServer()


class MoodAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Have a friendly, concise conversation with the caller.")

    async def on_enter(self) -> None:
        live = cast(DuplexSession, self.duplex_session)

        @live.on("decision_event")
        def on_decision(envelope: DecisionEventEnvelope) -> None:
            if envelope.event.get("type") == "decision.completed":
                # Answers may contain customer information; handle them in the application.
                logger.info("Mood evaluation completed: %s", envelope.event_id)

        self.session.generate_reply(instructions="Greet the caller.")


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=inference.DuplexModel(
            "openai/gpt-live-1",
            extra_kwargs={
                "triggers": [
                    {
                        "type": "decisions",
                        "event_type": "session.input_transcript.delta",
                        "decisions": {
                            "model": "openai/gpt-6-luna-decisions",
                            "questions": [
                                {
                                    "type": "choice",
                                    "name": "mood",
                                    "instructions": "What is the caller's mood?",
                                    "choices": [{"value": "happy"}, {"value": "sad"}],
                                }
                            ],
                        },
                    }
                ]
            },
        )
    )
    await session.start(agent=MoodAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
