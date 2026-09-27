"""Receptionist with background Jev decisions and a simulated human handoff.

Run: lk agent console examples/voice_agents/decision_receptionist.py
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import NOT_GIVEN, Agent, AgentServer, AgentSession, JobContext, cli, decisions
from livekit.plugins import openai, typesafe

load_dotenv()

logger = logging.getLogger("decision-receptionist")

CALLER_DECISIONS: dict[str, decisions.Decision] = {
    "intent": decisions.Choice(
        "What is the caller's current main request?",
        options={
            "booking": "Make or change a restaurant reservation.",
            "billing": "Resolve a charge or billing question.",
            "other": "Any other request, including asking for a person.",
        },
    ),
    "frustration": decisions.Score(
        "How frustrated is the caller in their latest turn?",
        levels=[
            "Calm; expresses no frustration.",
            "Expresses annoyance with the situation.",
            "Expresses strong anger or repeated complaints.",
        ],
    ),
}


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the receptionist at Maple House restaurant. Help with reservation "
                "and billing questions. Ask for the date, time, and party size for a booking. "
                "This is a demo: do not claim to book a table or transfer a call. If asked "
                "for a person, acknowledge the request. Keep spoken replies brief."
            ),
            decisions={
                "wants_human": decisions.Probability(
                    "The caller currently wants to speak to a human representative. "
                    "Use the latest user turn in context; a withdrawn request is false."
                ),
                **CALLER_DECISIONS,
            },
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the caller and offer to help.")


class HandoffRequested(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "The caller requested a human. This demo cannot transfer calls. "
                "Acknowledge their request and do not resume collecting reservation details."
            ),
            decisions=CALLER_DECISIONS,
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "I've noted your request to speak to a person. "
            "This demo cannot connect you to a real staff member."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    speech_enabled = bool(
        os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY"))
        and os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET"))
    )
    session: AgentSession = AgentSession(
        llm=openai.LLM.with_openrouter(model="openai/gpt-4.1-mini"),
        stt="deepgram/nova-3" if speech_enabled else NOT_GIVEN,
        tts=(
            "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc" if speech_enabled else NOT_GIVEN
        ),
        vad=NOT_GIVEN if speech_enabled else None,
        turn_handling={"turn_detection": "vad" if speech_enabled else None},
        decision_model=typesafe.Jev.with_openrouter(),
        decision_options={"turn_interval": 1, "max_context_turns": 6, "timeout": 5.0},
    )
    if not speech_enabled:
        session.output.set_audio_enabled(False)
        logger.info("Speech is disabled. Use --text, or configure LiveKit inference credentials.")

    @session.on("decisions_completed")
    def on_decisions(ev: decisions.DecisionsCompletedEvent) -> None:
        logger.info(
            "Decisions for %s: %s",
            ev.source_message_id,
            {name: result.value for name, result in ev.results.items()},
        )
        latest_user = next(
            (
                item
                for item in reversed(session.history.items)
                if item.type == "message" and item.role == "user"
            ),
            None,
        )
        if latest_user is None or latest_user.id != ev.source_message_id:
            return

        result = ev.results.get("wants_human")
        if result is not None and result.kind == "probability" and result.value >= 0.9:
            logger.info("Demo handoff requested (probability %.2f)", result.value)
            session.interrupt()
            session.update_agent(HandoffRequested())

    await session.start(agent=Receptionist(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
