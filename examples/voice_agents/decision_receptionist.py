"""Receptionist with Jev-driven callback requests, mocked in the console.

Run: lk agent console examples/voice_agents/decision_receptionist.py
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import NOT_GIVEN, Agent, AgentServer, AgentSession, JobContext, cli, decisions
from livekit.plugins import openai, typesafe

load_dotenv()

logger = logging.getLogger("decision-receptionist")


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the receptionist at Maple House restaurant. Help with reservation "
                "and billing questions. Ask for the date, time, and party size for a booking. "
                "You can request a callback from a staff member. If the caller wants a "
                "person, offer a callback instead of a live transfer. When they request "
                "or accept a callback, acknowledge their request briefly. Requests are "
                "processed in the background. Do not say a request has been recorded until "
                "you receive explicit confirmation. Do not collect contact details, promise a "
                "callback time, or claim to book a table. Keep spoken replies brief."
            ),
            decisions={
                "wants_callback": decisions.Probability(
                    "The caller has explicitly requested or accepted a staff callback and "
                    "has not withdrawn that request. Use the conversation context. Frustration "
                    "alone, asking for a live transfer, or an assistant offer without the caller's "
                    "acceptance do not count. A declined or withdrawn request is false."
                ),
                "intent": decisions.Choice(
                    "What is the caller's current main request?",
                    options={
                        "booking": "Make or change a restaurant reservation.",
                        "billing": "Resolve a charge or billing question.",
                        "other": "Any other request, including a staff callback.",
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
            },
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the caller and offer to help.")


def log_callback_desire(*, source_message_id: str, probability: float) -> None:
    logger.info(
        "MOCK callback request logged for this call (message=%s, probability=%.2f)",
        source_message_id,
        probability,
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

    callback_logged = False

    @session.on("decisions_completed")
    def on_decisions(ev: decisions.DecisionsCompletedEvent) -> None:
        nonlocal callback_logged
        logger.info(
            "Decisions for %s: %s",
            ev.source_message_id,
            {name: result.value for name, result in ev.results.items()},
        )
        if callback_logged:
            return

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

        result = ev.results["wants_callback"]
        if result.kind == "probability" and result.value >= 0.9:
            log_callback_desire(source_message_id=ev.source_message_id, probability=result.value)
            callback_logged = True
            session.generate_reply(
                instructions="The callback request was successfully recorded. "
                "Briefly confirm this to the caller."
            )

    await session.start(agent=Receptionist(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
