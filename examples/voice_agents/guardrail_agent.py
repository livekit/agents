"""
Guardrail example: Restaurant reservation with safety monitoring.

WHY GUARDRAILS?
The main agent prompt can be huge (10k+ tokens) with complex business logic.
Guardrails are small, focused monitors that run in parallel - they don't add
latency to the main conversation but catch critical issues.

Here: main agent handles full reservation flow, guardrail watches for ONE
safety-critical thing: dietary restrictions/allergies.

Run: python examples/voice_agents/guardrail_agent.py console
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, room_io
from livekit.agents.voice import Guardrail
from livekit.plugins import silero

logger = logging.getLogger("guardrail-agent")

load_dotenv()


class RestaurantAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the host at "The Golden Fork" fine dining restaurant.

RESERVATION FLOW:
1. Greet warmly, ask how you can help
2. Get party size and preferred date/time
3. Check availability (always say "let me check... yes we have that available")
4. Get name and contact number for confirmation
5. Mention dress code (smart casual, no sportswear)
6. Explain our 15-minute late policy
7. Offer to note any special requests
8. Confirm all details and thank them

UPSELLING (be subtle):
- Parties of 6+ → suggest private dining room ($50 extra)
- Weekend bookings → mention live jazz on Saturday nights
- If they mention celebration → offer complimentary dessert

ALSO HANDLE:
- Cancellations (need 24hr notice)
- Modifications to existing bookings
- Questions about menu, parking, accessibility
- Directions to the restaurant

Keep responses SHORT (1-2 sentences). Sound professional but warm.
Don't use emojis or markdown.""",
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the caller warmly and ask how you can help today."
        )


server = AgentServer()


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-2",
        vad=ctx.proc.userdata["vad"],
        guardrails=[
            Guardrail(
                name="safety",
                instructions="""You monitor restaurant reservations for FOOD SAFETY.

CRITICAL: Before confirming ANY new reservation, agent MUST ask about
dietary restrictions and allergies. This is a safety requirement.

Intervene if:
- Agent is about to confirm without asking about allergies/dietary needs
- Agent collected name, date, time, party size but skipped dietary question

Do NOT intervene if:
- Customer already mentioned dietary needs ("I'm vegetarian", "nut allergy")
- Agent already asked about restrictions
- Customer is just asking questions, not making a reservation
- Customer is canceling or modifying existing booking""",
                llm="openai/gpt-4o-mini",
                eval_interval=3,
                max_interventions=2,
                cooldown=15.0,
            ),
        ],
    )

    await session.start(
        agent=RestaurantAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
