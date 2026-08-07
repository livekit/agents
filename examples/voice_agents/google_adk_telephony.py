"""Example: Google ADK agent with telephony infrastructure.

The LiveKit agent acts as a generic telephony middleware layer providing four
capabilities. The ADK agent drives the conversation — the LiveKit side makes
no assumptions about the call flow.

Capabilities provided by the LiveKit agent:

1. **DTMF Input** — Keypad digits are collected with a debounce and forwarded
   to the ADK bot as user input.  The bot decides what to do with them.
2. **No User Input** — When the caller goes silent, the bot is notified so it
   can re-prompt however it sees fit.
3. **Bot Response Timeout** — If the ADK bot is slow to respond, a filler
   prompt is played so the caller isn't left in silence.
4. **Custom Payload** — JSON data messages can flow in/out of the room for
   cross-bot orchestration.  Inbound payloads are forwarded to the ADK bot.

Dependencies:
    pip install livekit-agents[silero,deepgram,turn_detector] \
                livekit-plugins-google-adk google-adk
"""

import asyncio
import json
import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    UserStateChangedEvent,
    cli,
    inference,
)
from livekit.plugins import google_adk, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("adk-telephony")

server = AgentServer()

# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

# ---------------------------------------------------------------------------
# ADK agent — this is where the call flow lives.
# The LiveKit side knows nothing about what the bot wants to do.
# ---------------------------------------------------------------------------


def verify_account(account_id: str) -> dict:
    """Verify an account by its ID and return status."""
    return {"account_id": account_id, "status": "active", "name": "Jane Doe"}


def transfer_to_department(department: str) -> dict:
    """Transfer the call to the specified department."""
    return {"transferred": True, "department": department}


adk_agent = LlmAgent(
    name="telephony_assistant",
    model="gemini-2.0-flash",
    instruction=(
        "You are a telephone customer-service agent. Keep responses concise "
        "since your output will be spoken aloud via TTS.\n\n"
        "The system will forward DTMF keypad digits to you as user messages "
        "prefixed with [DTMF]. You decide how to interpret them.\n\n"
        "If you receive a [NO_INPUT] message it means the caller has been "
        "silent. Decide whether to re-prompt, ask a simpler question, or "
        "offer to transfer them.\n\n"
        "If you receive an [EVENT] message it contains a JSON payload from "
        "an external system. Act on it if relevant.\n\n"
        "You have access to verify_account and transfer_to_department tools."
    ),
    tools=[verify_account, transfer_to_department],
)

# ---------------------------------------------------------------------------
# Infrastructure configuration — tune these, not the bot logic
# ---------------------------------------------------------------------------

RESPONSE_TIMEOUT = 8.0  # seconds before filler prompt plays
USER_AWAY_TIMEOUT = 12.0  # seconds of silence before notifying the bot
DTMF_DEBOUNCE = 3.0  # seconds after last digit before forwarding to bot
MAX_INACTIVITY_PINGS = 3  # times we notify the bot before hanging up

FILLER_PROMPTS = [
    "One moment please while I look that up.",
    "Bear with me, I'm still working on that.",
    "Almost there, just a moment.",
]

# ---------------------------------------------------------------------------
# Entrypoint — pure infrastructure, no call-flow assumptions
# ---------------------------------------------------------------------------


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    llm_adapter = google_adk.LLMAdapter(
        adk_agent,
        response_timeout=RESPONSE_TIMEOUT,
    )

    agent = Agent(instructions="", llm=llm_adapter)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT("deepgram/nova-3", language="multi"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        user_away_timeout=USER_AWAY_TIMEOUT,
    )

    # ------------------------------------------------------------------
    # 1. DTMF — accumulate digits with debounce, forward to bot
    # ------------------------------------------------------------------
    dtmf_buffer: list[str] = []
    dtmf_timer: asyncio.TimerHandle | None = None

    def _flush_dtmf() -> None:
        if not dtmf_buffer:
            return
        digits = " ".join(dtmf_buffer)
        dtmf_buffer.clear()
        logger.info("Forwarding DTMF digits to bot: %s", digits)
        session.generate_reply(user_input=f"[DTMF] {digits}")

    @ctx.room.on("sip_dtmf_received")
    def _on_dtmf(ev: rtc.SipDTMF) -> None:
        nonlocal dtmf_timer
        dtmf_buffer.append(ev.digit)
        logger.debug("DTMF digit received: %s (buffer: %s)", ev.digit, dtmf_buffer)
        if dtmf_timer is not None:
            dtmf_timer.cancel()
        loop = asyncio.get_event_loop()
        dtmf_timer = loop.call_later(DTMF_DEBOUNCE, _flush_dtmf)

    # ------------------------------------------------------------------
    # 2. No User Input — notify the bot, let it decide how to respond
    # ------------------------------------------------------------------
    inactivity_task: asyncio.Task[None] | None = None

    async def _handle_inactivity() -> None:
        for attempt in range(1, MAX_INACTIVITY_PINGS + 1):
            logger.info("User silent — notifying bot (%d/%d)", attempt, MAX_INACTIVITY_PINGS)
            await session.generate_reply(
                user_input=f"[NO_INPUT] The caller has been silent for "
                f"{USER_AWAY_TIMEOUT}s (attempt {attempt}/{MAX_INACTIVITY_PINGS}).",
            )
            await asyncio.sleep(USER_AWAY_TIMEOUT)

        logger.info("Max inactivity pings reached — shutting down")
        session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        nonlocal inactivity_task
        if ev.new_state == "away":
            inactivity_task = asyncio.create_task(_handle_inactivity())
            return
        if inactivity_task is not None:
            inactivity_task.cancel()
            inactivity_task = None

    # ------------------------------------------------------------------
    # 3. Bot Response Timeout — play filler when ADK is slow
    # ------------------------------------------------------------------
    filler_index = 0

    @session.on("error")
    def _on_error(ev) -> None:
        nonlocal filler_index
        inner = getattr(ev.error, "error", ev.error)
        if "no response within" in str(inner):
            filler = FILLER_PROMPTS[filler_index % len(FILLER_PROMPTS)]
            filler_index += 1
            logger.warning("Bot timeout — playing filler: %s", filler)
            session.say(filler, add_to_chat_ctx=False)

    # ------------------------------------------------------------------
    # 4. Custom Payload — bidirectional room data messages
    # ------------------------------------------------------------------
    @ctx.room.on("data_received")
    def _on_data_received(data_packet: rtc.DataPacket) -> None:
        try:
            payload = json.loads(data_packet.data.decode())
            logger.info("Received custom payload: %s", payload)
            session.generate_reply(
                user_input=f"[EVENT] {json.dumps(payload)}",
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Received non-JSON data packet, ignoring")

    # ------------------------------------------------------------------
    # Start — the ADK bot's instructions determine the greeting and flow
    # ------------------------------------------------------------------
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions="A new caller has connected. Greet them and begin.",
    )

    # Publish session-started event for other room participants
    await ctx.room.local_participant.publish_data(
        payload=json.dumps(
            {
                "event": "session_started",
                "room": ctx.room.name,
                "agent": "telephony_assistant",
            }
        ).encode(),
        topic="orchestration",
    )


if __name__ == "__main__":
    cli.run_app(server)
