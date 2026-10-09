#!/usr/bin/env python3
"""
Example: Voice Agent with Krisp Turn Taking and Interruption Prediction

Krisp TT decides when the user has finished their turn; Krisp IP decides whether
user speech that overlaps the agent is a genuine interruption or a backchannel
("uh-huh", "yeah"). Every verdict is logged so the behavior can be observed live.

Prerequisites:
    1. Standard agent env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.
       STT/LLM/TTS run on LiveKit Inference with these credentials, so they are
       needed in ``console`` mode too.
    2. Krisp license mode: the ``krisp_audio`` wheel, and
       KRISP_VIVA_SDK_LICENSE_KEY (may be empty for non-licensing wheels),
       KRISP_VIVA_TT_MODEL_PATH, KRISP_VIVA_IP_MODEL_PATH, KRISP_VIVA_VAD_MODEL_PATH

Usage:
    python krisp_tt_ip_example.py console   # local mic/speaker, use headphones
    python krisp_tt_ip_example.py dev       # join a LiveKit room
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
)
from livekit.plugins import krisp

logger = logging.getLogger("krisp-tt-ip-example")
load_dotenv()


class KrispAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. When asked, tell long stories so the "
                "user can try interrupting you. Do not use emojis or special characters."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=inference.VAD(),
        stt=inference.STT("deepgram/nova-3", language="en"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_handling={
            "turn_detection": krisp.TurnDetector(),
            # keep the turn open briefly after TT says "done", so resuming after a short
            # mid-sentence pause (or a slow final transcript) doesn't split the turn
            "endpointing": {"min_delay": 0.5},
            # 0.35 instead of the 0.5 default: unrecognized speech counts as an interruption
            # after ~1 s of overlap instead of ~1.5 s (0.3 also stopped the agent on "yeah")
            "interruption": {"mode": krisp.InterruptionDetector(threshold=0.35)},
        },
    )

    @session.on("overlapping_speech")
    def _on_overlap(ev) -> None:  # type: ignore[no-untyped-def]
        verdict = "INTERRUPTION" if ev.is_interruption else "backchannel"
        if ev.agent_ended:
            verdict = "agent ended (no verdict)"
        logger.info(
            "IP: %s (p=%.2f, delay=%.0f ms)",
            verdict,
            ev.probability,
            ev.detection_delay * 1000,
        )

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if ev.is_final:
            logger.info("user said: %s", ev.transcript)

    await session.start(agent=KrispAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
