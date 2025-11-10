"""
Example of integrating with a real LiveKit AgentSession.
Run locally once you have credentials in .env.
"""

import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import cli, WorkerOptions, JobContext
from livekit.agents.voice import AgentSession
from ..src import IHConfig, SpeechGate, InterruptOrchestrator
from ..src.logkit import make_logger

load_dotenv()
log = make_logger("run-worker")

async def entrypoint(ctx: JobContext):
    session = AgentSession()
    gate = SpeechGate()
    cfg = IHConfig.from_env()
    orchestrator = InterruptOrchestrator(session, gate, cfg)

    @session.on("agent_speech_started")
    def _tts_start(*_): gate.open()

    @session.on("agent_speech_ended")
    def _tts_end(*_): gate.close()

    @session.on("transcription")
    async def _on_transcription(text: str, confidence: float, **_):
        await orchestrator.on_transcription(text, confidence)

    # greet user
    session.generate_reply(instructions="say hello in English")
    log.info("Agent ready")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
