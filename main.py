# main.py
"""
Example integration of InterruptHandler into a LiveKit-like agent.
Adapt the event registration to your real agent API.

This example includes:
 - A simple AgentStub if a real LiveKit Agent isn't available.
 - Wiring of transcription events and agent speaking start/stop.
 - CLI simulation mode to test behaviors interactively.
"""

import asyncio
import os
import logging
from livekit_interrupt_handler import InterruptHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_example")


class AgentStub:
    """
    Minimal stub that simulates an agent with:
      - start/stop speaking events
      - simple stop_speaking() API used by the handler
      - a naive event registry (on)
    Use your real LiveKit Agent in production.
    """

    def __init__(self):
        self._listeners = {"start_speaking": [], "stop_speaking": [], "transcription": []}
        self.speaking = False

    def on(self, event_name, callback):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    async def emit(self, event_name, *args, **kwargs):
        for cb in self._listeners.get(event_name, []):
            try:
                res = cb(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.exception("listener raised")

    async def start_tts(self, text="Hello, I'm speaking"):
        self.speaking = True
        await self.emit("start_speaking")
        logger.info("AgentStub TTS started: %s", text)

    async def stop_speaking(self, info=None):
        if self.speaking:
            logger.info("AgentStub received stop_speaking request. Info: %s", info)
            self.speaking = False
            await self.emit("stop_speaking")
        else:
            logger.info("AgentStub stop_speaking called, but agent wasn't speaking.")


async def cli_simulation():
    agent = AgentStub()
    handler = InterruptHandler(agent=agent)

    # Register events
    agent.on("start_speaking", lambda *a, **k: asyncio.create_task(handler.on_agent_speaking()))
    agent.on("stop_speaking", lambda *a, **k: asyncio.create_task(handler.on_agent_speaking_stop()))
    agent.on("transcription", lambda event: asyncio.create_task(handler.on_transcription(event)))

    # Simple menu-driven CLI to simulate events
    print("Simulation CLI - commands:")
    print("  tts_start               -> agent starts speaking")
    print("  tts_stop                -> agent stops speaking (simulated)")
    print("  asr <text> <conf>       -> simulate ASR/transcription event")
    print("  stats                   -> print handler stats")
    print("  exit                    -> quit")

    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, input, "> ")
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        if cmd == "tts_start":
            await agent.start_tts()
        elif cmd == "tts_stop":
            await agent.stop_speaking()
        elif cmd == "asr":
            if len(parts) < 2:
                print("Usage: asr <text> [confidence]")
                continue
            conf = float(parts[-1]) if len(parts) > 2 else 1.0
            text = " ".join(parts[1:-1]) if len(parts) > 2 else parts[1]
            event = {"text": text, "confidence": conf}
            # Fire transcription event
            await agent.emit("transcription", event)
        elif cmd == "stats":
            print(handler.stats())
        elif cmd == "exit":
            break
        else:
            print("unknown command")

if __name__ == "__main__":
    asyncio.run(cli_simulation())
