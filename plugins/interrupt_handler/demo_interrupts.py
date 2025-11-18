from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from . import InterruptHandler, InterruptHandlerConfig


@dataclass
class DemoSession:
    """Minimal stub that mimics the interrupt() API."""

    stopped: bool = False

    def interrupt(self) -> None:
        self.stopped = True


async def main() -> None:
    session = DemoSession()
    handler = InterruptHandler(
        stop_callback=session.interrupt,
        config=InterruptHandlerConfig(),
    )

    async def simulate_event(
        transcript: str, *, is_final: bool, tts_speaking: bool, confidence: float = 0.92
    ) -> None:
        handler.on_tts_state(tts_speaking)
        await handler.on_transcription(
            transcript,
            words_meta=[{"text": word, "confidence": confidence} for word in transcript.split()],
            metadata={"is_final": is_final},
        )

    print("=== Interrupt handler demo ===")
    await simulate_event("uh", is_final=False, tts_speaking=True, confidence=0.7)
    print(f"Agent interrupted? {session.stopped}")

    await simulate_event("umm okay stop", is_final=True, tts_speaking=True)
    print(f"Agent interrupted? {session.stopped}")

    handler.on_tts_state(False)
    await handler.on_transcription("umm", metadata={"is_final": True})
    print("Passthrough while agent quiet complete.")


if __name__ == "__main__":
    asyncio.run(main())

