import asyncio
from livekit_interrupt_handler import InterruptHandler

async def test_filler_ignored():
    h = InterruptHandler()
    await h.set_agent_speaking(True)
    r = await h.on_transcription({"text": "uh", "confidence": 0.9})
    assert r["action"] == "ignore"

async def test_command_interrupts():
    h = InterruptHandler()
    await h.set_agent_speaking(True)
    r = await h.on_transcription({"text": "stop", "confidence": 0.5})
    assert r["action"] == "interrupt"

async def test_register_if_silent():
    h = InterruptHandler()
    await h.set_agent_speaking(False)
    r = await h.on_transcription({"text": "umm", "confidence": 0.9})
    assert r["action"] == "register"

def test_runner(event_loop):
    event_loop.run_until_complete(test_filler_ignored())
    event_loop.run_until_complete(test_command_interrupts())
    event_loop.run_until_complete(test_register_if_silent())
