import asyncio
from livekit_interrupt_handler import InterruptHandler

async def scenario_mixed():
    h = InterruptHandler()
    await h.set_agent_speaking(True)
    r1 = await h.on_transcription({"text":"umm okay stop","confidence":0.95})
    assert r1["action"] == "interrupt"
    await h.set_agent_speaking(False)
    r2 = await h.on_transcription({"text":"umm","confidence":0.98})
    assert r2["action"] == "register"

def test_runner(event_loop):
    event_loop.run_until_complete(scenario_mixed())
