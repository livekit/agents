import asyncio
import pytest
from ..src import IHConfig, SpeechGate, InterruptOrchestrator

class DummySession:
    def __init__(self): self.interrupted = False
    async def interrupt(self): self.interrupted = True

@pytest.mark.asyncio
async def test_interrupt_and_ignore():
    cfg = IHConfig()
    gate = SpeechGate()
    sess = DummySession()
    orch = InterruptOrchestrator(sess, gate, cfg)

    # agent speaking: filler ignored
    gate.open()
    res1 = await orch.on_transcription("umm hmm", confidence=0.9)
    assert res1 == "IGNORE"
    assert not sess.interrupted

    # agent speaking: hard intent triggers interrupt
    res2 = await orch.on_transcription("uh stop", confidence=0.9)
    assert res2 == "INTERRUPT"
    assert sess.interrupted

    # agent quiet: everything passes through
    gate.close()
    res3 = await orch.on_transcription("umm", confidence=0.9)
    assert res3 == "PASS"
