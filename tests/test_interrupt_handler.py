import asyncio
from interruption_handler.interrupt_handler import InterruptHandler, TranscriptEvent

def test_fillers_ignored_when_speaking():
    handler = InterruptHandler()
    ev = TranscriptEvent("uh umm", 0.9)
    act, reason, _ = asyncio.get_event_loop().run_until_complete(
        handler.on_transcript_event(True, ev)
    )
    assert act == "ignore"

def test_keyword_interrupt():
    handler = InterruptHandler()
    ev = TranscriptEvent("umm okay stop", 0.9)
    act, reason, _ = asyncio.get_event_loop().run_until_complete(
        handler.on_transcript_event(True, ev)
    )
    assert act == "interrupt"

def test_low_confidence_murmur():
    handler = InterruptHandler()
    ev = TranscriptEvent("hmm yeah", 0.2)
    act, reason, _ = asyncio.get_event_loop().run_until_complete(
        handler.on_transcript_event(True, ev)
    )
    assert act == "ignore"

def test_agent_quiet_registers_speech():
    handler = InterruptHandler()
    ev = TranscriptEvent("umm", 0.9)
    act, reason, _ = asyncio.get_event_loop().run_until_complete(
        handler.on_transcript_event(False, ev)
    )
    assert act == "register_speech"
