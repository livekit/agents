# tests/test_interrupt_handler.py
import asyncio
import pytest
from livekit_interrupt_handler import InterruptHandler

@pytest.mark.asyncio
async def test_ignore_filler_when_agent_speaking():
    handler = InterruptHandler(agent=None, ignored_words=["uh","umm"], confidence_threshold=0.4)
    await handler.on_agent_speaking()
    # filler only -> should be ignored
    await handler.on_transcription({"text":"uh", "tokens":["uh"], "confidence":0.9})
    stats = handler.stats()
    assert stats["ignored"] == 1
    assert stats["valid"] == 0

@pytest.mark.asyncio
async def test_valid_interrupt_when_agent_speaking_contains_command():
    handler = InterruptHandler(agent=None, ignored_words=["uh","umm"], command_words=["stop"])
    await handler.on_agent_speaking()
    await handler.on_transcription({"text":"uh stop", "tokens":["uh","stop"], "confidence":0.9})
    stats = handler.stats()
    assert stats["ignored"] == 0
    assert stats["valid"] == 1

@pytest.mark.asyncio
async def test_register_when_agent_not_speaking():
    handler = InterruptHandler(agent=None, ignored_words=["uh","umm"])
    # Agent not speaking -> should be valid
    await handler.on_transcription({"text":"uh", "tokens":["uh"], "confidence":0.9})
    stats = handler.stats()
    assert stats["ignored"] == 0
    assert stats["valid"] == 1

@pytest.mark.asyncio
async def test_confidence_low_ignored():
    handler = InterruptHandler(agent=None, ignored_words=["uh","umm"], confidence_threshold=0.8)
    await handler.on_agent_speaking()
    # low confidence -> ignored
    await handler.on_transcription({"text":"hmm yeah", "tokens":["hmm","yeah"], "confidence":0.5})
    stats = handler.stats()
    assert stats["ignored"] == 1
