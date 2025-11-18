import asyncio
import pytest
from interrupt_handler import InterruptHandler

@pytest.mark.asyncio
async def test_ignore_filler_while_speaking():
    ih = InterruptHandler(ignored_words=["uh","umm"], confidence_threshold=0.3)
    results = {"valid": [], "ignored": []}
    ih.on_valid_interrupt = lambda t: results["valid"].append(t)
    ih.on_ignored_filler = lambda t: results["ignored"].append(t)
    await ih.on_agent_state_change(True)
    await ih.on_transcription_event("umm", confidence=0.9)
    assert results["ignored"] == ["umm"]
    assert results["valid"] == []

@pytest.mark.asyncio
async def test_stop_command_stops():
    ih = InterruptHandler(ignored_words=["uh"], confidence_threshold=0.3)
    results = {"valid": []}
    ih.on_valid_interrupt = lambda t: results["valid"].append(t)
    await ih.on_agent_state_change(True)
    await ih.on_transcription_event("umm wait", confidence=0.9)
    assert any("wait" in t for t in results["valid"])

@pytest.mark.asyncio
async def test_low_confidence_ignored():
    ih = InterruptHandler(ignored_words=["uh"], confidence_threshold=0.5)
    results = {"valid": [], "ignored": []}
    ih.on_valid_interrupt = lambda t: results["valid"].append(t)
    ih.on_ignored_filler = lambda t: results["ignored"].append(t)
    await ih.on_agent_state_change(True)
    await ih.on_transcription_event("hmm", confidence=0.4)
    assert results["ignored"] == [] and results["valid"] == []
