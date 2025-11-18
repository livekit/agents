import asyncio
import pytest

from voice_interrupt.handler import InterruptHandler

class MockSession:
    def __init__(self):
        self.callbacks = {}
        self.interrupted = 0
    def on(self, name, cb):
        self.callbacks[name] = cb
    def interrupt(self):
        self.interrupted += 1

@pytest.mark.asyncio
async def test_filler_ignored_when_speaking():
    mock = MockSession()
    handler = InterruptHandler(mock, ignored_words={"uh","umm"}, stop_words={"stop"})
    handler.start()
    await handler._on_agent_state_changed({"new_state":"speaking"})
    await handler._on_user_input_transcribed({"is_final": True, "transcript":"uh", "confidence":1.0})
    assert mock.interrupted == 0

@pytest.mark.asyncio
async def test_stop_triggers_interrupt_when_speaking():
    mock = MockSession()
    handler = InterruptHandler(mock, ignored_words={"uh","umm"}, stop_words={"stop"})
    handler.start()
    await handler._on_agent_state_changed({"new_state":"speaking"})
    await handler._on_user_input_transcribed({"is_final": True, "transcript":"stop", "confidence":1.0})
    assert mock.interrupted == 1

@pytest.mark.asyncio
async def test_filler_passes_when_agent_idle():
    mock = MockSession()
    handler = InterruptHandler(mock, ignored_words={"uh","umm"}, stop_words={"stop"})
    handler.start()
    await handler._on_agent_state_changed({"new_state":"idle"})
    await handler._on_user_input_transcribed({"is_final": True, "transcript":"umm", "confidence":1.0})
    assert mock.interrupted == 0
