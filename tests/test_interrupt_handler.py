import pytest

from salescode_interrupt_handler.config import InterruptConfig
from salescode_interrupt_handler.interrupt_handler import FillerAwareInterruptController


class DummySession:
    def __init__(self):
        self._handlers = {}
        self.interrupt_called = False

    def on(self, event_name: str):
        def decorator(fn):
            self._handlers.setdefault(event_name, []).append(fn)
            return fn
        return decorator

    async def interrupt(self):
        self.interrupt_called = True


class DummyTranscribedEvent:
    def __init__(self, transcript, is_final=True, confidence=None):
        self.transcript = transcript
        self.is_final = is_final
        if confidence is not None:
            self.confidence = confidence


@pytest.mark.asyncio
async def test_ignores_pure_filler_when_speaking():
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session, cfg)
    ctrl._agent_speaking = True

    ev = DummyTranscribedEvent("uh hmm", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev)

    assert session.interrupt_called is False


@pytest.mark.asyncio
async def test_triggers_on_hard_interrupt_phrase():
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session, cfg)
    ctrl._agent_speaking = True

    ev = DummyTranscribedEvent("umm wait stop please", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev)

    assert session.interrupt_called is True


@pytest.mark.asyncio
async def test_triggers_on_real_sentence():
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session, cfg)
    ctrl._agent_speaking = True

    ev = DummyTranscribedEvent("i have a question", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev)

    assert session.interrupt_called is True


@pytest.mark.asyncio
async def test_low_confidence_ignored():
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False, min_confidence=0.7)
    ctrl = FillerAwareInterruptController(session, cfg)
    ctrl._agent_speaking = True

    ev = DummyTranscribedEvent("stop", is_final=True, confidence=0.3)
    await ctrl._handle_transcription_event(ev)

    assert session.interrupt_called is False
