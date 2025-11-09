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
import os
import sys
import pytest

# Ensure project root on sys.path (if not already at top of file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    def __init__(self, transcript, is_final=True, confidence=0.9):
        self.transcript = transcript
        self.is_final = is_final
        self.confidence = confidence


@pytest.mark.asyncio
async def test_multilanguage_hindi_filler_ignored_while_speaking():
    # "haan" and "acha" are configured fillers
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session=session, config=cfg)

    ctrl._agent_speaking = True  # simulate TTS speaking

    ev = DummyTranscribedEvent("haan acha", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev)

    # Should NOT interrupt on pure Hindi fillers
    assert session.interrupt_called is False


@pytest.mark.asyncio
async def test_multilanguage_hindi_command_triggers_interrupt():
    # "thoda ruk" is a hard interrupt phrase
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session=session, config=cfg)

    ctrl._agent_speaking = True

    ev = DummyTranscribedEvent("umm thoda ruk please", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev)

    assert session.interrupt_called is True


@pytest.mark.asyncio
async def test_dynamic_update_of_filler_and_commands():
    session = DummySession()
    cfg = InterruptConfig(debug_logging=False)
    ctrl = FillerAwareInterruptController(session=session, config=cfg)

    # 1) At start, "xyz" is NOT a filler, so it should cause interrupt when speaking.
    ctrl._agent_speaking = True
    ev1 = DummyTranscribedEvent("xyz xyz", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev1)
    assert session.interrupt_called is True

    # Reset for next check
    session.interrupt_called = False

    # 2) Dynamically add "xyz" as filler and refresh.
    cfg.add_filler_words(["xyz"])
    ctrl.refresh_hard_commands()  # refresh internal caches if needed

    # Now same input while speaking should be ignored.
    ev2 = DummyTranscribedEvent("xyz xyz", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev2)
    assert session.interrupt_called is False

    # 3) Dynamically add a new hard phrase and verify it triggers.
    cfg.add_hard_phrases(["par karo"])
    ctrl.refresh_hard_commands()

    ev3 = DummyTranscribedEvent("umm par karo please", is_final=True, confidence=0.9)
    await ctrl._handle_transcription_event(ev3)
    assert session.interrupt_called is True
