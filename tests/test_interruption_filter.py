import os
import types
import pytest

from livekit.agents.voice.interruption_filter import InterruptionFilter
from livekit.agents.stt import SpeechEvent, SpeechEventType, SpeechData


class _DummySpeechHandle:
    def __init__(self, interrupted=False):
        self._interrupted = interrupted

    @property
    def interrupted(self) -> bool:
        return self._interrupted


class _DummySession:
    def __init__(self, speaking: bool):
        self._speaking = speaking
        self._current = _DummySpeechHandle(interrupted=False) if speaking else None
        self._agent_state = "speaking" if speaking else "listening"

    @property
    def current_speech(self):
        return self._current

    @property
    def agent_state(self):
        return self._agent_state


class _CaptureHooks:
    def __init__(self):
        self.calls = []

    # RecognitionHooks interface methods
    def on_start_of_speech(self, ev):
        self.calls.append(("start", ev))

    def on_vad_inference_done(self, ev):
        self.calls.append(("vad_infer", ev))

    def on_end_of_speech(self, ev):
        self.calls.append(("end", ev))

    def on_interim_transcript(self, ev, *, speaking: bool | None):
        self.calls.append(("interim", ev, speaking))

    def on_final_transcript(self, ev):
        self.calls.append(("final", ev))

    def on_end_of_turn(self, info):
        self.calls.append(("eot", info))
        return True

    def on_preemptive_generation(self, info):
        self.calls.append(("preempt", info))

    def retrieve_chat_ctx(self):
        # minimal object with required API in AudioRecognition downstream, unused here
        return types.SimpleNamespace(copy=lambda: types.SimpleNamespace(add_message=lambda **_: None))


def _speech_event(text: str, confidence: float, final: bool) -> SpeechEvent:
    t = SpeechEventType.FINAL_TRANSCRIPT if final else SpeechEventType.INTERIM_TRANSCRIPT
    return SpeechEvent(type=t, alternatives=[SpeechData(language="en", text=text, confidence=confidence)])


@pytest.mark.parametrize("speaking", [True, False])
@pytest.mark.parametrize("final", [False, True])
def test_fillers_ignored_only_when_agent_speaks(speaking: bool, final: bool):
    hooks = _CaptureHooks()
    sess = _DummySession(speaking=speaking)
    f = InterruptionFilter(hooks=hooks, session=sess)

    ev = _speech_event("umm", 0.9, final)
    if final:
        f.on_final_transcript(ev)
    else:
        f.on_interim_transcript(ev, speaking=True)

    if speaking:
        # filtered
        assert not any(name in ("interim", "final") for name, *_ in hooks.calls)
    else:
        # forwarded
        names = [name for name, *_ in hooks.calls]
        assert ("final" if final else "interim") in names


@pytest.mark.parametrize("final", [False, True])
def test_keywords_always_interrupt_even_when_speaking(final: bool):
    hooks = _CaptureHooks()
    sess = _DummySession(speaking=True)  # agent speaking
    f = InterruptionFilter(hooks=hooks, session=sess)

    ev = _speech_event("please wait one second", 0.2, final)
    if final:
        f.on_final_transcript(ev)
    else:
        f.on_interim_transcript(ev, speaking=True)

    # forwarded because keyword present
    names = [name for name, *_ in hooks.calls]
    assert ("final" if final else "interim") in names


def test_low_confidence_filtered_when_speaking():
    hooks = _CaptureHooks()
    sess = _DummySession(speaking=True)
    f = InterruptionFilter(hooks=hooks, session=sess)

    # below default 0.5 threshold
    ev = _speech_event("random", 0.1, False)
    f.on_interim_transcript(ev, speaking=True)

    assert not any(name == "interim" for name, *_ in hooks.calls)
