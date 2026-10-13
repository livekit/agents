from __future__ import annotations

import pytest

from livekit.agents import Agent

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


def test_update_options_not_running_replaces_fields() -> None:
    stt1, stt2 = FakeSTT(), FakeSTT()
    vad1, vad2 = FakeVAD(), FakeVAD()
    llm1, llm2 = FakeLLM(), FakeLLM()
    tts1, tts2 = FakeTTS(), FakeTTS()

    agent = Agent(instructions="test", stt=stt1, vad=vad1, llm=llm1, tts=tts1)
    agent.update_options(stt=stt2, vad=vad2, llm=llm2, tts=tts2)

    assert agent.stt is stt2
    assert agent.vad is vad2
    assert agent.llm is llm2
    assert agent.tts is tts2


def test_update_options_not_running_only_touches_given() -> None:
    stt1 = FakeSTT()
    llm1 = FakeLLM()
    agent = Agent(instructions="test", stt=stt1, llm=llm1)

    tts_new = FakeTTS()
    agent.update_options(tts=tts_new)

    assert agent.stt is stt1  # untouched
    assert agent.llm is llm1  # untouched
    assert agent.tts is tts_new


class _LabeledSTT(FakeSTT):
    @property
    def model(self) -> str:
        return "new-model"

    @property
    def provider(self) -> str:
        return "new-provider"


class _LowSilenceVAD(FakeVAD):
    @property
    def min_silence_duration(self) -> float:
        return 0.0
