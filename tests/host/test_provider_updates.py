"""A provider's options changed after a session took it reach its next request: the session reads
what can change (its model, its sample rate, its markup dialect) when each request is made."""

from __future__ import annotations

from livekit.agents import Agent, VoiceSession, tts
from livekit.agents.metrics import LLMMetrics
from livekit.agents.voice.io import PlaybackFinishedEvent
from tests.fakes import FakeAudioOutput, FakeLLM, FakeTranscriptionOutput, FakeTTS


# Catches: an entry's sample rate read once when a fallback chain was built, so its audio after
# `update_options(sample_rate=..)` is resampled from the old rate: wrong pitch and duration.
async def test_a_tts_whose_rate_changed_plays_its_next_speech_at_the_new_rate() -> None:
    primary = FakeTTS(seconds=0.2, sample_rate=16000)
    chain = tts.FallbackAdapter([primary, FakeTTS(sample_rate=24000)])
    output = FakeAudioOutput()
    played: list[PlaybackFinishedEvent] = []
    output.on("playback_finished", played.append)
    transcription = FakeTranscriptionOutput()

    async with VoiceSession(tts=chain, vad=None) as session:
        session.output.audio = output
        session.output.transcription = transcription
        await session.start(Agent(instructions="Help."))
        await session.say("before")
        primary.update_options(sample_rate=48000)
        await session.say("after")

    assert [round(ev.playback_position, 2) for ev in played] == [0.2, 0.2]
    assert [ev.interrupted for ev in played] == [False, False]
    assert transcription.texts[-2:] == ["before", "after"]


# Catches: an LLM's model read once when the session took it, so after `update_options(model=..)`
# its requests are still counted against the old model.
async def test_a_model_change_labels_the_next_requests_metrics() -> None:
    model = FakeLLM(["first", "second"], model="model-a")
    metrics: list[LLMMetrics] = []
    model.on("metrics_collected", metrics.append)

    async with VoiceSession(llm=model, tts=FakeTTS(), vad=None) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="one")
        model.update_options(model="model-b")
        await session.generate_reply(user_input="two")

    labels = [m.metadata.model_name for m in metrics if m.metadata is not None]
    assert labels == ["model-a", "model-b"]
