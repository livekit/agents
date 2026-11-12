"""The noise cancellation models users already ship (livekit-plugins-noise-cancellation NC, BVC
and BVCTelephony, livekit-plugins-krisp, livekit-plugins-ai-coustics) work with 2.0 passed the
documented way, `AudioInputOptions(noise_cancellation=...)`.

The noise-cancellation models are LiveKit Cloud models: on a self-hosted server they are not
authorized and pass the caller's voice through untouched; Krisp runs locally and removes the
noise anywhere. The cloud test needs a LiveKit Cloud project in LIVEKIT_URL, LIVEKIT_API_KEY and
LIVEKIT_API_SECRET."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from livekit import api, rtc
from livekit.agents import Agent
from livekit.agents.utils.media import audio_frames_from_file
from livekit.agents.voice.events import ErrorEvent
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions

from .sfu import KEY, SECRET
from .test_audio_processors import listening_session, loudness

VOICE = Path(__file__).parents[1] / "fixtures" / "change-sophie.opus"
NOISE = 1500  # the RMS of the white noise under the voice
Model = rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame]


def nc(model: str) -> Callable[[], Model]:
    def build() -> Model:
        from livekit.plugins import noise_cancellation

        built: Model = getattr(noise_cancellation, model)()
        return built

    return build


def krisp(telephony: bool) -> Callable[[], Model]:
    def build() -> Model:
        from livekit.plugins import krisp

        return krisp.voice_isolation_telephony() if telephony else krisp.voice_isolation()

    return build


def ai_coustics() -> Model:
    from livekit.plugins import ai_coustics

    enhancement: Model = ai_coustics.audio_enhancement()
    return enhancement


MODELS = {
    "nc": nc("NC"),
    "bvc": nc("BVC"),
    "bvc-telephony": nc("BVCTelephony"),
    "krisp": krisp(telephony=False),
    "krisp-telephony": krisp(telephony=True),
    "ai-coustics": ai_coustics,
}


def models(*names: str) -> list[object]:
    """``names`` as test parameters; ai-coustics is a strict xfail."""
    return [pytest.param(MODELS[name], id=name) for name in names]


CLOUD_ONLY = ("nc", "bvc", "bvc-telephony")
LOCAL = ("krisp", "krisp-telephony", "ai-coustics")


async def say(source: rtc.AudioSource) -> float:
    """Plays the noisy voice into ``source`` in real time; returns its duration."""
    decoded = [f async for f in audio_frames_from_file(str(VOICE), sample_rate=48000)]
    voice = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in decoded])[: 3 * 48000]
    pcm = np.concatenate([np.zeros(48000, dtype=np.int16), voice]).astype(np.float64)
    pcm += np.random.default_rng(0).normal(0, NOISE, pcm.size)
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    for start in range(0, pcm.size, 480):
        frame = pcm[start : start + 480]
        await source.capture_frame(rtc.AudioFrame(frame.tobytes(), 48000, 1, frame.size))
    return pcm.size / 48000


def token(key: str, secret: str, room: str, identity: str) -> str:
    grants = api.VideoGrants(room_join=True, room=room)
    return api.AccessToken(key, secret).with_identity(identity).with_grants(grants).to_jwt()


async def hear_through(
    model: Model, url: str, key: str, secret: str
) -> tuple[list[rtc.AudioFrame], float, list[ErrorEvent]]:
    """A caller says the noisy voice to an agent processing it with ``model``; returns what the
    agent's STT heard, how long the caller spoke, and the session's errors."""
    name = "noise-cancellation-" + uuid.uuid4().hex
    person, agent = rtc.Room(), rtc.Room()
    await person.connect(url, token(key, secret, name, "person"))
    await agent.connect(url, token(key, secret, name, "agent"))
    heard: list[rtc.AudioFrame] = []
    errors: list[ErrorEvent] = []
    session = listening_session(heard)
    session.on("error", errors.append)
    source = rtc.AudioSource(48000, 1)
    try:
        async with session:
            await session.start(
                Agent(instructions="Listen."),
                room=agent,
                room_options=RoomOptions(
                    audio_input=AudioInputOptions(noise_cancellation=model),
                    audio_output=False,
                    transcription_output=False,
                ),
            )
            await person.local_participant.publish_track(
                rtc.LocalAudioTrack.create_audio_track("microphone", source),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            spoken = await say(source)
            await source.wait_for_playout()
    finally:
        await source.aclose()
        await person.disconnect()
        await agent.disconnect()
    return heard, spoken, errors


def noise_floor(frames: list[rtc.AudioFrame]) -> float:
    """The loudness of the quietest tenth of ``frames``: the noise between words."""
    return float(np.percentile([loudness(f) for f in frames], 10))


@pytest.fixture
def livekit_cloud() -> tuple[str, str, str]:
    names = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    if missing := [name for name in names if not os.environ.get(name)]:
        pytest.skip("a LiveKit Cloud project is needed in " + ", ".join(missing))
    return os.environ[names[0]], os.environ[names[1]], os.environ[names[2]]


@pytest.mark.parametrize("model", models(*CLOUD_ONLY))
async def test_off_cloud_a_cloud_model_passes_the_callers_voice_through(
    sfu: str, model: Callable[[], Model]
) -> None:
    """On a self-hosted server a Cloud model is not authorized: the STT still hears the whole
    call, noise included, in the session's format, and the session closes without an error."""
    heard, spoken, errors = await hear_through(model(), sfu, KEY, SECRET)

    assert {(f.sample_rate, f.num_channels) for f in heard} == {(24000, 1)}
    assert sum(f.duration for f in heard) > 0.9 * spoken
    assert noise_floor(heard) > NOISE / 4
    assert errors == []


@pytest.mark.parametrize("model", models(*LOCAL))
async def test_off_cloud_a_local_model_removes_the_noise(
    sfu: str, model: Callable[[], Model]
) -> None:
    """A model that runs locally removes the noise on a self-hosted server and keeps the voice."""
    heard, spoken, errors = await hear_through(model(), sfu, KEY, SECRET)

    assert sum(f.duration for f in heard) > 0.9 * spoken
    assert noise_floor(heard) < NOISE / 4
    assert max(loudness(f) for f in heard) > NOISE
    assert errors == []


@pytest.mark.parametrize("model", models(*MODELS))
async def test_on_livekit_cloud_the_model_removes_the_noise(
    livekit_cloud: tuple[str, str, str], model: Callable[[], Model]
) -> None:
    """On LiveKit Cloud the STT hears the voice with the noise between words removed."""
    heard, spoken, errors = await hear_through(model(), *livekit_cloud)

    assert sum(f.duration for f in heard) > 0.9 * spoken
    assert noise_floor(heard) < NOISE / 10
    assert errors == []
