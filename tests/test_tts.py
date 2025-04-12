from __future__ import annotations

import os
import pathlib
import pytest

from livekit import agents, rtc
from livekit.agents.utils import AudioBuffer
from livekit.plugins import cartesia, openai
from .utils import wer
from .toxic_proxy import Toxiproxy

WER_THRESHOLD = 0.2
TEST_AUDIO_SYNTHESIZE = pathlib.Path(os.path.dirname(__file__), "long_synthesize.txt").read_text()


async def assert_valid_synthesized_audio(frames: AudioBuffer, sample_rate: int, num_channels: int):
    # use whisper as the source of truth to verify synthesized speech (smallest WER)
    whisper_stt = openai.STT(model="whisper-1")
    res = await whisper_stt.recognize(buffer=frames)
    assert wer(res.alternatives[0].text, TEST_AUDIO_SYNTHESIZE) <= WER_THRESHOLD

    combined_frame = rtc.combine_audio_frames(frames)
    assert combined_frame.sample_rate == sample_rate, "sample rate should be the same"
    assert combined_frame.num_channels == num_channels, "num channels should be the same"


SYNTHESIZE_TTS = [
    pytest.param(lambda: cartesia.TTS(), id="cartesia"),
]


# @pytest.mark.usefixtures("job_process")
# @pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
# async def test_synthesize(tts_factory):
#     tts = tts_factory()

#     frames = []
#     async for audio in tts.synthesize(text=TEST_AUDIO_SYNTHESIZE):
#         frames.append(audio.frame)

#     await assert_valid_synthesized_audio(frames, tts.sample_rate, tts.num_channels)


@pytest.mark.usefixtures("job_process")
@pytest.mark.parametrize("tts_factory", SYNTHESIZE_TTS)
async def test_synthesize_failure(tts_factory):
    tts = tts_factory()

    toxiproxy = Toxiproxy()
    p = toxiproxy.create("api.cartesia.ai:443", "cartesia-proxy", listen="0.0.0.0:443", enabled=True)
    #p.add_toxic(type="timeout", attributes={"timeout": 1})

    frames = []

    async for audio in tts.synthesize(text=TEST_AUDIO_SYNTHESIZE):
        frames.append(audio.frame)

