"""livekit-plugins-sherpa TTS against real models, checked by ear: the plugin's own STT
has to hear what each voice said.

The STT is not perfect on synthetic speech ("TODAY" can come back as "TO DAY"), so the
check is a character error rate, not an exact match. It catches silence, noise, a wrong
sample rate or a wrong voice table, which is what this test is for.

Needs the model files once (downloaded on first run); skipped when offline without them.
"""

from __future__ import annotations

import jiwer
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import tts, utils
from livekit.plugins import sherpa
from livekit.plugins.sherpa.models import STT_MODELS, TTS_MODELS, download_model

pytestmark = pytest.mark.plugin("sherpa")


@pytest.fixture(scope="module")
def voice() -> sherpa.TTS:
    try:
        download_model(TTS_MODELS["kokoro-int8-multi-lang-v1_0"])
        download_model(STT_MODELS["zipformer-en-2023-06-26"])
    except Exception as e:
        pytest.skip(f"sherpa models not available: {e}")
    return sherpa.TTS()


MAX_CER = 0.2


async def _transcribe(audio: rtc.AudioFrame) -> str:
    lead = np.zeros(audio.sample_rate // 2, dtype=np.int16).tobytes()
    stream = sherpa.STT().stream()
    for frame in utils.audio.AudioByteStream(
        audio.sample_rate, 1, samples_per_channel=audio.sample_rate // 100
    ).push(lead + audio.data):
        stream.push_frame(frame)
    stream.end_input()
    finals = [ev.alternatives[0].text async for ev in stream if ev.type == "final_transcript"]
    await stream.aclose()
    return " ".join(finals)


async def test_synthesized_speech_is_understood(voice: sherpa.TTS) -> None:
    audio = await voice.synthesize("The weather in Toronto is sunny today.").collect()

    assert audio.sample_rate == 24000
    assert 1.0 < audio.duration < 6.0
    heard = await _transcribe(audio)
    assert jiwer.cer("THE WEATHER IN TORONTO IS SUNNY TODAY", heard) <= MAX_CER, heard


async def test_stream_adapter_speaks_llm_tokens(voice: sherpa.TTS) -> None:
    # How AgentSession drives a non-streaming TTS: tokens in, sentences synthesized one by one.
    adapter = tts.StreamAdapter(tts=voice)
    stream = adapter.stream()
    for token in ["Your table ", "is booked. ", "See you ", "at seven."]:
        stream.push_text(token)
    stream.end_input()
    frames = [ev.frame async for ev in stream]
    await stream.aclose()

    audio = rtc.combine_audio_frames(frames)
    assert np.abs(np.frombuffer(audio.data, dtype=np.int16)).max() > 1000
    heard = await _transcribe(audio)
    assert jiwer.cer("YOUR TABLE IS BOOKED SEE YOU AT SEVEN", heard) <= MAX_CER, heard


@pytest.mark.parametrize(
    ("model", "voice_name", "rate"),
    [("kitten-nano-en-v0_2", "expr-voice-2-f", 24000), ("piper-en_US-amy-low", "amy", 16000)],
)
async def test_other_sherpa_voices_are_understood(
    voice: sherpa.TTS, model: sherpa.TTSModels, voice_name: str, rate: int
) -> None:
    try:
        download_model(TTS_MODELS[model])
    except Exception as e:
        pytest.skip(f"{model} not available: {e}")
    other = sherpa.TTS(model=model, voice=voice_name)

    audio = await other.synthesize("The weather in Toronto is sunny today.").collect()

    assert audio.sample_rate == rate
    heard = await _transcribe(audio)
    assert jiwer.cer("THE WEATHER IN TORONTO IS SUNNY TODAY", heard) <= MAX_CER, heard


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.upper() if c.isalnum() or c.isspace()).split())


@pytest.mark.parametrize("batch_model", ["moonshine-tiny-en-int8", "whisper-tiny.en-int8"])
async def test_batch_stt_hears_kokoro(voice: sherpa.TTS, batch_model: sherpa.STTModels) -> None:
    try:
        download_model(STT_MODELS[batch_model])
    except Exception as e:
        pytest.skip(f"{batch_model} not available: {e}")
    audio = await voice.synthesize("The weather in Toronto is sunny today.").collect()

    event = await sherpa.STT(model=batch_model).recognize([audio])

    heard = _norm(event.alternatives[0].text)
    assert jiwer.cer("THE WEATHER IN TORONTO IS SUNNY TODAY", heard) <= MAX_CER, heard
