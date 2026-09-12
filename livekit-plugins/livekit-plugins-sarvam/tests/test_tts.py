from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import tts, utils
from livekit.plugins.sarvam.tts import (
    _CODEC_TO_MIME,
    ALLOWED_OUTPUT_AUDIO_CODECS,
    TTS,
    _codec_to_mime_type,
)

pytestmark = pytest.mark.unit

SAMPLE_RATE = 22050
NUM_CHANNELS = 1


def _generate_raw_pcm(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate raw 16-bit PCM bytes (no RIFF/WAVE header)."""
    num_samples = sample_rate * duration_ms // 1000
    return b"\x00\x00" * num_samples


def test_allowed_output_audio_codecs_contains_wav():
    """Verify that wav is in the allowed output codecs."""
    assert "wav" in ALLOWED_OUTPUT_AUDIO_CODECS


def test_codec_to_mime_maps_wav_to_audio_pcm():
    """Verify that wav maps to audio/pcm so AudioEmitter treats it as raw PCM."""
    assert _CODEC_TO_MIME["wav"] == "audio/pcm"
    assert _codec_to_mime_type("wav") == "audio/pcm"
    assert _codec_to_mime_type("linear16") == "audio/pcm"
    assert _codec_to_mime_type("mp3") == "audio/mp3"


def test_tts_init_and_update_with_wav_codec():
    """Verify TTS initializes and updates with output_audio_codec=\"wav\"."""
    sarvam_tts = TTS(api_key="test-api-key", output_audio_codec="wav")
    assert sarvam_tts._opts.output_audio_codec == "wav"

    sarvam_tts.update_options(output_audio_codec="mp3")
    assert sarvam_tts._opts.output_audio_codec == "mp3"

    sarvam_tts.update_options(output_audio_codec="wav")
    assert sarvam_tts._opts.output_audio_codec == "wav"


@pytest.mark.asyncio
async def test_chunked_stream_with_wav_codec_emits_audio_without_wav_header_error():
    """Verify ChunkedStream with output_audio_codec=\"wav\" handles raw PCM successfully."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    b64_audio = base64.b64encode(raw_pcm).decode("ascii")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "request_id": "test-req-id",
            "audios": [b64_audio],
        }
    )

    post_cm = AsyncMock()
    post_cm.__aenter__.return_value = mock_response
    post_cm.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.post.return_value = post_cm
    sarvam_tts._session = mock_session

    chunked_stream = sarvam_tts.synthesize("Test text")

    events: list[tts.SynthesizedAudio] = []
    async for ev in chunked_stream:
        events.append(ev)

    # Verify request payload sent to Sarvam API has output_audio_codec == "wav"
    mock_session.post.assert_called_once()
    _, kwargs = mock_session.post.call_args
    assert kwargs["json"]["output_audio_codec"] == "wav"

    # Verify AudioEmitter initialized with audio/pcm and frames were received
    assert len(events) > 0
    assert events[-1].is_final
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    expected_samples = len(raw_pcm) // 2
    assert total_samples == expected_samples


@pytest.mark.asyncio
async def test_synthesize_stream_with_wav_codec_handles_raw_pcm():
    """Verify SynthesizeStream with output_audio_codec=\"wav\" pushes raw PCM without error."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    b64_audio = base64.b64encode(raw_pcm).decode("ascii")

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream", dst_ch=dst_ch)
    mime_type = _codec_to_mime_type(stream._opts.output_audio_codec)
    assert mime_type == "audio/pcm"

    emitter.initialize(
        request_id="test-req-stream",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type=mime_type,
        stream=True,
    )
    emitter.start_segment(segment_id="seg-1")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    # Handle audio message containing raw PCM bytes
    msg = {
        "type": "audio",
        "data": {
            "audio": b64_audio,
        },
    }
    success = await stream._handle_audio_message(msg, emitter)
    assert success is True

    emitter.end_segment()
    emitter.end_input()
    await emitter.join()
    await collect_task

    assert len(events) > 0
    assert events[-1].is_final
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    expected_samples = len(raw_pcm) // 2
    assert total_samples == expected_samples


@pytest.mark.asyncio
async def test_audio_emitter_wav_vs_pcm_regression():
    """Demonstrate that audio/pcm succeeds on raw PCM where audio/wav fails with missing RIFF."""
    from livekit.agents.utils.codecs.decoder import _WavInlineDecoder

    raw_pcm = _generate_raw_pcm(duration_ms=50, sample_rate=SAMPLE_RATE)

    # 1. WAV decoder raises ValueError on raw PCM because RIFF/WAVE header is missing
    ch = utils.aio.Chan()
    wav_decoder = _WavInlineDecoder(ch, SAMPLE_RATE)
    with pytest.raises(ValueError, match="Invalid WAV file: missing RIFF/WAVE"):
        wav_decoder.push(raw_pcm)

    # 2. audio/pcm (our mapping for wav codec) succeeds on raw PCM and emits frames
    pcm_dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    pcm_emitter = tts.AudioEmitter(label="test-pcm-pass", dst_ch=pcm_dst_ch)
    pcm_emitter.initialize(
        request_id="req-pcm",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type=_codec_to_mime_type("wav"),
    )
    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in pcm_dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())
    pcm_emitter.push(raw_pcm)
    pcm_emitter.end_input()
    await pcm_emitter.join()
    await collect_task

    assert len(events) > 0
    assert events[-1].is_final
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    expected_samples = len(raw_pcm) // 2
    assert total_samples == expected_samples
