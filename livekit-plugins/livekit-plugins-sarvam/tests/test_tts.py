from __future__ import annotations

import asyncio
import base64
import io
import wave
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


def _generate_wav_bytes(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate a complete RIFF/WAVE file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(_generate_raw_pcm(duration_ms, sample_rate))
    return buf.getvalue()


def test_allowed_output_audio_codecs_contains_wav():
    """Verify that wav is in the allowed output codecs."""
    assert "wav" in ALLOWED_OUTPUT_AUDIO_CODECS


def test_codec_to_mime_maps_wav_to_audio_wav():
    """Verify that wav maps to audio/wav for REST synthesis."""
    assert _CODEC_TO_MIME["wav"] == "audio/wav"
    assert _codec_to_mime_type("wav") == "audio/wav"
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
async def test_chunked_stream_with_wav_codec_handles_riff_wav():
    """Verify ChunkedStream with output_audio_codec=\"wav\" handles valid RIFF WAV from REST."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    wav_bytes = _generate_wav_bytes(duration_ms=100, sample_rate=SAMPLE_RATE)
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")

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

    mock_session.post.assert_called_once()
    _, kwargs = mock_session.post.call_args
    assert kwargs["json"]["output_audio_codec"] == "wav"

    assert len(events) > 0
    assert events[-1].is_final
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    expected_samples = (len(wav_bytes) - 44) // 2
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
    emitter.initialize(
        request_id="test-req-stream",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
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
async def test_synthesize_stream_with_wav_codec_strips_riff_header_if_present():
    """Verify SynthesizeStream strips RIFF header if provider sends full WAV container on WebSocket."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    wav_bytes = _generate_wav_bytes(duration_ms=100, sample_rate=SAMPLE_RATE)
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-riff", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-riff",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-2")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

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
    expected_samples = (len(wav_bytes) - 44) // 2
    assert total_samples == expected_samples


@pytest.mark.asyncio
async def test_synthesize_stream_with_wav_codec_handles_extended_chunks():
    """Verify SynthesizeStream locates the data chunk even with metadata/extended chunks."""
    import struct

    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    junk_payload = b"HelloMetadata\x00"
    junk_chunk = struct.pack("<4sI", b"JUNK", len(junk_payload)) + junk_payload
    data_header = struct.pack("<4sI", b"data", len(raw_pcm))
    body = fmt_chunk + junk_chunk + data_header + raw_pcm
    riff_header = struct.pack("<4sI4s", b"RIFF", 4 + len(body), b"WAVE")
    wav_bytes = riff_header + body

    b64_audio = base64.b64encode(wav_bytes).decode("ascii")
    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-extended", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-extended",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-3")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

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
async def test_synthesize_stream_with_wav_codec_handles_split_header_chunks():
    """Verify SynthesizeStream reassembles WAV headers split across multiple WebSocket messages."""
    import struct

    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    junk_payload = b"MetadataAcrossChunks"
    junk_chunk = struct.pack("<4sI", b"JUNK", len(junk_payload)) + junk_payload
    data_header = struct.pack("<4sI", b"data", len(raw_pcm))
    body = fmt_chunk + junk_chunk + data_header + raw_pcm
    riff_header = struct.pack("<4sI4s", b"RIFF", 4 + len(body), b"WAVE")
    wav_bytes = riff_header + body

    # Split midway through the headers (before 'data' chunk)
    split_pos = len(riff_header) + len(fmt_chunk) + 4  # inside the JUNK chunk
    chunk1 = wav_bytes[:split_pos]
    chunk2 = wav_bytes[split_pos:]

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-split", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-split",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-split")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    # Send first chunk (header only)
    msg1 = {"type": "audio", "data": {"audio": base64.b64encode(chunk1).decode("ascii")}}
    success1 = await stream._handle_audio_message(msg1, emitter)
    assert success1 is True

    # Send second chunk (remainder of headers + PCM)
    msg2 = {"type": "audio", "data": {"audio": base64.b64encode(chunk2).decode("ascii")}}
    success2 = await stream._handle_audio_message(msg2, emitter)
    assert success2 is True

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
async def test_synthesize_stream_with_wav_codec_buffers_partial_riff_prefix():
    """Verify SynthesizeStream buffers partial RIFF prefixes across messages without classifying as PCM."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    wav_bytes = _generate_wav_bytes(duration_ms=100, sample_rate=SAMPLE_RATE)

    # Split within the 4-byte RIFF signature (e.g. b"RI" in msg1, remainder in msg2)
    chunk1 = wav_bytes[:2]
    chunk2 = wav_bytes[2:]

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-partial-riff", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-partial-riff",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-partial-riff")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    msg1 = {"type": "audio", "data": {"audio": base64.b64encode(chunk1).decode("ascii")}}
    success1 = await stream._handle_audio_message(msg1, emitter)
    assert success1 is True
    # At this point, no audio should have been emitted because b"RI" is buffered
    assert len(events) == 0

    msg2 = {"type": "audio", "data": {"audio": base64.b64encode(chunk2).decode("ascii")}}
    success2 = await stream._handle_audio_message(msg2, emitter)
    assert success2 is True

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
async def test_synthesize_stream_with_wav_codec_handles_raw_pcm_with_partial_riff_prefix():
    """Verify SynthesizeStream handles raw PCM that coincidentally begins with 'RI'."""
    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    # Raw PCM starting with b"RI" followed by non-WAVE bytes
    raw_pcm = b"RI" + b"\x01\x02\x03\x04" * 50
    chunk1 = raw_pcm[:2]
    chunk2 = raw_pcm[2:]

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-pcm-ri", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-pcm-ri",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-pcm-ri")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    msg1 = {"type": "audio", "data": {"audio": base64.b64encode(chunk1).decode("ascii")}}
    success1 = await stream._handle_audio_message(msg1, emitter)
    assert success1 is True

    msg2 = {"type": "audio", "data": {"audio": base64.b64encode(chunk2).decode("ascii")}}
    success2 = await stream._handle_audio_message(msg2, emitter)
    assert success2 is True

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
async def test_synthesize_stream_with_wav_codec_discards_trailing_metadata_chunks():
    """Verify SynthesizeStream honors data chunk length and discards trailing chunks like LIST."""
    import struct

    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    data_header = struct.pack("<4sI", b"data", len(raw_pcm))
    list_payload = b"INFOINAMTestAudioTrack\x00"
    list_chunk = struct.pack("<4sI", b"LIST", len(list_payload)) + list_payload

    body = fmt_chunk + data_header + raw_pcm + list_chunk
    riff_header = struct.pack("<4sI4s", b"RIFF", 4 + len(body), b"WAVE")
    wav_bytes = riff_header + body

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-trailing", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-trailing",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-trailing")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    msg = {"type": "audio", "data": {"audio": base64.b64encode(wav_bytes).decode("ascii")}}
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
async def test_synthesize_stream_with_wav_codec_discards_trailing_metadata_across_chunks():
    """Verify SynthesizeStream discards trailing metadata chunks split across multiple WebSocket messages."""
    import struct

    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    raw_pcm = _generate_raw_pcm(duration_ms=100, sample_rate=SAMPLE_RATE)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    data_header = struct.pack("<4sI", b"data", len(raw_pcm))
    list_payload = b"INFOINAMTestAudioTrackAcrossWebSocketMessages\x00"
    list_chunk = struct.pack("<4sI", b"LIST", len(list_payload)) + list_payload

    body = fmt_chunk + data_header + raw_pcm + list_chunk
    riff_header = struct.pack("<4sI4s", b"RIFF", 4 + len(body), b"WAVE")
    wav_bytes = riff_header + body

    # Chunk 1: header + half of PCM
    half_pcm = len(riff_header) + len(fmt_chunk) + len(data_header) + (len(raw_pcm) // 2)
    chunk1 = wav_bytes[:half_pcm]
    # Chunk 2: rest of PCM + half of trailing LIST chunk
    half_list = half_pcm + (len(raw_pcm) - len(raw_pcm) // 2) + (len(list_chunk) // 2)
    chunk2 = wav_bytes[half_pcm:half_list]
    # Chunk 3: rest of trailing LIST chunk
    chunk3 = wav_bytes[half_list:]

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-trailing-split", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-trailing-split",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-trailing-split")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    for chunk in [chunk1, chunk2, chunk3]:
        msg = {"type": "audio", "data": {"audio": base64.b64encode(chunk).decode("ascii")}}
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
async def test_synthesize_stream_with_wav_codec_handles_consecutive_riff_containers_with_trailing_data():
    """Verify SynthesizeStream seamlessly parses consecutive RIFF containers even with trailing chunks."""
    import struct

    sarvam_tts = TTS(
        api_key="test-api-key",
        speech_sample_rate=SAMPLE_RATE,
        output_audio_codec="wav",
    )

    def _make_container(duration_ms: int) -> tuple[bytes, bytes]:
        pcm = _generate_raw_pcm(duration_ms=duration_ms, sample_rate=SAMPLE_RATE)
        fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        data = struct.pack("<4sI", b"data", len(pcm))
        meta = b"JUNK" + struct.pack("<I", 8) + b"ExtraPad"
        body = fmt + data + pcm + meta
        riff = struct.pack("<4sI4s", b"RIFF", 4 + len(body), b"WAVE")
        return riff + body, pcm

    container1, pcm1 = _make_container(duration_ms=60)
    container2, pcm2 = _make_container(duration_ms=80)

    stream = sarvam_tts.stream()

    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test-sarvam-tts-stream-consecutive", dst_ch=dst_ch)
    emitter.initialize(
        request_id="test-req-stream-consecutive",
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        mime_type="audio/pcm",
        stream=True,
    )
    emitter.start_segment(segment_id="seg-consecutive")

    events: list[tts.SynthesizedAudio] = []

    async def collect():
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                return

    collect_task = asyncio.create_task(collect())

    # Send container 1
    msg1 = {"type": "audio", "data": {"audio": base64.b64encode(container1).decode("ascii")}}
    success1 = await stream._handle_audio_message(msg1, emitter)
    assert success1 is True

    # Send container 2
    msg2 = {"type": "audio", "data": {"audio": base64.b64encode(container2).decode("ascii")}}
    success2 = await stream._handle_audio_message(msg2, emitter)
    assert success2 is True

    emitter.end_segment()
    emitter.end_input()
    await emitter.join()
    await collect_task

    assert len(events) > 0
    assert events[-1].is_final
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    expected_samples = (len(pcm1) + len(pcm2)) // 2
    assert total_samples == expected_samples
