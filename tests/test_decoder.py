import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pytest
from livekit.agents.stt import SpeechEventType
from livekit.agents.utils.codecs import AudioStreamDecoder, StreamBuffer
from livekit.plugins import deepgram

from .utils import wer

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.opus")


@pytest.mark.asyncio
async def test_decode_and_transcribe():
    # Skip if test file doesn't exist
    if not os.path.exists(TEST_AUDIO_FILEPATH):
        pytest.skip(f"Test file not found: {TEST_AUDIO_FILEPATH}")

    decoder = AudioStreamDecoder()
    with open(TEST_AUDIO_FILEPATH, "rb") as f:
        opus_data = f.read()
    decoder.push(opus_data)
    decoder.end_input()

    session = aiohttp.ClientSession()
    stt = deepgram.STT(http_session=session)
    stream = stt.stream()

    # Push frames to STT
    async for frame in decoder:
        stream.push_frame(frame)

    # Mark end of input
    stream.end_input()

    # Collect results
    final_text = ""
    async for event in stream:
        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            if event.alternatives:
                if final_text:
                    final_text += " "
                final_text += event.alternatives[0].text

    await decoder.aclose()
    await stream.aclose()
    await session.close()

    # Verify the transcription
    expected_text = (
        "the people that are crazy enough to think they can change the world are the ones who do"
    )
    assert wer(final_text, expected_text) < 0.2


def test_stream_buffer():
    buffer = StreamBuffer()
    data_chunks = [b"hello", b"world", b"test", b"data"]
    received_data = bytearray()
    write_completed = threading.Event()

    def writer():
        for chunk in data_chunks:
            buffer.write(chunk)
            time.sleep(0.01)  # Simulate some processing time
        buffer.end_input()
        write_completed.set()

    def reader():
        while True:
            data = buffer.read(4)  # Read in small chunks
            if not data:  # EOF
                break
            received_data.extend(data)

    # Run writer and reader in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        reader_future = executor.submit(reader)
        writer_future = executor.submit(writer)

        # Wait for both threads to complete
        writer_future.result()
        reader_future.result()

    # Verify that all data was received correctly
    expected_data = b"".join(data_chunks)
    assert bytes(received_data) == expected_data


def test_stream_buffer_large_chunks():
    import hashlib

    buffer = StreamBuffer()
    large_chunk = os.urandom(1024 * 1024)  # 1MB of random bytes
    num_chunks = 5
    total_size = 0
    write_completed = threading.Event()
    input_hasher = hashlib.sha256()

    def writer():
        nonlocal total_size
        for _ in range(num_chunks):
            buffer.write(large_chunk)
            total_size += len(large_chunk)
            input_hasher.update(large_chunk)
        buffer.end_input()
        write_completed.set()

    received_size = 0
    output_hasher = hashlib.sha256()

    def reader():
        nonlocal received_size
        # allow writer to start first
        time.sleep(1)
        while True:
            chunk = buffer.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            received_size += len(chunk)
            output_hasher.update(chunk)

    # Run writer and reader in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        reader_future = executor.submit(reader)
        writer_future = executor.submit(writer)

        # Wait for both threads to complete
        writer_future.result()
        reader_future.result()

    assert received_size == total_size
    assert total_size == num_chunks * len(large_chunk)
    assert input_hasher.hexdigest() == output_hasher.hexdigest()


def test_stream_buffer_early_close():
    buffer = StreamBuffer()

    # Write some data
    buffer.write(b"test data")

    # Close the buffer
    buffer.close()

    # Reading from closed buffer should return empty bytes
    assert buffer.read() == b""
