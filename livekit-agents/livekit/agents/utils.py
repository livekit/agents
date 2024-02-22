import asyncio
import logging
from typing import Union, List
from livekit import rtc

AudioBuffer = Union[List[rtc.AudioFrame], rtc.AudioFrame]


def merge_frames(buffer: AudioBuffer) -> rtc.AudioFrame:
    """
    Merges one or more AudioFrames into a single one
    Args:
        buffer: either a rtc.AudioFrame or a list of rtc.AudioFrame
    """
    if isinstance(buffer, list):
        # merge all frames into one
        if len(buffer) == 0:
            raise ValueError("buffer is empty")

        sample_rate = buffer[0].sample_rate
        num_channels = buffer[0].num_channels
        samples_per_channel = 0
        data = b""
        for frame in buffer:
            if frame.sample_rate != sample_rate:
                raise ValueError("sample rate mismatch")

            if frame.num_channels != num_channels:
                raise ValueError("channel count mismatch")

            data += frame.data
            samples_per_channel += frame.samples_per_channel

        return rtc.AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

    return buffer


class Mp3Chunker:
    """Mp3Chunker is a class that takes in byte chunks and outputs mp3 chunks
    by finding mp3 headers. It is an async iterator that yields playable mp3 chunks. It
    is not thread safe.
    """

    def __init__(self):
        self.working_bytes = b""
        self.decode_chunks = asyncio.Queue[bytes]()

    def push_chunk(self, chunk: bytes | bytearray | None):
        """Push a new bytes chunk (i.e. from a network response) to the chunker. When you want to close the chunker and
        make sure all chunks are processed, call with None. No additional chunks can be pushed after None is pushed
        and the chunker will raise StopAsyncIteration after all chunks are processed.

        Args:
            chunk (bytes | bytearray | None)
        """

        if chunk is None:
            if len(self.working_bytes) > 0:
                logging.error(
                    "Mp3Chunker had remaining bytes, unchunked bytes after pushing None. This means data has been lost"
                )
            self.decode_chunks.put_nowait(None)
            return

        # find mp3 header
        first_header_index = -1
        last_header_index = -1
        for i in range(len(chunk)):
            if chunk[i] == 0xFF and chunk[i + 1] & 0xE0 == 0xE0:
                if first_header_index == -1:
                    first_header_index = i
                if last_header_index == -1 or last_header_index < i:
                    last_header_index = i

        # Note: this algorithm is designed so that the working bytes always start with a header.
        # Assumption: the first chunk will always start with a header

        # In this chunk, we found at least one mp3 headers
        if first_header_index > 0:
            # Add everything before the first header to the working bytes
            working_bytes += chunk[:first_header_index]

            # If we have a full mp3 chunk in the http chunk we take the working bytes
            # which always start with a header and add it to the decode chunks
            if last_header_index > first_header_index:
                self.decode_chunks.put_nowait(
                    working_bytes
                    + chunk[first_header_index:last_header_index]
                )
                # Whatever is left in the chunk is set to the working bytes
                working_bytes = chunk[last_header_index:]
            # Otherwise this is an incomplete mp3 chunk so we add it to the working bytes
            else:
                working_bytes = chunk[first_header_index:]
        # If the http chunk had no mp3 headers, we just add it to the working bytes
        else:
            working_bytes += chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.decode_chunks.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
