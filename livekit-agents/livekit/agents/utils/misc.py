from __future__ import annotations

import time
import uuid
from typing import AsyncIterable, List, Union, overload

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


def time_ms() -> int:
    return int(time.time() * 1000)


def shortuuid() -> str:
    return str(uuid.uuid4().hex)[:12]


@overload
def replace_words(
    *,
    text: str,
    replacements: dict[str, str],
) -> str: ...


@overload
def replace_words(
    *,
    text: AsyncIterable[str],
    replacements: dict[str, str],
) -> AsyncIterable[str]: ...


def replace_words(
    *,
    text: str | AsyncIterable[str],
    replacements: dict[str, str],
) -> str | AsyncIterable[str]:
    """
    Replace words in text with another str
    Args:
        text: text to replace words in
        words: dictionary of words to replace
    """
    if isinstance(text, str):
        for word, replacement in replacements.items():
            text = text.replace(word, replacement)
        return text
    else:

        async def _replace_words():
            buffer = ""
            async for chunk in text:
                for char in chunk:
                    if not char.isspace():
                        buffer += char
                    else:
                        if buffer:
                            yield replacements.get(buffer, buffer)
                            buffer = ""
                        yield char

            if buffer:
                yield replacements.get(buffer, buffer)

        return _replace_words()
