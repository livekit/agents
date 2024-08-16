from __future__ import annotations

import io
import os
import pathlib
import wave

import jiwer as tr
from livekit import rtc
from livekit.agents import utils

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "long.mp3")
TEST_AUDIO_TRANSCRIPT = pathlib.Path(
    os.path.dirname(__file__), "long_transcript.txt"
).read_text()


def wer(hypothesis: str, reference: str) -> float:
    wer_standardize_contiguous = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.RemoveKaldiNonWords(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            tr.ReduceToListOfListOfWords(),
        ]
    )

    return tr.wer(
        reference,
        hypothesis,
        reference_transform=wer_standardize_contiguous,
        hypothesis_transform=wer_standardize_contiguous,
    )


def read_mp3_file(path) -> rtc.AudioFrame:
    mp3 = utils.codecs.Mp3StreamDecoder()
    frames: list[rtc.AudioFrame] = []
    with open(path, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break

            frames.extend(mp3.decode_chunk(chunk))

    return utils.merge_frames(frames)  # merging just for ease of use


def make_test_audio(
    chunk_duration_ms: int | None = None,
) -> (list[rtc.AudioFrame], str):
    mp3_audio = read_mp3_file(TEST_AUDIO_FILEPATH)

    if not chunk_duration_ms:
        return [mp3_audio], TEST_AUDIO_TRANSCRIPT

    chunk_size = int(mp3_audio.sample_rate / (1000 / chunk_duration_ms))
    bstream = utils.audio.AudioByteStream(
        sample_rate=mp3_audio.sample_rate,
        num_channels=mp3_audio.num_channels,
        samples_per_channel=chunk_size,
    )

    frames = bstream.write(mp3_audio.data.tobytes())
    frames.extend(bstream.flush())
    return frames, TEST_AUDIO_TRANSCRIPT


def make_wav_file(frames: list[rtc.AudioFrame]) -> bytes:
    buffer = utils.merge_frames(frames)
    io_buffer = io.BytesIO()
    with wave.open(io_buffer, "wb") as wav:
        wav.setnchannels(buffer.num_channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(buffer.sample_rate)
        wav.writeframes(buffer.data)

    return io_buffer.getvalue()
