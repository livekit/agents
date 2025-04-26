from __future__ import annotations

import os
import pathlib

import jiwer as tr

from livekit import rtc
from livekit.agents import utils

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "long.mp3")
TEST_AUDIO_TRANSCRIPT = pathlib.Path(os.path.dirname(__file__), "long_transcript.txt").read_text()


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


async def read_mp3_file(path) -> rtc.AudioFrame:
    decoder = utils.codecs.AudioStreamDecoder(
        sample_rate=48000,
        num_channels=1,
    )
    frames: list[rtc.AudioFrame] = []
    with open(path, "rb") as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            decoder.push(chunk)
        decoder.end_input()
    async for frame in decoder:
        frames.append(frame)

    return rtc.combine_audio_frames(frames)  # merging just for ease of use


async def make_test_speech(
    *,
    chunk_duration_ms: int | None = None,
    sample_rate: int | None = None,  # resample if not None
) -> tuple[list[rtc.AudioFrame], str]:
    input_audio = await read_mp3_file(TEST_AUDIO_FILEPATH)

    if sample_rate is not None and input_audio.sample_rate != sample_rate:
        resampler = rtc.AudioResampler(
            input_rate=input_audio.sample_rate,
            output_rate=sample_rate,
            num_channels=input_audio.num_channels,
        )

        frames = []
        if resampler:
            frames = resampler.push(input_audio)
            frames.extend(resampler.flush())

        input_audio = rtc.combine_audio_frames(frames)

    if not chunk_duration_ms:
        return [input_audio], TEST_AUDIO_TRANSCRIPT

    chunk_size = int(input_audio.sample_rate / (1000 / chunk_duration_ms))
    bstream = utils.audio.AudioByteStream(
        sample_rate=input_audio.sample_rate,
        num_channels=input_audio.num_channels,
        samples_per_channel=chunk_size,
    )

    frames = bstream.write(input_audio.data.tobytes())
    frames.extend(bstream.flush())
    return frames, TEST_AUDIO_TRANSCRIPT


def setup_plugin_credentials(plugin: str):
    if plugin.startswith("google") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        # For Google TTS, we need to extract credentials from env var to a file
        google_creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
        if google_creds:
            import tempfile

            temp_dir = tempfile.TemporaryDirectory()
            creds_path = os.path.join(temp_dir.name, "google.json")

            # Write credentials to file
            with open(creds_path, "w") as f:
                f.write(google_creds)

            # Set environment variable to point to the credentials file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
