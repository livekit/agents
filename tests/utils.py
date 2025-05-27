from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import jiwer as tr
import tiktoken

from livekit import rtc
from livekit.agents import utils

# TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "long.mp3")
# TEST_AUDIO_TRANSCRIPT = pathlib.Path(os.path.dirname(__file__), "long_transcript.txt").read_text()


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


class EventCollector:
    def __init__(self, emitter: rtc.EventEmitter, event: str) -> None:
        emitter.on(event, self._on_event)
        self._events = []

    def _on_event(self, *args, **kwargs) -> None:
        self._events.append((args, kwargs))

    @property
    def events(self) -> list[tuple[tuple, dict]]:
        return self._events

    @property
    def count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()


async def read_audio_file(path) -> rtc.AudioFrame:
    frames = []
    async for f in utils.audio.audio_frames_from_file(path, sample_rate=48000, num_channels=1):
        frames.append(f)

    return rtc.combine_audio_frames(frames)


# async def make_test_speech(
#     *,
#     chunk_duration_ms: int | None = None,
#     sample_rate: int | None = None,  # resample if not None
# ) -> tuple[list[rtc.AudioFrame], str]:
#     input_audio = await read_mp3_file(TEST_AUDIO_FILEPATH)

#     if sample_rate is not None and input_audio.sample_rate != sample_rate:
#         resampler = rtc.AudioResampler(
#             input_rate=input_audio.sample_rate,
#             output_rate=sample_rate,
#             num_channels=input_audio.num_channels,
#         )

#         frames = []
#         if resampler:
#             frames = resampler.push(input_audio)
#             frames.extend(resampler.flush())

#         input_audio = rtc.combine_audio_frames(frames)

#     if not chunk_duration_ms:
#         return [input_audio], TEST_AUDIO_TRANSCRIPT

#     chunk_size = int(input_audio.sample_rate / (1000 / chunk_duration_ms))
#     bstream = utils.audio.AudioByteStream(
#         sample_rate=input_audio.sample_rate,
#         num_channels=input_audio.num_channels,
#         samples_per_channel=chunk_size,
#     )

#     frames = bstream.write(input_audio.data.tobytes())
#     frames.extend(bstream.flush())
#     return frames, TEST_AUDIO_TRANSCRIPT


async def fake_llm_stream(
    text: str, *, model: str = "gpt-4o-mini", tokens_per_second: float = 3.0
) -> AsyncGenerator[str, None]:
    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    sleep_time = 1.0 / max(tokens_per_second, 1e-6)

    for tok_id in token_ids:
        yield enc.decode([tok_id])
        await asyncio.sleep(sleep_time)
