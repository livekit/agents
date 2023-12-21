from deepgram.transcription import (
    LiveOptions,
    LiveTranscriptionEvent,
    LiveTranscriptionResponse,
    PrerecordedOptions,
    PrerecordedTranscriptionResponse,
    TranscriptionSource,
)
from livekit import agents, rtc
from dataclasses import dataclass
from typing import Literal, Optional
import os
import logging
import asyncio
import deepgram
import wave
import io

DeepgramModels = Literal[
    "nova-general",
    "nova-phonecall",
    "nova-meeting",
    "nova-2-general",
    "nova-2-meeting",
    "nova-2-phonecall",
    "nova-2-finance",
    "nova-2-conversationalai",
    "nova-2-voicemail",
    "nova-2-video",
    "nova-2-medical",
    "nova-2-drivethru",
    "nova-2-automotive",
    "enhanced-general",
    "enhanced-meeting",
    "enhanced-phonecall",
    "enhanced-finance",
    "base",
    "meeting",
    "phonecall",
    "finance",
    "conversationalai",
    "voicemail",
    "video",
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
]

DeepgramLanguages = Literal[
    "zh",
    "zh-CN",
    "zh-TW",
    "da",
    "nl",
    "en",
    "en-US",
    "en-AU",
    "en-GB",
    "en-NZ",
    "en-IN",
    "fr",
    "fr-CA",
    "de",
    "hi",
    "hi-Latn",
    "pt",
    "pt-BR",
    "es",
    "es-419",
    "hi",
    "hi-Latn",
    "it",
    "ja",
    "ko",
    "no",
    "pl",
    "pt",
    "pt-BR",
    "es-LATAM",
    "sv",
    "ta",
    "taq",
    "uk",
    "tr",
    "sv",
    "id",
    "pt",
    "pt-BR",
    "ru",
    "th",
]


@dataclass
class StreamOptions(agents.StreamOptions):
    model: DeepgramModels = "nova-2-general"
    language: DeepgramLanguages = "en-US"
    smart_format: bool = True


@dataclass
class RecognizeOptions(agents.RecognizeOptions):
    model: DeepgramModels = "nova-2-general"
    language: DeepgramLanguages = "en-US"
    smart_format: bool = True


class STT(agents.STT):
    def __init__(
        self, api_key: Optional[str] = None, api_url: Optional[str] = None
    ) -> None:
        api_key = api_key or os.environ.get("DG_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key is required")

        dg_opts: deepgram.Options = {"api_key": api_key}
        if api_url:
            dg_opts["api_url"] = api_url

        self._client = deepgram.Deepgram(dg_opts)

    async def recognize(
        self,
        buffer: agents.AudioBuffer,
        opts: RecognizeOptions = RecognizeOptions(),
    ) -> agents.SpeechEvent:
        # Deepgram prerecorded API requires WAV/MP3, so we write our PCM into a file
        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        source: TranscriptionSource = {
            "buffer": io_buffer.getvalue(),
            "mimetype": "audio/wav",
        }

        dg_opts: PrerecordedOptions = {
            "model": opts.model,
            "language": opts.language,
            "smart_format": opts.smart_format,
        }

        dg_res = await self._client.transcription.prerecorded(source, dg_opts)
        return prerecorded_transcription_to_speech_event(opts, dg_res)

    def stream(self, opts: StreamOptions = StreamOptions()) -> "SpeechStream":
        return SpeechStream(opts, self._client)


class SpeechStream(agents.SpeechStream):
    def __init__(self, opts: StreamOptions, client: deepgram.Deepgram) -> None:
        self._opts = opts
        self._client = client
        self._queue = asyncio.Queue[rtc.AudioFrame]()
        self._transcript_queue = asyncio.Queue[agents.SpeechEvent]()
        self._closed = False

        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"deepgram task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self, max_retry: int = 5) -> None:
        """Try to connect to Deepgram with exponential backoff and send frames"""
        retry_count = 0
        while True:
            try:
                dg_opts: LiveOptions = {
                    "model": self._opts.model,
                    "language": self._opts.language,
                    "encoding": "linear16",
                    "interim_results": self._opts.interim_results,
                    "channels": self._opts.num_channels,
                    "sample_rate": self._opts.sample_rate,
                    "smart_format": self._opts.smart_format,
                }

                self._live = await self._client.transcription.live(dg_opts)
                retry_count = 0
                opened = True

                def on_close(code: int) -> None:
                    nonlocal opened
                    opened = False

                def on_transcript_received(event: LiveTranscriptionResponse) -> None:
                    if event.get("type") != "Results":  # not documented by Deepgram
                        return
                    speech_event = live_transcription_to_speech_event(self._opts, event)
                    self._transcript_queue.put_nowait(speech_event)

                self._live.register_handler(LiveTranscriptionEvent.CLOSE, on_close)
                self._live.register_handler(
                    LiveTranscriptionEvent.TRANSCRIPT_RECEIVED,
                    on_transcript_received,
                )

                while opened:
                    frame = await self._queue.get()
                    frame = frame.remix_and_resample(
                        self._opts.sample_rate, self._opts.num_channels
                    )
                    self._live.send(frame.data.tobytes())
                    self._queue.task_done()

            except asyncio.CancelledError:

                async def _close() -> None:
                    await self._live.finish()

                await asyncio.shield(_close())
                break
            except Exception as e:
                if retry_count > max_retry and max_retry > 0:
                    logging.error(f"failed to connect to Deepgram: {e}")
                    break

                retry_delay = 2**retry_count - 1
                retry_count += 1
                logging.warning(
                    f"failed to connect to Deepgram: {e} - retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        self._closed = True

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    def __aiter__(self) -> "SpeechStream":
        return self

    async def __anext__(self) -> agents.SpeechEvent:
        if self._closed and self._transcript_queue.empty():
            raise StopAsyncIteration

        return await self._transcript_queue.get()


def live_transcription_to_speech_event(
    opts: StreamOptions,
    event: LiveTranscriptionResponse,
) -> agents.SpeechEvent:
    dg_alts = event["channel"]["alternatives"]
    return agents.SpeechEvent(
        is_final=event["is_final"],
        alternatives=[
            agents.SpeechData(
                language=alt.get("detected_language", opts.language),
                start_time=0,  # alt["words"][0]["start"],
                end_time=0,  # alt["words"][-1]["end"],
                confidence=alt["confidence"],
                text=alt["transcript"],
            )
            for alt in dg_alts
        ],
    )


def prerecorded_transcription_to_speech_event(
    opts: RecognizeOptions,
    event: PrerecordedTranscriptionResponse,
) -> agents.SpeechEvent:
    dg_alts = event["results"]["channels"][0]["alternatives"]  # type: ignore
    return agents.SpeechEvent(
        is_final=True,
        alternatives=[
            agents.SpeechData(
                language=alt.get("detected_language", opts.language),
                start_time=alt["words"][0]["start"],
                end_time=alt["words"][-1]["end"],
                confidence=alt["confidence"],
                text=alt["transcript"],
            )
            for alt in dg_alts
        ],
    )
