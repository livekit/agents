from deepgram.transcription import (
    LiveOptions,
    LiveTranscriptionEvent,
    LiveTranscriptionResponse,
    PrerecordedOptions,
    PrerecordedTranscriptionResponse,
    TranscriptionSource,
)
from livekit import rtc, agents
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer
from typing import Union, Optional
from .models import DeepgramModels, DeepgramLanguages
import os
import logging
import asyncio
import deepgram
import wave
import io


class STT(stt.STT):
    def __init__(
        self, api_key: Optional[str] = None, api_url: Optional[str] = None
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key is required")

        dg_opts: deepgram.Options = {"api_key": api_key}
        if api_url:
            dg_opts["api_url"] = api_url

        self._client = deepgram.Deepgram(dg_opts)

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Union[DeepgramLanguages, str] = "en-US",
        detect_language: bool = False,
        num_channels: int = 1,
        sample_rate: int = 16000,
        punctuate: bool = True,
        model: DeepgramModels = "nova-2-general",
        smart_format: bool = True,
    ) -> stt.SpeechEvent:
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
            "model": model,
            "language": language,
            "smart_format": smart_format,
            "punctuate": punctuate,
        }

        dg_res = await self._client.transcription.prerecorded(source, dg_opts)
        return prerecorded_transcription_to_speech_event(language, dg_res)

    def stream(
        self,
        *,
        language: str = "en-US",
        detect_language: bool = False,
        interim_results: bool = True,
        num_channels: int = 1,
        sample_rate: int = 16000,
        punctuate: bool = True,
        model: DeepgramModels = "nova-2-general",
        smart_format: bool = True,
    ) -> "SpeechStream":
        return SpeechStream(
            self._client,
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            num_channels=num_channels,
            sample_rate=sample_rate,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        client: deepgram.Deepgram,
        *,
        language: str,
        detect_language: bool,
        interim_results: bool,
        num_channels: int,
        sample_rate: int,
        punctuate: bool,
        model: DeepgramModels,
        smart_format: bool,
    ) -> None:
        super().__init__()
        self._language = language
        self._detect_language = detect_language
        self._interim_results = interim_results
        self._num_channels = num_channels
        self._sample_rate = sample_rate
        self._punctuate = punctuate
        self._model = model
        self._smart_format = smart_format

        self._client = client
        self._queue = asyncio.Queue[rtc.AudioFrame]()
        self._transcript_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"deepgram task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Deepgram with exponential backoff and forward frames"""
        retry_count = 0
        while True:
            try:
                dg_opts: LiveOptions = {
                    "model": self._model,
                    "language": self._language,
                    "encoding": "linear16",
                    "interim_results": self._interim_results,
                    "channels": self._num_channels,
                    "sample_rate": self._sample_rate,
                    "smart_format": self._smart_format,
                    "punctuate": self._punctuate,
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
                    speech_event = live_transcription_to_speech_event(
                        self._language, event
                    )
                    self._transcript_queue.put_nowait(speech_event)

                self._live.register_handler(LiveTranscriptionEvent.CLOSE, on_close)
                self._live.register_handler(
                    LiveTranscriptionEvent.TRANSCRIPT_RECEIVED,
                    on_transcript_received,
                )

                while opened:
                    frame = await self._queue.get()
                    frame = frame.remix_and_resample(
                        self._sample_rate, self._num_channels
                    )
                    self._live.send(frame.data.tobytes())
                    self._queue.task_done()

            except asyncio.CancelledError:
                await asyncio.shield(self._live.finish())
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

    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._transcript_queue.empty():
            raise StopAsyncIteration

        return await self._transcript_queue.get()


def live_transcription_to_speech_event(
    language: str,
    event: LiveTranscriptionResponse,
) -> stt.SpeechEvent:
    dg_alts = event["channel"]["alternatives"]
    return stt.SpeechEvent(
        is_final=event["is_final"],
        alternatives=[
            stt.SpeechData(
                language=alt.get("detected_language", language),
                start_time=alt["words"][0]["start"] if alt["words"] else 0,
                end_time=alt["words"][-1]["end"] if alt["words"] else 0,
                confidence=alt["confidence"],
                text=alt["transcript"],
            )
            for alt in dg_alts
        ],
    )


def prerecorded_transcription_to_speech_event(
    language: str,
    event: PrerecordedTranscriptionResponse,
) -> stt.SpeechEvent:
    dg_alts = event["results"]["channels"][0]["alternatives"]  # type: ignore
    return stt.SpeechEvent(
        is_final=True,
        alternatives=[
            stt.SpeechData(
                language=alt.get("detected_language", language),
                start_time=alt["words"][0]["start"],
                end_time=alt["words"][-1]["end"],
                confidence=alt["confidence"],
                text=alt["transcript"],
            )
            for alt in dg_alts
        ],
    )
