from livekit import rtc, agents
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer
from typing import Union, Optional
from .models import DeepgramModels, DeepgramLanguages
from dataclasses import dataclass
import dataclasses
import os
import logging
import asyncio
import deepgram
import wave
import io


# internal
@dataclass
class STTOptions:
    language: Optional[Union[DeepgramLanguages, str]]
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels
    smart_format: bool
    endpointing: Optional[str]


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: DeepgramLanguages = "en-US",
        detect_language: bool = True,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        model: DeepgramModels = "nova-2-general",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        min_silence_duration: int = 10,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key is required")

        dg_opts = deepgram.DeepgramClientOptions(api_key=api_key, url=api_url or "")
        self._client = deepgram.DeepgramClient(config=dg_opts)
        self._config = STTOptions(
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            smart_format=smart_format,
            endpointing=str(min_silence_duration),
        )

    def _sanitize_options(
        self,
        *,
        language: Optional[str] = None,
    ) -> STTOptions:
        config = dataclasses.replace(self._config)

        if config.detect_language:
            config.language = None

        elif isinstance(language, list):
            logging.warning("deepgram only supports one language at a time")
            config.language = config.language[0]  # type: ignore
        else:
            config.language = language or config.language

        return config

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[Union[DeepgramLanguages, str]] = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        # Deepgram prerecorded API requires WAV/MP3, so we write our PCM into a wav buffer
        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        source: deepgram.BufferSource = {
            "buffer": io_buffer.getvalue(),
        }

        dg_opts = deepgram.PrerecordedOptions(
            model=config.model,
            smart_format=config.smart_format,
            language=config.language,
            punctuate=config.punctuate,
            detect_language=config.detect_language,
        )

        dg_res = await self._client.listen.asyncprerecorded.v("1").transcribe_file(
            source, dg_opts
        )
        return prerecorded_transcription_to_speech_event(config.language, dg_res)

    def stream(
        self,
        *,
        language: Optional[Union[DeepgramLanguages, str]] = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(
            self._client,
            config,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        client: deepgram.DeepgramClient,
        config: STTOptions,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ) -> None:
        super().__init__()
        self._client = client
        self._config = config
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        self._queue = asyncio.Queue[rtc.AudioFrame]()
        self._event_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"deepgram task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        self._main_task.cancel()
        try:
            await self._main_task
        except asyncio.CancelledError:
            pass

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Deepgram with exponential backoff and forward frames"""
        retry_count = 0
        while True:
            try:
                self._live = self._client.listen.asynclive.v("1")

                opened = False

                async def on_close(_, **kwargs) -> None:
                    nonlocal opened
                    opened = False

                async def on_transcript_received(
                    _, result: deepgram.LiveResultResponse, **kwargs
                ) -> None:
                    if result.type != "Results":
                        return

                    speech_event = live_transcription_to_speech_event(
                        self._config.language, result
                    )
                    self._event_queue.put_nowait(speech_event)

                self._live.on(deepgram.LiveTranscriptionEvents.Close, on_close)
                self._live.on(
                    deepgram.LiveTranscriptionEvents.Transcript,
                    on_transcript_received,
                )

                dg_opts = deepgram.LiveOptions(
                    model=self._config.model,
                    language=self._config.language,
                    encoding="linear16",
                    interim_results=self._config.interim_results,
                    channels=self._num_channels,
                    sample_rate=self._sample_rate,
                    smart_format=self._config.smart_format,
                    punctuate=self._config.punctuate,
                    endpointing=self._config.endpointing,
                )
                await self._live.start(dg_opts)
                opened = True
                retry_count = 0

                while opened:
                    frame = await self._queue.get()
                    frame = frame.remix_and_resample(
                        self._sample_rate, self._num_channels
                    )
                    await self._live.send(frame.data.tobytes())
                    self._queue.task_done()

            except asyncio.CancelledError:
                await asyncio.shield(self._live.finish())
                break
            except Exception as e:
                if retry_count > max_retry and max_retry > 0:
                    logging.error(f"failed to connect to Deepgram: {e}")
                    break

                retry_delay = min(retry_count * 5, 5)  # max 5s
                retry_count += 1
                logging.warning(
                    f"failed to connect to Deepgram: {e} - retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

        self._closed = True

    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()


def live_transcription_to_speech_event(
    language: Optional[str],
    event: deepgram.LiveResultResponse,
) -> stt.SpeechEvent:
    dg_alts = event.channel.alternatives  # type: ignore
    if not dg_alts:
        raise ValueError("no alternatives in response")

    return stt.SpeechEvent(
        is_final=event.is_final or False,  # could be None?
        end_of_speech=event.speech_final or False,
        alternatives=[
            stt.SpeechData(
                language=language or "",
                start_time=(alt.words[0].start if alt.words else 0) or 0,
                end_time=(alt.words[-1].end if alt.words else 0) or 0,
                confidence=alt.confidence or 0,
                text=alt.transcript or "",
            )
            for alt in dg_alts
        ],
    )


def prerecorded_transcription_to_speech_event(
    language: Optional[str],
    event: deepgram.PrerecordedResponse,
) -> stt.SpeechEvent:
    dg_alts = event.results.channels[0].alternatives  # type: ignore
    if not dg_alts:
        raise ValueError("no alternatives in response")

    return stt.SpeechEvent(
        is_final=True,
        end_of_speech=True,
        alternatives=[
            stt.SpeechData(
                language=language or "",
                start_time=(alt.words[0].start if alt.words else 0) or 0,
                end_time=(alt.words[-1].end if alt.words else 0) or 0,
                confidence=alt.confidence or 0,
                # not sure why transcript is Optional inside DG SDK ...
                text=alt.transcript or "",
            )
            for alt in dg_alts
        ],
    )
