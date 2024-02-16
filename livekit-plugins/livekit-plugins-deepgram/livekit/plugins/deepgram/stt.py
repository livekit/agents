from __future__ import annotations

import asyncio
import dataclasses
from re import I
from typing import List
import io
import json
import logging
import os
import wave
from contextlib import suppress
from urllib.parse import urlencode
from dataclasses import dataclass
from typing import Optional, Union

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from .models import DeepgramLanguages, DeepgramModels


@dataclass
class STTOptions:
    language: DeepgramLanguages | str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: DeepgramModels
    smart_format: bool
    endpointing: str | None


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
        api_key: str | None = None,
        min_silence_duration: int = 10,
    ) -> None:
        super().__init__(streaming_supported=True)
        api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if api_key is None:
            raise ValueError("Deepgram API key is required")
        self._api_key = api_key

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
        language: str | None = None,
    ) -> STTOptions:
        config = dataclasses.replace(self._config)
        config.language = language or config.language

        if config.detect_language:
            config.language = None

        return config

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: DeepgramLanguages | str | None = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        recognize_config = {
            "model": config.model,
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
        }
        if config.language:
            recognize_config["language"] = config.language

        url = f"https://api.deepgram.com/v1/listen?{urlencode(recognize_config)}"

        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        data = io_buffer.getvalue()

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "audio/wav",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, data=data) as res:
                return prerecorded_transcription_to_speech_event(
                    config.language, await res.json()
                )

    def stream(
        self,
        *,
        language: DeepgramLanguages | str | None = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(config, api_key=self._api_key)


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})

    def __init__(
        self,
        config: STTOptions,
        api_key: str,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ) -> None:
        super().__init__()
        self._config = config
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._api_key = api_key

        self._session = aiohttp.ClientSession()
        self._queue = asyncio.Queue()
        self._event_queue = asyncio.Queue[stt.SpeechEvent]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=32))

        # keep a list of final transcripts to combine them inside the END_OF_SPEECH event
        self._final_events = []

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"deepgram task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame)

    async def aclose(self, wait: bool = True) -> None:
        await self._queue.put(SpeechStream._CLOSE_MSG)
        await self._main_task
        await self._session.close()

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Deepgram with exponential backoff and forward frames"""

        closing = False
        live_config = {
            "model": self._config.model,
            "punctuate": self._config.punctuate,
            "smart_format": self._config.smart_format,
            "interim_results": self._config.interim_results,
            "encoding": "linear16",
            "sample_rate": self._sample_rate,
            "vad_events": True,
            "channels": self._num_channels,
            "endpointing": str(self._config.endpointing or "10"),
        }

        if self._config.language:
            live_config["language"] = self._config.language

        headers={"Authorization": f"Token {self._api_key}"}

        url = f"wss://api.deepgram.com/v1/listen?{urlencode(live_config)}"
        ws = await self._session.ws_connect(url, headers=headers)

        async def send_task():
            # if we wan't to keep the connection alive even if no audio is sent,
            # Deepgram expects a keepalive message.
            # https://developers.deepgram.com/reference/listen-live#stream-keepalive
            async def keepalive():
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)

            keepalive_task = asyncio.create_task(keepalive())

            # forward inputs to deepgram
            # if we receive a close message, signal it to deepgram and break.
            # the recv task will then make sure to process the remaining audio and stop
            while True:
                data = await self._queue.get()
                self._queue.task_done()

                if isinstance(data, rtc.AudioFrame):
                    frame = data.remix_and_resample(self._sample_rate, self._num_channels)
                    await ws.send_bytes(frame.data.tobytes())
                elif data == SpeechStream._CLOSE_MSG:
                    closing = True
                    await ws.send_str(data) # tell deepgram we are done with inputs
                    break

            keepalive_task.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive_task

        async def recv_task():
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing: # close is expected, see SpeechStream.aclose
                        break
                    
                    logging.warning("deepgram connection closed unexpectedly")
                    # TODO: We should try to reconnect here
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logging.warning("unexpected deepgram message type %s", msg.type)
                    continue

                try:
                    # received a message from deepgram
                    data = json.loads(msg.data)
                    self._process_stream_event(data)
                except Exception as e:
                    logging.error(f"failed to process deepgram message: {e}")

        await asyncio.gather(send_task(), recv_task())


    def _process_stream_event(self, data: dict) -> None:
        # https://developers.deepgram.com/docs/speech-started
        if data["type"] == "SpeechStarted":
            start_event = stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH
            )
            self._event_queue.put_nowait(start_event)
            return

        # see this page https://developers.deepgram.com/docs/understand-endpointing-interim-results#using-endpointing-speech_final
        # for more information about the different types of events
        if data["type"] == "Results":
            alts = data["channel"]["alternatives"]

            if data["speech_final"]: # end of speech
                if len(self._final_events) == 0:
                    logging.warning("received end of speech without any final transcription")
                    return

                # combine all final transcripts since the start of the speech
                sentence = ""
                confidence = 0
                self._final_events = []
                for alt in self._final_events:
                    sentence += alt.alternatives[0].text
                    confidence += alt.alternatives[0].confidence

                confidence /= len(self._final_events)

                end_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    alternatives=[
                        stt.SpeechData(   
                            language=self._config.language,
                            start_time=self._final_events[0].alternatives[0].start_time,
                            end_time=self._final_events[-1].alternatives[0].end_time,
                            confidence=confidence,
                            text=sentence,
                        )
                    ]
                )
                self._event_queue.put_nowait(end_event)
                self._final_events = []
            elif data["is_final"]:
                # final transcription of a segment
                alts = live_transcription_to_speech_data(self._config.language, data)
                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alts,
                )
                self._final_events.append(final_event)
                self._event_queue.put_nowait(final_event)
            else:
                # interim transcription
                alts = live_transcription_to_speech_data(self._config.language, data)
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alts,
                )
                self._event_queue.put_nowait(interim_event)

            return

        logging.warning("skipping non-results message %s", data)
       
    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()


def live_transcription_to_speech_data(
    language: str,
    data: dict,
) -> List[stt.SpeechData]:
    dg_alts = data["channel"]["alternatives"]

    return [
        stt.SpeechData(
            language=language,
            start_time=alt["words"][0]["start"] if alt["words"] else 0,
            end_time=alt["words"][-1]["end"] if alt["words"] else 0,
            confidence=alt["confidence"],
            text=alt["transcript"],
        )
        for alt in dg_alts
    ]

def prerecorded_transcription_to_speech_event(
    language: Optional[str], # language should be None when 'detect_language' is enabled
    event: dict,
) -> stt.SpeechEvent:
    # We only support one channel for now
    channel = event["results"]["channels"][0]
    dg_alts = channel["alternatives"]

    # Use the detected language if enabled
    # https://developers.deepgram.com/docs/language-detection
    detected_language = channel.get("detected_language")

    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=language or detected_language,
                start_time=alt["words"][0]["start"] if alt["words"] else 0,
                end_time=alt["words"][-1]["end"] if alt["words"] else 0,
                confidence=alt["confidence"],
                text=alt["transcript"],
            )
            for alt in dg_alts
        ],
    )
