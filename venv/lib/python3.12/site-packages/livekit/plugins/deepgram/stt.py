import asyncio
import dataclasses
import io
import json
import logging
import os
from urllib.parse import urlencode
import wave
from dataclasses import dataclass
from typing import Optional, Union

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from .models import DeepgramLanguages, DeepgramModels

STREAM_KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
STREAM_CLOSE_MSG: str = json.dumps({"type": "CloseStream"})


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
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            raise ValueError("Deepgram API key is required")

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

    def _config_to_query(self, config: STTOptions) -> str:
        params = {
            "model": config.model,
            "punctuate": config.punctuate,
            "detect_language": config.detect_language,
            "smart_format": config.smart_format,
        }
        if config.language:
            params["language"] = config.language
        return urlencode(params).lower()

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[Union[DeepgramLanguages, str]] = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        query_params = self._config_to_query(config)
        url = f"https://api.deepgram.com/v1/listen?{query_params}"
        # Deepgram prerecorded API requires WAV/MP3, so we write our PCM into a wav buffer
        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        async with aiohttp.ClientSession(
            headers={
                "Authorization": f"Token {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "audio/wav",
            }
        ) as session:
            async with session.post(
                url=url,
                data=io_buffer.getvalue(),
            ) as res:
                json_res = await res.json()
                return prerecorded_transcription_to_speech_event(
                    config.language, json_res
                )

    def stream(
        self,
        *,
        language: Optional[Union[DeepgramLanguages, str]] = None,
    ) -> "SpeechStream":
        config = self._sanitize_options(language=language)
        return SpeechStream(
            config,
            api_key=self._api_key,
        )


class SpeechStream(stt.SpeechStream):
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

        self._queue = asyncio.Queue()
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

        self._queue.put_nowait(
            frame.remix_and_resample(self._sample_rate, self._num_channels)
        )

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        await self._queue.put(STREAM_CLOSE_MSG)
        await self._main_task

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Deepgram with exponential backoff and forward frames"""
        async with aiohttp.ClientSession() as session:
            retry_count = 0
            ws: Optional[aiohttp.ClientWebSocketResponse] = None
            listen_task: Optional[asyncio.Task] = None
            keepalive_task: Optional[asyncio.Task] = None
            while True:
                try:
                    ws = await self._try_connect(session)
                    listen_task = asyncio.create_task(self._listen_loop(ws))
                    keepalive_task = asyncio.create_task(self._keepalive_loop(ws))
                    # break out of the retry loop if we are done
                    if await self._send_loop(ws):
                        keepalive_task.cancel()
                        await asyncio.wait_for(listen_task, timeout=5)
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

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        while not ws.closed:
            data = await self._queue.get()
            # fire and forget, we don't care if we miss frames in the error case
            self._queue.task_done()

            if ws.closed:
                raise Exception("websocket closed")

            if isinstance(data, rtc.AudioFrame):
                await ws.send_bytes(data.data.tobytes())
            else:
                if data == STREAM_CLOSE_MSG:
                    await ws.send_str(data)
                    return True
        return False

    async def _keepalive_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            await ws.send_str(STREAM_KEEPALIVE_MSG)
            await asyncio.sleep(5)

    async def _listen_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

            try:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data["type"] != "Results":
                        logging.warning("Skipping non-results message %s", data)
                        continue
                    stt_event = live_transcription_to_speech_event(
                        self._config.language, data
                    )
                    await self._event_queue.put(stt_event)
                    continue
            except Exception as e:
                logging.error("Error handling message %s: %s", msg, e)
                continue

            logging.warning("Unhandled message %s", msg)

    async def _try_connect(
        self, session: aiohttp.ClientSession
    ) -> aiohttp.ClientWebSocketResponse:
        live_config = {
            "model": self._config.model,
            "punctuate": self._config.punctuate,
            "smart_format": self._config.smart_format,
            "interim_results": self._config.interim_results,
            "encoding": "linear16",
            "sample_rate": self._sample_rate,
            "channels": self._num_channels,
            "endpointing": str(self._config.endpointing or "10"),
        }

        if self._config.language:
            live_config["language"] = self._config.language

        query_params = urlencode(live_config).lower()

        url = f"wss://api.deepgram.com/v1/listen?{query_params}"
        ws = await session.ws_connect(
            url, headers={"Authorization": f"Token {self._api_key}"}
        )

        return ws

    async def __anext__(self) -> stt.SpeechEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()


def live_transcription_to_speech_event(
    language: Optional[str],
    event: dict,
) -> stt.SpeechEvent:
    try:
        dg_alts = event["channel"]["alternatives"]
    except KeyError:
        raise ValueError("no alternatives in response")

    return stt.SpeechEvent(
        is_final=event["is_final"] or False,  # could be None?
        end_of_speech=event["speech_final"] or False,
        alternatives=[
            stt.SpeechData(
                language=language or "",
                start_time=(alt["words"][0]["start"] if alt["words"] else 0) or 0,
                end_time=(alt["words"][-1]["end"] if alt["words"] else 0) or 0,
                confidence=alt["confidence"] or 0,
                text=alt["transcript"] or "",
            )
            for alt in dg_alts
        ],
    )


def prerecorded_transcription_to_speech_event(
    language: Optional[str],
    event: dict,
) -> stt.SpeechEvent:
    try:
        dg_alts = event["results"]["channels"][0]["alternatives"]
    except KeyError:
        raise ValueError("no alternatives in response")

    return stt.SpeechEvent(
        is_final=True,
        end_of_speech=True,
        alternatives=[
            stt.SpeechData(
                language=language or "",
                start_time=(alt["words"][0]["start"] if alt["words"] else 0) or 0,
                end_time=(alt["words"][-1]["end"] if alt["words"] else 0) or 0,
                confidence=alt["confidence"] or 0,
                # not sure why transcript is Optional inside DG SDK ...
                text=alt["transcript"] or "",
            )
            for alt in dg_alts
        ],
    )
