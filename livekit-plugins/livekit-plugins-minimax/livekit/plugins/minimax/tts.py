from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, cast

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .log import logger

TTSModel = Literal[
    "speech-2.5-hd-preview",
    "speech-2.5-turbo-preview",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
]

# Minimax TTS Voice IDs
# Defines small part of supported voices using a Literal type for static analysis.
# See more voices in docs of Minimax
TTSVoice = Literal[
    "voice_agent_Female_Phone_4",
    "voice_agent_Male_Phone_1",
    "English_StressedLady",
    "English_SentimentalLady",
    "English_WiseScholar",
    "English_radiant_girl",
    "moss_audio_84f32de9-2363-11f0-b7ab-d255fae1f27b",
    "japanese_male_social_media_1_v2",
    "japanese_female_social_media_1_v2",
    "French_CasualMan",
    "French_Female Journalist",
    "Spanish_Narrator",
    "Spanish_WiseScholar",
    "Spanish_ThoughtfulMan",
    "Arabic_CalmWoman",
    "Arabic_FriendlyGuy",
    "Portuguese_ThoughtfulLady",
    "German_PlayfulMan",
    "German_SweetLady",
]

DEFAULT_MODEL = "speech-02-turbo"
DEFAULT_VOICE_ID = "English_radiant_girl"


TTSEmotion = Literal["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]


TTSAudioFormat = Literal["pcm", "mp3", "flac", "wav"]
TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 44100]
TTSBitRate = Literal[32000, 64000, 128000, 256000]  # only for mp3 format

DEFAULT_BASE_URL = "https://api.minimaxi.io"  # or "https://api.minimaxi.com"


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: TTSModel | str
    voice_id: TTSVoice | str
    sample_rate: TTSSampleRate
    bitrate: TTSBitRate
    emotion: TTSEmotion | None
    speed: float  # [0.5, 2.0]
    vol: float  # (0, 10]
    pitch: int  # [-12, 12]
    english_normalization: bool
    pronunciation_dict: dict[str, list[str]] | None
    # voice_modify
    intensity: int | None
    timbre: int | None
    audio_format: TTSAudioFormat


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModel | str = DEFAULT_MODEL,
        voice: TTSVoice | str = DEFAULT_VOICE_ID,
        emotion: TTSEmotion | None = None,
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: int = 0,
        english_normalization: bool = False,
        audio_format: TTSAudioFormat = "mp3",
        pronunciation_dict: dict[str, list[str]] | None = None,
        intensity: int | None = None,
        timbre: int | None = None,
        sample_rate: TTSSampleRate = 24000,
        bitrate: TTSBitRate = 128000,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Minimax TTS plugin

        Args:
            model (TTSModel | str, optional): The Minimax TTS model to use. Defaults to DEFAULT_MODEL.
            voice (TTSVoice | str, optional): The voice to use. Defaults to DEFAULT_VOICE_ID.
            emotion (TTSEmotion | None, optional): Emotion control for speech synthesis. Defaults to None.
            speed (float, optional): Speech speed, higher values speak faster. Range is [0.5, 2.0].
            vol (float, optional): Speech volume, range is [0, 10].
            pitch (int, optional): Speech pitch adjustment, range is [-12, 12].
            english_normalization (bool, optional): Enable text normalization in English. Improves performance
                in digit-reading scenarios at the cost of slightly higher latency. Defaults to False.
            audio_format (TTSAudioFormat, optional): The audio format to use. Defaults to "mp3".
            pronunciation_dict (dict[str, list[str]] | None, optional): Defines pronunciation rules for specific characters or symbols.
            intensity (int | None, optional): Corresponds to the "Strong/Softer" slider on the official page. Range [-100, 100].
            timbre (int | None, optional): Corresponds to the "Nasal/Crisp" slider on the official page. Range: [-100, 100].
            sample_rate (TTSSampleRate, optional): The audio sample rate in Hz. Defaults to 24000.
            bitrate (TTSBitRate, optional): The audio bitrate in kbps. Defaults to 128000.
            tokenizer (NotGivenOr[tokenize.SentenceTokenizer], optional): The sentence tokenizer to use. Defaults to NOT_GIVEN.
            text_pacing (tts.SentenceStreamPacer | bool, optional): Enable text pacing for sentence-level timing control. Defaults to False.
            api_key (str | None, optional): The Minimax API key. Defaults to None.
            base_url (NotGivenOr[str], optional): The base URL for the Minimax API. Defaults to NOT_GIVEN.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        base_url = (
            base_url
            if utils.is_given(base_url)
            else os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL)
        )

        minimax_api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not minimax_api_key:
            raise ValueError("MINIMAX_API_KEY must be set")

        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"speed must be between 0.5 and 2.0, but got {speed}")
        if intensity is not None and not (-100 <= intensity <= 100):
            raise ValueError(f"intensity must be between -100 and 100, but got {intensity}")
        if timbre is not None and not (-100 <= timbre <= 100):
            raise ValueError(f"timbre must be between -100 and 100, but got {timbre}")

        self._sentence_tokenizer = (
            tokenizer if utils.is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice,
            api_key=minimax_api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            emotion=emotion,
            bitrate=bitrate,
            speed=speed,
            pitch=pitch,
            vol=vol,
            english_normalization=english_normalization,
            timbre=timbre,
            pronunciation_dict=pronunciation_dict,
            intensity=intensity,
            audio_format=audio_format,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MiniMax"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModel | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoice | str] = NOT_GIVEN,
        emotion: NotGivenOr[TTSEmotion | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        vol: NotGivenOr[float] = NOT_GIVEN,
        pitch: NotGivenOr[int] = NOT_GIVEN,
        english_normalization: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[TTSAudioFormat] = NOT_GIVEN,
        pronunciation_dict: NotGivenOr[dict[str, list[str]]] = NOT_GIVEN,
        intensity: NotGivenOr[int] = NOT_GIVEN,
        timbre: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Update the TTS configuration options."""
        if utils.is_given(model):
            self._opts.model = model

        if utils.is_given(voice):
            self._opts.voice_id = voice

        if utils.is_given(emotion):
            self._opts.emotion = cast(Optional[TTSEmotion], emotion)

        if utils.is_given(speed):
            self._opts.speed = speed

        if utils.is_given(vol):
            self._opts.vol = vol

        if utils.is_given(pitch):
            self._opts.pitch = pitch

        if utils.is_given(english_normalization):
            self._opts.english_normalization = english_normalization

        if utils.is_given(audio_format):
            self._opts.audio_format = cast(TTSAudioFormat, audio_format)

        if utils.is_given(pronunciation_dict):
            self._opts.pronunciation_dict = pronunciation_dict

        if utils.is_given(intensity):
            self._opts.intensity = intensity

        if utils.is_given(timbre):
            self._opts.timbre = timbre

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url = self._opts.base_url
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)
        url = f"{url}/ws/v1/t2a_v2"

        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=f"audio/{self._opts.audio_format}",
            stream=True,
        )

        sentence_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sentence_stream = self._tts._stream_pacer.wrap(
                sent_stream=sentence_stream,
                audio_emitter=output_emitter,
            )

        task_started = asyncio.Future[None]()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue

                sentence_stream.push_text(data)
            sentence_stream.end_input()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            start_msg = _to_minimax_options(self._opts)
            start_msg["event"] = "task_start"
            await ws.send_str(json.dumps(start_msg))

            try:
                await asyncio.wait_for(task_started, self._conn_options.timeout)
            except asyncio.TimeoutError as e:
                raise APITimeoutError("task_start timed out") from e

            async for sentence in sentence_stream:
                self._mark_started()
                await ws.send_str(json.dumps({"event": "task_continue", "text": sentence.token}))

            await ws.send_str(json.dumps({"event": "task_finish"}))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Minimax connection closed unexpectedly", request_id=request_id
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Minimax message type %s", msg.type)
                    continue

                data: dict[str, Any] = json.loads(msg.data)
                status_code = data.get("base_resp", {}).get("status_code")
                if status_code != 0:
                    raise APIStatusError(
                        f"Minimax returned non-zero status code: {status_code}",
                        request_id=request_id,
                        body=data,
                    )

                if data.get("event") == "connected_success":
                    pass

                elif data.get("event") == "task_started":
                    task_started.set_result(None)
                    session_id = data.get("session_id", "")
                    output_emitter.start_segment(segment_id=session_id)

                elif data.get("event") == "task_continued":
                    audio = data.get("data", {}).get("audio")
                    if audio:
                        output_emitter.push(bytes.fromhex(audio))
                    if data.get("is_final"):
                        output_emitter.flush()

                elif data.get("event") == "task_finished":
                    output_emitter.end_input()
                    break

                elif data.get("event") == "task_failed":
                    raise APIError(f"Minimax returned task failed: {msg.data}")

                else:
                    logger.warning(f"unexpected Minimax message: {msg.data}")

        try:
            ws = await self._tts._connect_ws(self._conn_options.timeout)
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_sentence_stream_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await self._tts._close_ws(ws)
                await sentence_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    async def aclose(self) -> None:
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self._input_text.strip():
            return

        url = self._opts.base_url + "/v1/t2a_v2"
        msg = _to_minimax_options(self._opts)
        msg.update(
            {
                "text": self._input_text,
                "stream": True,
                "stream_options": {
                    "exclude_aggregated_audio": True,  # don't include complete audio in last chunk
                },
            }
        )
        try:
            async with self._tts._ensure_session().post(
                url,
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                },
                json=msg,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=f"audio/{self._opts.audio_format}",
                )

                async for chunk in resp.content:
                    line = chunk.decode().strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        logger.warning("unexpected Minimax message: %s", line)
                        continue

                    data = json.loads(line[5:])
                    if audio := data.get("data", {}).get("audio"):
                        output_emitter.push(bytes.fromhex(audio))
                    elif (status_code := data.get("base_resp", {}).get("status_code")) != 0:
                        raise APIStatusError(
                            f"Minimax returned non-zero status code: {status_code}",
                            request_id=None,
                            body=data,
                        )
                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _to_minimax_options(opts: _TTSOptions) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": opts.model,
        "voice_setting": {
            "voice_id": opts.voice_id,
            "speed": opts.speed,
            "vol": opts.vol,
            "pitch": opts.pitch,
        },
        "audio_setting": {
            "sample_rate": opts.sample_rate,
            "bitrate": opts.bitrate,
            "format": opts.audio_format,
            "channel": 1,
        },
    }

    if opts.emotion is not None:
        config["voice_setting"]["emotion"] = opts.emotion

    if opts.pronunciation_dict:
        config["pronunciation_dict"] = opts.pronunciation_dict

    voice_modify: dict[str, Any] = {}
    if opts.intensity is not None:
        voice_modify["intensity"] = opts.intensity
    if opts.timbre is not None:
        voice_modify["timbre"] = opts.timbre

    if voice_modify:
        config["voice_modify"] = voice_modify

    return config
