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
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-2.5-hd-preview",
    "speech-2.5-turbo-preview",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
]

# Minimax TTS Voice IDs
# Defines commonly used voices for static analysis.
# See full voice list in Minimax documentation
TTSVoice = Literal[
    # Social Media Voices
    "socialmedia_female_2_v1",
    "socialmedia_female_1_v1",
    # Voice Agent Series
    "voice_agent_Female_Phone_4",
    "voice_agent_Male_Phone_1",
    "voice_agent_Male_Phone_2",
    # English Voices - Female
    "English_StressedLady",
    "English_SentimentalLady",
    "English_radiant_girl",
    # English Voices - Male
    "English_WiseScholar",
    "English_Persuasive_Man",
    "English_Explanatory_Man",
    "English_Insightful_Speaker",
    # Japanese Voices
    "japanese_male_social_media_1_v2",
    "japanese_female_social_media_1_v2",
    # French Voices
    "French_CasualMan",
    "French_Female Journalist",
    # Spanish Voices
    "Spanish_Narrator",
    "Spanish_WiseScholar",
    "Spanish_ThoughtfulMan",
    # Arabic Voices
    "Arabic_CalmWoman",
    "Arabic_FriendlyGuy",
    # Portuguese Voices
    "Portuguese_ThoughtfulLady",
    # German Voices
    "German_PlayfulMan",
    "German_SweetLady",
    # MOSS Audio Series
    "moss_audio_7c7e7ae2-7356-11f0-9540-7ef9b4b62566",
    "moss_audio_b118f320-78c0-11f0-bbeb-26e8167c4779",
    "moss_audio_84f32de9-2363-11f0-b7ab-d255fae1f27b",
    "moss_audio_82ebf67c-78c8-11f0-8e8e-36b92fbb4f95",
]

DEFAULT_MODEL = "speech-02-turbo"
DEFAULT_VOICE_ID = "socialmedia_female_2_v1"


# Note: "fluent" emotion is only supported by speech-2.6-* models
TTSEmotion = Literal[
    "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "fluent"
]

TTSLanguageBoost = Literal[
    "auto",
    "Chinese",
    "Chinese,Yue",
    "English",
    "Arabic",
    "Russian",
    "Spanish",
    "French",
    "Portuguese",
    "German",
    "Turkish",
    "Dutch",
    "Ukrainian",
    "Vietnamese",
    "Indonesian",
    "Japanese",
    "Italian",
    "Korean",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
    "Bulgarian",
    "Danish",
    "Hebrew",
    "Malay",
    "Persian",
    "Slovak",
    "Swedish",
    "Croatian",
    "Filipino",
    "Hungarian",
    "Norwegian",
    "Slovenian",
    "Catalan",
    "Nynorsk",
    "Tamil",
    "Afrikaans",
]

TTSAudioFormat = Literal["pcm", "mp3", "flac", "wav"]
TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 44100]
TTSBitRate = Literal[32000, 64000, 128000, 256000]  # only for mp3 format

DEFAULT_BASE_URL = (
    "https://api-uw.minimax.io"  # or "https://api.minimaxi.chat or https://api.minimax.io"
)


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
    text_normalization: bool
    pronunciation_dict: dict[str, list[str]] | None
    language_boost: TTSLanguageBoost | None
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
        text_normalization: bool = False,
        audio_format: TTSAudioFormat = "mp3",
        pronunciation_dict: dict[str, list[str]] | None = None,
        intensity: int | None = None,
        timbre: int | None = None,
        language_boost: TTSLanguageBoost | None = None,
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
                Available models: speech-2.6-hd, speech-2.6-turbo, speech-2.5-hd-preview,
                speech-2.5-turbo-preview, speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo.
            voice (TTSVoice | str, optional): The voice to use. Defaults to DEFAULT_VOICE_ID.
            emotion (TTSEmotion | None, optional): Emotion control for speech synthesis.
                Options: "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "fluent".
                Note: "fluent" emotion is only supported by speech-2.6-* models. Defaults to None.
            speed (float, optional): Speech speed, higher values speak faster. Range is [0.5, 2.0].
            vol (float, optional): Speech volume, range is [0, 10].
            pitch (int, optional): Speech pitch adjustment, range is [-12, 12].
            text_normalization (bool, optional): Enable text normalization (Chinese/English). Improves performance
                in digit-reading scenarios at the cost of slightly higher latency. Defaults to False.
            audio_format (TTSAudioFormat, optional): The audio format to use. Defaults to "mp3".
            pronunciation_dict (dict[str, list[str]] | None, optional): Defines pronunciation rules for specific characters or symbols.
            intensity (int | None, optional): Corresponds to the "Strong/Softer" slider on the official page. Range [-100, 100].
            language_boost(TTSLanguageBoost | None, optional): Controls whether recognition for specific minority languages and dialects is enhanced. Defaults to None.
            timbre (int | None, optional): Corresponds to the "Nasal/Crisp" slider on the official page. Range: [-100, 100].
            sample_rate (TTSSampleRate, optional): The audio sample rate in Hz. Defaults to 24000.
            bitrate (TTSBitRate, optional): The audio bitrate in kbps. Defaults to 128000.
            tokenizer (NotGivenOr[tokenize.SentenceTokenizer], optional): The sentence tokenizer to use. Defaults to `livekit.agents.tokenize.basic.SentenceTokenizer`.
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

        # Validate fluent emotion is only used with speech-2.6-* models
        if emotion == "fluent" and not model.startswith("speech-2.6"):
            raise ValueError(
                f'"fluent" emotion is only supported by speech-2.6-* models, '
                f'but got model "{model}". Please use speech-2.6-hd or speech-2.6-turbo.'
            )

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
            text_normalization=text_normalization,
            timbre=timbre,
            language_boost=language_boost,
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
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[TTSAudioFormat] = NOT_GIVEN,
        pronunciation_dict: NotGivenOr[dict[str, list[str]]] = NOT_GIVEN,
        intensity: NotGivenOr[int] = NOT_GIVEN,
        timbre: NotGivenOr[int] = NOT_GIVEN,
        language_boost: NotGivenOr[TTSLanguageBoost | None] = NOT_GIVEN,
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

        if utils.is_given(text_normalization):
            self._opts.text_normalization = text_normalization

        if utils.is_given(audio_format):
            self._opts.audio_format = cast(TTSAudioFormat, audio_format)

        if utils.is_given(pronunciation_dict):
            self._opts.pronunciation_dict = pronunciation_dict

        if utils.is_given(intensity):
            self._opts.intensity = intensity

        if utils.is_given(timbre):
            self._opts.timbre = timbre

        if utils.is_given(language_boost):
            self._opts.language_boost = cast(Optional[TTSLanguageBoost], language_boost)

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

        # Log WebSocket connection establishment
        logger.debug(f"MiniMax WebSocket connected to {url}")

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
        # Initialize with temporary ID, will be updated from WebSocket messages
        request_id = utils.shortuuid()
        trace_id = request_id  # Use trace_id directly instead of creating a dict

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
            # Initialize trace_id to ensure it's available in all code paths
            current_trace_id = trace_id

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    error_msg = (
                        f"MiniMax connection closed unexpectedly (trace_id: {current_trace_id})"
                    )
                    logger.error(error_msg)
                    raise APIStatusError(error_msg, request_id=current_trace_id)

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Minimax message type %s", msg.type)
                    continue

                data: dict[str, Any] = json.loads(msg.data)

                # Extract trace_id (priority: root.trace_id > base_resp.trace_id)
                # api.minimax.io returns trace_id in root.trace_id, api.minimaxi.com may return in base_resp.trace_id
                msg_trace_id = data.get("trace_id") or data.get("base_resp", {}).get("trace_id")
                if msg_trace_id and msg_trace_id != current_trace_id:
                    current_trace_id = msg_trace_id
                    logger.debug(f"MiniMax WebSocket trace_id updated: {msg_trace_id}")

                base_resp = data.get("base_resp", {})
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    status_msg = base_resp.get("status_msg", "Unknown error")
                    error_trace_id = msg_trace_id or current_trace_id

                    logger.error(
                        f"MiniMax WebSocket error: code={status_code}, msg={status_msg}, trace_id={error_trace_id}",
                        extra={"request_id": request_id, "full_response": data},
                    )

                    raise APIStatusError(
                        f"MiniMax error [{status_code}]: {status_msg} (trace_id: {error_trace_id})",
                        request_id=error_trace_id,
                        body=data,
                    )

                if data.get("event") == "connected_success":
                    logger.debug(f"MiniMax WebSocket connected, trace_id={current_trace_id}")

                elif data.get("event") == "task_started":
                    task_started.set_result(None)
                    session_id = data.get("session_id", "")
                    logger.debug(
                        f"MiniMax WebSocket task_started, session_id={session_id}, trace_id={current_trace_id}"
                    )
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
                    error_msg = (
                        f"MiniMax returned task failed (trace_id: {current_trace_id}): {msg.data}"
                    )
                    logger.error(error_msg)
                    raise APIError(error_msg)

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
            logger.error(
                f"MiniMax WebSocket request timeout after {self._conn_options.timeout}s, trace_id={trace_id}"
            )
            raise APITimeoutError(
                f"WebSocket TTS synthesis timed out after {self._conn_options.timeout}s (trace_id: {trace_id})"
            ) from None
        except aiohttp.ClientResponseError as e:
            logger.error(
                f"WebSocket HTTP error: status={e.status}, message={e.message}, trace_id={trace_id}",
                exc_info=True,
            )
            raise APIStatusError(
                message=f"WebSocket HTTP {e.status}: {e.message} (trace_id: {trace_id})",
                status_code=e.status,
                request_id=trace_id,
                body=None,
            ) from e
        except Exception as e:
            if not isinstance(e, (APIStatusError, APITimeoutError, APIConnectionError)):
                logger.error(
                    f"MiniMax WebSocket unexpected error: {type(e).__name__}: {e}, trace_id={trace_id}",
                    exc_info=True,
                )
            raise APIConnectionError(
                f"WebSocket connection failed: {type(e).__name__}: {e} (trace_id: {trace_id})"
            ) from e

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
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                # Extract trace_id from response headers (all requests have this)
                # Note: api.minimax.io also returns trace_id in response body root.trace_id
                trace_id = resp.headers.get("Trace-Id") or resp.headers.get("X-Trace-Id")
                minimax_request_id = resp.headers.get("Minimax-Request-Id")

                if trace_id:
                    logger.debug(
                        f"MiniMax HTTP stream request started, trace_id={trace_id}, minimax_request_id={minimax_request_id}"
                    )
                else:
                    trace_id = utils.shortuuid()
                    logger.warning(
                        f"No Trace-Id in response headers, using generated ID: {trace_id}"
                    )

                output_emitter.initialize(
                    request_id=trace_id,
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

                    # api.minimax.io returns trace_id in response body root level
                    body_trace_id = data.get("trace_id")
                    if body_trace_id and body_trace_id != trace_id:
                        logger.debug(f"Found trace_id in response body: {body_trace_id}")

                    if audio := data.get("data", {}).get("audio"):
                        output_emitter.push(bytes.fromhex(audio))
                    else:
                        base_resp = data.get("base_resp", {})
                        status_code = base_resp.get("status_code", 0)
                        if status_code != 0:
                            status_msg = base_resp.get("status_msg", "Unknown error")
                            # trace_id priority: response body top level > response headers
                            error_trace_id = body_trace_id or trace_id

                            logger.error(
                                f"MiniMax HTTP stream error: code={status_code}, msg={status_msg}, trace_id={error_trace_id}",
                                extra={"full_response": data},
                            )

                            raise APIStatusError(
                                f"MiniMax error [{status_code}]: {status_msg} (trace_id: {error_trace_id})",
                                request_id=error_trace_id,
                                body=data,
                            )
                output_emitter.flush()

        except asyncio.TimeoutError:
            logger.error(f"Minimax HTTP stream request timeout after {self._conn_options.timeout}s")
            raise APITimeoutError(
                f"TTS synthesis timed out after {self._conn_options.timeout}s"
            ) from None
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error: status={e.status}, message={e.message}", exc_info=True)
            raise APIStatusError(
                message=f"HTTP {e.status}: {e.message}",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            if not isinstance(e, (APIStatusError, APITimeoutError, APIConnectionError)):
                logger.error(
                    f"Minimax TTS unexpected error: {type(e).__name__}: {e}", exc_info=True
                )
            raise APIConnectionError(f"Connection failed: {type(e).__name__}: {e}") from e


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
        "text_normalization": opts.text_normalization,
    }

    if opts.emotion is not None:
        config["voice_setting"]["emotion"] = opts.emotion

    if opts.language_boost is not None:
        config["language_boost"] = opts.language_boost

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
