from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr

TTSModel = Literal[
    "speech-2.8-hd",
    "speech-2.8-turbo",
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

    def validate(self) -> None:
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(f"speed must be between 0.5 and 2.0, but got {self.speed}")
        if self.intensity is not None and not (-100 <= self.intensity <= 100):
            raise ValueError(f"intensity must be between -100 and 100, but got {self.intensity}")
        if self.timbre is not None and not (-100 <= self.timbre <= 100):
            raise ValueError(f"timbre must be between -100 and 100, but got {self.timbre}")
        if self.emotion == "fluent" and not self.model.startswith("speech-2.6"):
            raise ValueError(
                f'"fluent" emotion is only supported by speech-2.6-* models, '
                f'but got model "{self.model}". Please use speech-2.6-hd or speech-2.6-turbo.'
            )


# MiniMax's request options, shared by the HTTP body and the websocket's task_start


class _VoiceSetting(TypedDict):
    voice_id: str
    speed: float
    vol: float
    pitch: int
    emotion: NotRequired[str]


class _AudioSetting(TypedDict):
    sample_rate: int
    bitrate: int
    format: str
    channel: int


class _VoiceModify(TypedDict, total=False):
    intensity: int
    timbre: int


class _Options(TypedDict):
    model: str
    voice_setting: _VoiceSetting
    audio_setting: _AudioSetting
    text_normalization: bool
    language_boost: NotRequired[str]
    pronunciation_dict: NotRequired[dict[str, list[str]]]
    voice_modify: NotRequired[_VoiceModify]


class _StreamOptions(TypedDict):
    exclude_aggregated_audio: bool


class _HttpRequest(_Options):
    text: str
    stream: bool
    stream_options: _StreamOptions


class _TaskStart(_Options):
    event: Literal["task_start"]


class _TaskContinue(TypedDict):
    event: Literal["task_continue"]
    text: str


class _TaskFinish(TypedDict):
    event: Literal["task_finish"]


_Command = _TaskStart | _TaskContinue | _TaskFinish


# MiniMax's answers, on the websocket and the HTTP stream: each declares only the fields read.


class _BaseResp(TypedDict, total=False):
    status_code: int
    status_msg: str
    trace_id: str


class _AudioData(TypedDict, total=False):
    audio: str


class _ExtraInfo(TypedDict, total=False):
    usage_characters: int


class _Answer(TypedDict, total=False):
    trace_id: str
    base_resp: _BaseResp | None
    data: _AudioData | None
    is_final: bool
    extra_info: _ExtraInfo | None


class _Connected(_Answer):
    event: Literal["connected_success"]


class _Started(_Answer):
    event: Literal["task_started"]


class _Continued(_Answer):
    event: Literal["task_continued"]


class _Finished(_Answer):
    event: Literal["task_finished"]


class _Failed(_Answer):
    event: Literal["task_failed"]


_Message = _Connected | _Started | _Continued | _Finished | _Failed
_HTTP_ANSWER = TypeAdapter(_Answer)


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
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
    ):
        """Minimax TTS plugin

        Args:
            model (TTSModel | str, optional): The Minimax TTS model to use. Defaults to DEFAULT_MODEL.
                Available models: speech-2.8-hd, speech-2.8-turbo, speech-2.6-hd, speech-2.6-turbo,
                speech-2.5-hd-preview, speech-2.5-turbo-preview, speech-02-hd, speech-02-turbo,
                speech-01-hd, speech-01-turbo.
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
            language_boost (TTSLanguageBoost | None, optional): Controls whether recognition for specific minority languages and dialects is enhanced. Defaults to None.
            timbre (int | None, optional): Corresponds to the "Nasal/Crisp" slider on the official page. Range: [-100, 100].
            sample_rate (TTSSampleRate, optional): The audio sample rate in Hz. Defaults to 24000.
            bitrate (TTSBitRate, optional): The audio bitrate in kbps. Defaults to 128000.
            api_key (str | None, optional): The Minimax API key. Defaults to None.
            base_url (NotGivenOr[str], optional): The base URL for the Minimax API. Defaults to NOT_GIVEN.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, the job's shared session is used.
        """  # noqa: E501
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
            raise ValueError(
                "MiniMax API key is required, either as argument or set"
                " MINIMAX_API_KEY environment variable"
            )

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
        self._opts.validate()

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
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update the TTS configuration options; an invalid combination is refused and changes
        nothing."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            opts = replace(self._opts)
            if utils.is_given(model):
                opts.model = model
            if utils.is_given(voice):
                opts.voice_id = voice
            if utils.is_given(emotion):
                opts.emotion = emotion
            if utils.is_given(speed):
                opts.speed = speed
            if utils.is_given(vol):
                opts.vol = vol
            if utils.is_given(pitch):
                opts.pitch = pitch
            if utils.is_given(text_normalization):
                opts.text_normalization = text_normalization
            if utils.is_given(audio_format):
                opts.audio_format = audio_format
            if utils.is_given(pronunciation_dict):
                opts.pronunciation_dict = pronunciation_dict
            if utils.is_given(intensity):
                opts.intensity = intensity
            if utils.is_given(timbre):
                opts.timbre = timbre
            if utils.is_given(language_boost):
                opts.language_boost = language_boost
            opts.validate()
        self._opts = opts

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self, opts=replace(self._opts), input_text=chunk.text, conn_options=conn_options
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(opts=replace(self._opts), tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """Each segment is one MiniMax task on its own connection: ``task_start`` (its setup),
    a ``task_continue`` per text, ``task_finish`` at its end, answered by ``task_finished``."""

    connection_per_segment = True
    awaits_ready = True

    def __init__(self, *, opts: _TTSOptions, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self.mime_type = f"audio/{opts.audio_format}"
        self._segment = ""

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        url = self._opts.base_url.replace("http", "ws", 1)
        return tts.WebSocketTarget(
            f"{url}/ws/v1/t2a_v2", headers={"Authorization": f"Bearer {self._opts.api_key}"}
        )

    def setup(self) -> _TaskStart:
        return {**_to_minimax_options(self._opts), "event": "task_start"}

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._segment = item.segment_id
                return None
            case tts.SegmentText():
                return _TaskContinue(event="task_continue", text=item.chunk.text)
            case tts.SegmentEnd():
                return _TaskFinish(event="task_finish")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        trace_id = message.get("trace_id") or (message.get("base_resp") or {}).get("trace_id")
        if trace_id and trace_id != self.request_id:
            self.report(tts.ProviderRequestId(trace_id))
        _raise_status(message, trace_id)
        if message["event"] == "connected_success":
            pass
        elif message["event"] == "task_started":
            self.report(tts.Ready())
        elif message["event"] == "task_continued":
            if audio := (message.get("data") or {}).get("audio"):
                self.report(tts.SegmentAudio(self._segment, bytes.fromhex(audio)))
            if (billed := (message.get("extra_info") or {}).get("usage_characters")) is not None:
                self.report(tts.SegmentUsage(self._segment, characters_count=billed))
        elif message["event"] == "task_finished":
            self.report(tts.SegmentDone(self._segment))
        elif message["event"] == "task_failed":
            raise APIError(
                f"MiniMax returned task failed (trace_id: {trace_id})",
                kind="invalid_request",
                body=message,
            )
        else:
            self._skip_unknown_message(message)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, opts: _TTSOptions, input_text: str, conn_options: APIConnectOptions
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self._input_text.strip():
            return

        body: _HttpRequest = {
            **_to_minimax_options(self._opts),
            "text": self._input_text,
            "stream": True,
            # the last chunk does not repeat the whole audio
            "stream_options": {"exclude_aggregated_audio": True},
        }
        try:
            async with self._tts._ensure_session().post(
                self._opts.base_url + "/v1/t2a_v2",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                json=body,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()
                trace_id = resp.headers.get("Trace-Id") or resp.headers.get("X-Trace-Id")
                output_emitter.initialize(
                    request_id=trace_id or utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=f"audio/{self._opts.audio_format}",
                )

                async for chunk in resp.iter_lines():
                    line = chunk.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = _HTTP_ANSWER.validate_json(line[5:])
                    except ValidationError as e:
                        raise APIError(
                            "MiniMax sent a malformed message", kind="protocol", detail=line
                        ) from e
                    _raise_status(data, data.get("trace_id") or trace_id)
                    extra_info = data.get("extra_info") or {}
                    if (billed := extra_info.get("usage_characters")) is not None:
                        output_emitter.report_usage(characters_count=billed)
                    if audio := (data.get("data") or {}).get("audio"):
                        output_emitter.push(bytes.fromhex(audio))
                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"Connection failed: {type(e).__name__}: {e}") from e


# MiniMax status codes by what failed; content and parameter errors refused the request
_STATUS_KINDS: dict[int, ErrorKind] = {
    1000: "server",  # unknown error
    1001: "timeout",
    1002: "rate_limit",
    1004: "auth",
    1008: "quota",  # insufficient balance
    1024: "server",  # internal error
    1033: "server",  # system error
    1039: "rate_limit",  # token rate limit
    1041: "rate_limit",  # connection limit
    2045: "rate_limit",  # rate growth limit
    2049: "auth",  # invalid api key
}


def _raise_status(answer: _Answer, trace_id: str | None) -> None:
    base_resp = answer.get("base_resp") or {}
    if status_code := base_resp.get("status_code", 0):
        raise APIStatusError(
            f"MiniMax error [{status_code}] (trace_id: {trace_id})",
            request_id=trace_id,
            status_code=status_code,
            body=answer,
            kind=_STATUS_KINDS.get(status_code, "invalid_request"),
            detail=base_resp.get("status_msg"),
        )


def _to_minimax_options(opts: _TTSOptions) -> _Options:
    voice_setting = _VoiceSetting(
        voice_id=opts.voice_id, speed=opts.speed, vol=opts.vol, pitch=opts.pitch
    )
    if opts.emotion is not None:
        voice_setting["emotion"] = opts.emotion
    config = _Options(
        model=opts.model,
        voice_setting=voice_setting,
        audio_setting=_AudioSetting(
            sample_rate=opts.sample_rate,
            bitrate=opts.bitrate,
            format=opts.audio_format,
            channel=1,
        ),
        text_normalization=opts.text_normalization,
    )
    if opts.language_boost is not None:
        config["language_boost"] = opts.language_boost
    if opts.pronunciation_dict:
        config["pronunciation_dict"] = opts.pronunciation_dict
    voice_modify = _VoiceModify()
    if opts.intensity is not None:
        voice_modify["intensity"] = opts.intensity
    if opts.timbre is not None:
        voice_modify["timbre"] = opts.timbre
    if voice_modify:
        config["voice_modify"] = voice_modify
    return config
