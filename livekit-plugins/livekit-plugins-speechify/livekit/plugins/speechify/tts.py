# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any, cast

import httpx

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString
from speechify.client import AsyncSpeechify
from speechify.core.api_error import ApiError
from speechify.types.get_voice import GetVoice

from .log import logger
from .models import Gender, TTSModels, VoiceType

DEFAULT_VOICE_ID = "dominic_32"
DEFAULT_MODEL: TTSModels = "simba-3.2"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
AUDIO_FORMAT = "pcm"
MIME_TYPE = "audio/pcm"
CALLER_HEADER = "Speechify-Caller"


@dataclass
class _TTSOptions:
    voice_id: str
    model: NotGivenOr[TTSModels]
    language: NotGivenOr[str]
    loudness_normalization: NotGivenOr[bool]
    text_normalization: NotGivenOr[bool]


@dataclass
class Voice:
    id: str
    type: VoiceType
    display_name: str
    gender: Gender
    avatar_image: str | None
    models: list[TTSModels]
    locale: str


def _voice_from_sdk(v: GetVoice) -> Voice:
    # The SDK exposes "notSpecified"; the plugin's Gender literal is "neutral".
    gender = "neutral" if v.gender == "notSpecified" else v.gender
    return Voice(
        id=v.id,
        type=cast(VoiceType, v.type),
        display_name=v.display_name,
        gender=cast(Gender, gender),
        avatar_image=v.avatar_image,
        models=[cast(TTSModels, m.name) for m in v.models],
        locale=v.locale,
    )


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model: TTSModels = DEFAULT_MODEL,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        client: AsyncSpeechify | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new instance of Speechify TTS.

        Synthesis uses the Speechify ``/audio/speech`` endpoint, which returns
        raw PCM (24 kHz mono) together with word-level speech marks. ``stream()``
        splits input into sentences and issues one request per sentence, emitting
        audio and aligned word timestamps as each sentence completes for
        near-streaming time-to-first-audio.

        Args:
            voice_id: Id of the voice to synthesize with. The voice must support
                the chosen ``model`` (see the ``/v1/voices`` endpoint). Defaults
                to ``dominic_32``.
            model: Synthesis model. One of ``simba-english``,
                ``simba-multilingual``, ``simba-3.0`` or ``simba-3.2``. Defaults
                to ``simba-3.2``.
            language: BCP-47 language code of the input (e.g. ``en-US``).
            loudness_normalization: Normalize output loudness to a standard
                level. Increases latency slightly when enabled.
            text_normalization: Expand numbers, dates, etc. into words before
                synthesis. Increases latency slightly when enabled.
            api_key: Speechify API key. Falls back to the ``SPEECHIFY_API_KEY``
                environment variable.
            base_url: Override the Speechify API base URL.
            tokenizer: Sentence tokenizer used to chunk input in ``stream()``.
            client: A preconfigured ``AsyncSpeechify`` client. When provided,
                ``api_key`` and ``base_url`` are ignored.
            **kwargs: Catches deprecated parameters. A warning is logged for
                any recognised deprecated name.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._owns_client = client is None

        if client is not None:
            # Use preconfigured client and extract its credentials
            self._client = client
            # Extract the httpx client from the SDK client wrapper
            self._httpx_client = client._client_wrapper.httpx_client
            # Extract base_url and token from the configured client
            self._base_url = client._client_wrapper.base_url or "https://api.sws.speechify.com"
            # Extract token from the client's auth header
            auth_header = client._client_wrapper._get_default_headers().get("Authorization", "")
            if auth_header.startswith("Bearer "):
                self._api_key = auth_header[7:]
            else:
                self._api_key = auth_header or ""
        else:
            # Create our own client with validated credentials
            resolved_key = api_key if is_given(api_key) else os.environ.get("SPEECHIFY_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Speechify API key is required, either as the api_key argument "
                    "or via the SPEECHIFY_API_KEY environment variable"
                )
            self._api_key = resolved_key
            resolved_base_url = base_url if is_given(base_url) else "https://api.sws.speechify.com"
            # Validate base URL scheme for security
            if not resolved_base_url.startswith(("https://", "http://")):
                raise ValueError(
                    f"base_url must start with http:// or https://, got: {resolved_base_url}"
                )
            self._base_url = resolved_base_url
            # Fixed httpx.AsyncClient default header so every request the SDK
            # issues is attributed to this integration, regardless of call site.
            # Timeout/limits mirror the openai plugin's owned-client defaults —
            # httpx's own 5s default is too short for longer synthesis requests.
            self._httpx_client = httpx.AsyncClient(
                headers={CALLER_HEADER: "livekit"},
                timeout=httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=5.0),
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            )
            self._client = AsyncSpeechify(
                token=resolved_key,
                base_url=self._base_url,
                httpx_client=self._httpx_client,
            )

        self._tokenizer = tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        self._opts = _TTSOptions(
            voice_id=voice_id,
            model=model,
            language=language,
            loudness_normalization=loudness_normalization,
            text_normalization=text_normalization,
        )

        _check_deprecated_args(kwargs)

    @property
    def model(self) -> str:
        return self._opts.model if is_given(self._opts.model) else "unknown"

    @property
    def provider(self) -> str:
        return "Speechify"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._httpx_client.aclose()

    async def list_voices(self) -> list[Voice]:
        """List the voices available for the configured Speechify account."""
        sdk_voices: list[GetVoice] = await self._client.voices.list()
        return [_voice_from_sdk(v) for v in sdk_voices]

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(loudness_normalization):
            self._opts.loudness_normalization = loudness_normalization
        if is_given(text_normalization):
            self._opts.text_normalization = text_normalization

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


def _request_kwargs(text: str, opts: _TTSOptions) -> dict[str, object]:
    options: dict[str, bool] = {}
    if is_given(opts.loudness_normalization):
        options["loudness_normalization"] = opts.loudness_normalization
    if is_given(opts.text_normalization):
        options["text_normalization"] = opts.text_normalization

    kwargs: dict[str, object] = {
        "audio_format": AUDIO_FORMAT,
        "input": text,
        "voice_id": opts.voice_id,
    }
    if is_given(opts.model):
        kwargs["model"] = opts.model
    if is_given(opts.language):
        kwargs["language"] = opts.language
    if options:
        kwargs["options"] = options
    return kwargs


def _timed_transcript(speech_marks: object, offset: float) -> list[TimedString]:
    chunks = getattr(speech_marks, "chunks", None)
    if not chunks:
        return []
    out: list[TimedString] = []
    for chunk in chunks:
        value = getattr(chunk, "value", None)
        start = getattr(chunk, "start_time", None)
        if value is None or start is None:
            continue
        end = getattr(chunk, "end_time", None)
        out.append(
            TimedString(
                text=value,
                start_time=start / 1000 + offset,
                end_time=(end / 1000 + offset) if end is not None else NOT_GIVEN,
            )
        )
    return out


async def _stream_with_timestamps(
    *,
    text: str,
    opts: _TTSOptions,
    timeout: float,
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
) -> tuple[bytes, list[dict]]:
    """Call /v1/audio/stream/with-timestamps and parse SSE response."""
    url = f"{base_url}/v1/audio/stream/with-timestamps"

    request_body = {
        "input": text,
        "voice_id": opts.voice_id,
        "output_format": f"pcm_{SAMPLE_RATE}",
    }

    if is_given(opts.model):
        request_body["model"] = opts.model
    if is_given(opts.language):
        request_body["language"] = opts.language

    options = {}
    if is_given(opts.loudness_normalization):
        options["loudness_normalization"] = opts.loudness_normalization
    if is_given(opts.text_normalization):
        options["text_normalization"] = opts.text_normalization
    if options:
        request_body["options"] = options

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        CALLER_HEADER: "livekit",
    }

    audio_chunks = []
    all_speech_marks = []

    try:
        async with client.stream(
            "POST", url, json=request_body, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()

            event_type = None
            async for line in response.aiter_lines():
                line = line.strip()

                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()

                    try:
                        parsed = json.loads(data_str)

                        if event_type == "speech.chunk":
                            if "audio" in parsed:
                                audio_b64 = parsed["audio"]
                                audio_bytes = base64.b64decode(audio_b64)
                                audio_chunks.append(audio_bytes)

                            if "speech_marks" in parsed:
                                all_speech_marks.extend(parsed["speech_marks"])
                    except json.JSONDecodeError:
                        pass

        audio_bytes = b"".join(audio_chunks)
        return audio_bytes, all_speech_marks

    except httpx.TimeoutException:
        raise APITimeoutError() from None
    except httpx.HTTPStatusError as e:
        raise APIStatusError(
            message=str(e),
            status_code=e.response.status_code,
            request_id=None,
            body=None,
        ) from None
    except Exception as e:
        raise APIConnectionError() from e


def _timed_transcript_from_marks(speech_marks: list[dict], offset: float) -> list[TimedString]:
    """Convert API speech marks to TimedString segments."""
    if not speech_marks:
        return []

    out: list[TimedString] = []
    for mark in speech_marks:
        if mark.get("type") != "word":
            continue

        value = mark.get("value")
        start = mark.get("start_time")
        if value is None or start is None:
            continue

        end = mark.get("end_time")
        out.append(
            TimedString(
                text=value,
                start_time=start / 1000 + offset,
                end_time=(end / 1000 + offset) if end is not None else NOT_GIVEN,
            )
        )
    return out


def _raise_from(e: Exception) -> None:
    if isinstance(e, APIError):
        raise e
    if isinstance(e, ApiError):
        raise APIStatusError(
            message=str(e.body) if e.body is not None else "Speechify API error",
            status_code=e.status_code or -1,
            request_id=None,
            body=None,
        ) from None
    if isinstance(e, asyncio.TimeoutError):
        raise APITimeoutError() from None
    if isinstance(e, ConnectionError):
        raise APIConnectionError() from e
    raise e


def _check_deprecated_args(kwargs: dict[str, Any]) -> None:
    """Warn about kwargs from earlier plugin versions that no longer apply."""
    removed = {
        "encoding": "output is fixed to 24 kHz PCM",
        "http_session": "the official Speechify SDK owns the HTTP client",
        "follow_redirects": "the official Speechify SDK handles redirects",
    }
    for name, reason in removed.items():
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used ({reason})")


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            audio_bytes, speech_marks = await _stream_with_timestamps(
                text=self._input_text,
                opts=self._opts,
                timeout=self._conn_options.timeout,
                client=self._tts._httpx_client,
                api_key=self._tts._api_key,
                base_url=self._tts._base_url,
            )
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=MIME_TYPE,
            )
            timed = _timed_transcript_from_marks(speech_marks, 0.0)
            if timed:
                output_emitter.push_timed_transcript(timed)
            output_emitter.push(audio_bytes)
            output_emitter.flush()
        except Exception as e:
            _raise_from(e)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE,
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        sent_stream = self._tts._tokenizer.stream()

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _synthesize() -> None:
            offset = 0.0
            async for ev in sent_stream:
                if not (text := ev.token.strip()):
                    continue
                self._mark_started()
                audio_bytes, speech_marks = await _stream_with_timestamps(
                    text=text,
                    opts=self._opts,
                    timeout=self._conn_options.timeout,
                    client=self._tts._httpx_client,
                    api_key=self._tts._api_key,
                    base_url=self._tts._base_url,
                )
                timed = _timed_transcript_from_marks(speech_marks, offset)
                if timed:
                    output_emitter.push_timed_transcript(timed)
                output_emitter.push(audio_bytes)
                output_emitter.flush()
                offset += len(audio_bytes) / (2 * SAMPLE_RATE * NUM_CHANNELS)

            output_emitter.end_segment()

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            _raise_from(e)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
