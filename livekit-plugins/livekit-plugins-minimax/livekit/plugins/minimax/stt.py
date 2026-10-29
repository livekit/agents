# Copyright 2025 LiveKit, Inc.
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

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .log import logger

ASRModel = Literal["asr-1.0"]

ASRResponseFormat = Literal["json", "verbose_json"]

# An optional BCP-47 hint for the *primary* spoken language. Omitting it (or passing an
# empty value) is what enables MiniMax's mixed-language recognition, so we deliberately
# do NOT default this to a language: a bilingual caller should leave it unset.
ASRLanguage = Literal[
    "zh",
    "yue",
    "en",
    "ja",
    "ko",
    "th",
    "vi",
    "id",
    "ms",
    "fil",
    "ar",
    "tr",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "pl",
    "ru",
    "uk",
]

DEFAULT_BASE_URL = (
    "https://api-uw.minimax.io"  # or "https://api.minimax.io or https://api.minimaxi.chat"
)
DEFAULT_MODEL: ASRModel = "asr-1.0"

# MiniMax caps a single transcription request at 500 seconds / 50 MB. Segments handed to
# `recognize()` come from VAD, so they are far shorter in practice; these bounds only exist
# to turn a silent truncation into a clear error before we spend an upload on it.
MAX_AUDIO_SECONDS = 500.0
MAX_AUDIO_BYTES = 50 * 1024 * 1024

# MiniMax reports application errors with HTTP 200 and a non-zero `base_resp.status_code`
# in the body, so a plain status check is not enough to detect a failure.
_SUCCESS_STATUS_CODE = 0

# The framework decides whether to retry from the status code: only 408/429/499 and 5xx
# are retryable. MiniMax's own codes fall outside that range, so a permanent failure such
# as "not authorized" would otherwise be retried pointlessly. These are the codes where
# MiniMax's own documentation says to retry later.
# https://platform.minimax.io/docs/api-reference/errorcode
_RETRYABLE_ERROR_CODES = frozenset(
    {
        1000,  # unknown error
        1001,  # request timeout
        1002,  # rate limit
        1024,  # internal error
        1033,  # system error
        1039,  # token limit
        1041,  # conn limit
        2045,  # rate growth limit
        2056,  # usage limit exceeded
    }
)


@dataclass
class _STTOptions:
    model: ASRModel | str
    language: ASRLanguage | str | None
    response_format: ASRResponseFormat
    api_key: str
    base_url: str


class STT(stt.STT):
    """MiniMax speech-to-text.

    MiniMax transcribes a complete audio file rather than a continuous stream, so this
    plugin reports ``streaming=False``. LiveKit wraps a non-streaming STT in
    [`stt.StreamAdapter`][livekit.agents.stt.StreamAdapter] using the session's VAD, which
    is what turns it into a streaming model — pass a `vad` to the `AgentSession` (or the
    `Agent`) and `recognize()` will be called once per detected speech segment.

    The transcript itself may additionally be streamed from MiniMax with SSE by passing
    ``stream=True``; that only affects how quickly the (already segmented) result arrives
    and is not the same thing as a streaming STT, so `capabilities.streaming` stays False
    either way.

    Example:
        ```python
        from livekit.agents import AgentSession
        from livekit.plugins import minimax, silero

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=minimax.STT(),  # language omitted on purpose: enables mixed zh/en
        )
        ```
    """

    def __init__(
        self,
        *,
        model: ASRModel | str = DEFAULT_MODEL,
        language: NotGivenOr[ASRLanguage | str | None] = NOT_GIVEN,
        response_format: ASRResponseFormat = "json",
        stream: bool = False,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new MiniMax STT instance.

        Args:
            model (ASRModel | str, optional): The MiniMax ASR model to use. Defaults to "asr-1.0".
            language (ASRLanguage | str | None, optional): An optional BCP-47 hint for the primary
                spoken language. Leave unset to enable MiniMax's mixed-language recognition, which
                is required for code-switched speech. Defaults to unset.
            response_format (ASRResponseFormat, optional): "json" returns the text and duration;
                "verbose_json" additionally asks for speaker diarization and timestamps, which
                MiniMax cannot combine with `stream=True`. Defaults to "json".
            stream (bool, optional): Ask MiniMax to push the transcript incrementally over SSE.
                Note that the whole segment still has to be uploaded before transcription starts.
                Defaults to False.
            api_key (str | None, optional): The MiniMax API key. Defaults to None.
            base_url (NotGivenOr[str], optional): The base URL for the MiniMax API. Defaults to NOT_GIVEN.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession
                to use. If not provided, a new session is created.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=response_format == "verbose_json",
            )
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

        if stream and response_format != "json":
            raise ValueError(
                "MiniMax cannot combine stream=True with response_format="
                f'"{response_format}": diarization and forced alignment require the full'
                ' audio, so only response_format="json" is supported when streaming'
            )

        self._opts = _STTOptions(
            model=model,
            # an explicit `language=None` is the same as leaving it unset: mixed-language mode
            language=language if utils.is_given(language) and language else None,
            response_format=response_format,
            api_key=minimax_api_key,
            base_url=base_url,
        )
        self._stream = stream
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MiniMax"

    def update_options(
        self,
        *,
        model: NotGivenOr[ASRModel | str] = NOT_GIVEN,
        language: NotGivenOr[ASRLanguage | str | None] = NOT_GIVEN,
    ) -> None:
        """Update the transcription options.

        Args:
            model (NotGivenOr[ASRModel | str], optional): The MiniMax ASR model to use.
            language (NotGivenOr[ASRLanguage | str | None], optional): An optional BCP-47 hint for
                the primary spoken language. Pass `None` to go back to mixed-language recognition.
        """
        if utils.is_given(model):
            self._opts.model = model
        if utils.is_given(language):
            self._opts.language = language or None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if not buffer:
            raise ValueError("MiniMax STT received an empty audio buffer")

        # a per-call language overrides the instance default for this request only
        requested_language = language if utils.is_given(language) else self._opts.language

        # miniMax needs a real container (wav/mp3/...); raw PCM is rejected. The buffer
        # arrives as mono int16 PCM, so `to_wav_bytes()` wraps it without re-encoding.
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        duration = utils.audio.calculate_audio_duration(buffer)
        if duration > MAX_AUDIO_SECONDS:
            raise APIStatusError(
                f"MiniMax transcription request is limited to {MAX_AUDIO_SECONDS:.0f}s of audio,"
                f" but the buffer is {duration:.1f}s long",
                status_code=400,
            )
        if len(wav_bytes) > MAX_AUDIO_BYTES:
            raise APIStatusError(
                f"MiniMax transcription request is limited to"
                f" {MAX_AUDIO_BYTES // (1024 * 1024)} MB, but the buffer is"
                f" {len(wav_bytes) / (1024 * 1024):.1f} MB",
                status_code=413,
            )

        try:
            text = await self._transcribe(wav_bytes, requested_language, conn_options)
        except APITimeoutError:
            raise
        except APIStatusError:
            raise
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    # MiniMax may answer in a different language than the hint (and does not
                    # echo the hint back), so we report what was asked for and leave the
                    # unknown case empty rather than guessing.
                    language=LanguageCode(requested_language or ""),
                )
            ],
        )

    async def _transcribe(
        self,
        wav_bytes: bytes,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> str:
        session = self._ensure_session()
        url = f"{self._opts.base_url}/v1/speech_to_text"

        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        if language:
            headers["language"] = language

        form = aiohttp.FormData()
        form.add_field("model", self._opts.model)
        form.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("response_format", self._opts.response_format)
        form.add_field("stream", "true" if self._stream else "false")

        async with session.post(
            url,
            headers=headers,
            data=form,
            timeout=aiohttp.ClientTimeout(
                total=conn_options.timeout,
                sock_connect=conn_options.timeout,
            ),
        ) as res:
            if res.status != 200:
                body = await res.text()
                raise APIStatusError(
                    f"MiniMax speech-to-text failed with status {res.status}: {body}",
                    status_code=res.status,
                    body=body,
                )

            if self._stream:
                return await self._read_sse(res)

            data: dict[str, Any] = await res.json()

        # MiniMax answers 200 with a non-zero status_code in `base_resp` on failure, so the
        # body is authoritative, not the HTTP status.
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code", _SUCCESS_STATUS_CODE)
        if status_code != _SUCCESS_STATUS_CODE:
            status_msg = base_resp.get("status_msg", "Unknown error")
            raise APIStatusError(
                f"MiniMax speech-to-text error [{status_code}]: {status_msg}",
                status_code=status_code,
                request_id=data.get("trace_id"),
                body=data,
                retryable=status_code in _RETRYABLE_ERROR_CODES,
            )

        return data.get("text", "") or ""

    async def _read_sse(self, res: aiohttp.ClientResponse) -> str:
        """Concatenate the `delta` values from MiniMax's SSE transcription stream.

        Events arrive as `data: {"index":0,"delta":"...","finish":false}`, separated by
        blank lines, with the terminating event carrying `finish: true`.
        """
        parts: list[tuple[int, str]] = []
        async for raw_line in res.content:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                # blank separators and keepalive comments
                continue
            if not line.startswith("data:"):
                logger.debug("ignoring unexpected MiniMax SSE line: %s", line)
                continue

            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue

            try:
                event: dict[str, Any] = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("failed to decode MiniMax SSE event: %s", payload)
                continue

            # a failure can also be reported mid-stream
            base_resp = event.get("base_resp") or {}
            status_code = base_resp.get("status_code", _SUCCESS_STATUS_CODE)
            if status_code != _SUCCESS_STATUS_CODE:
                raise APIStatusError(
                    f"MiniMax speech-to-text stream error [{status_code}]:"
                    f" {base_resp.get('status_msg', 'Unknown error')}",
                    status_code=status_code,
                    body=event,
                    retryable=status_code in _RETRYABLE_ERROR_CODES,
                )

            if event.get("delta"):
                parts.append((event.get("index", len(parts)), event["delta"]))

            if event.get("finish"):
                break

        # `index` restarts at 0 per request; sort defensively rather than trusting arrival order
        return "".join(text for _, text in sorted(parts, key=lambda item: item[0]))
