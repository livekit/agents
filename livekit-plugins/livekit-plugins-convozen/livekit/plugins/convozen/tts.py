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

"""Text-to-Speech implementation for ConvoZen Ragini.

Ragini synthesizes a whole request at a time — there is no incremental
text-in/audio-out socket — so this is declared as a non-streaming TTS and
``Agent.tts_node`` wraps it in :class:`livekit.agents.tts.StreamAdapter`, which
splits the LLM's token stream into sentences and synthesizes them one by one.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import (
    DEFAULT_BASE_URL,
    TTS_DEFAULT_SAMPLE_RATES,
    TTS_FALLBACK_SAMPLE_RATE,
    TTS_LANGUAGES,
    TTSLanguages,
    TTSModels,
    TTSVoices,
)

_TTS_PATH = "/v1/ragini/tts"

_MAX_ERROR_BODY = 512

NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: TTSModels | str
    voice: str
    language: str
    sample_rate: int
    speed: float
    stream_response: bool

    def tts_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{_TTS_PATH}"

    def to_form(self, text: str) -> dict[str, str]:
        """Build the urlencoded body the /v1/ragini/tts endpoint expects.

        Note the wire name for the voice is ``speaker``. There is no ``format``
        field — the server always returns WAV.
        """
        return {
            "text": text,
            "language": self.language,
            "speaker": self.voice,
            "model": str(self.model),
            "sample_rate": str(self.sample_rate),
            "speed": str(self.speed),
            "stream": "true" if self.stream_response else "false",
        }


class TTS(tts.TTS):
    """ConvoZen Ragini text-to-speech.

    Ragini covers nine Indian languages.
    """

    def __init__(
        self,
        *,
        voice: TTSVoices | str = "roohi",
        model: TTSModels | str = "ragini-v1",
        language: TTSLanguages | str = "en",
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speed: float = 1.0,
        stream_response: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of ConvoZen Ragini TTS.

        Args:
            voice: Voice (``speaker``) id. Any non-empty string is accepted so that
                voices added server-side work without a plugin upgrade.
            model: ``"ragini-v1"`` (default) or the lighter ``"ragini-lite"``.
            language: Language of the text to synthesize.
            sample_rate: Output rate in Hz. Defaults to the model's native rate —
                24000 for ``ragini-v1``, 22050 for ``ragini-lite``.
            speed: Speaking rate multiplier; 1.0 is the natural pace.
            stream_response: Ask the server to stream the audio back as it is
                generated, which cuts time-to-first-byte. Set to ``False`` to receive
                one complete WAV instead.
            api_key: ConvoZen API key. Falls back to the ``CONVOZEN_API_KEY``
                environment variable.
            base_url: API base URL. Falls back to ``CONVOZEN_BASE_URL``, then to the
                public endpoint.
            http_session: An existing aiohttp session to use. By default the shared
                LiveKit session is used.

        Raises:
            ValueError: If no API key is available, or ``language``/``voice`` is not
                a value Ragini accepts.
        """
        if language not in TTS_LANGUAGES:
            raise ValueError(
                f"unsupported language: {language!r}. supported: {sorted(TTS_LANGUAGES)}"
            )
        if not voice or not str(voice).strip():
            raise ValueError("voice cannot be empty")

        resolved_sample_rate = (
            sample_rate
            if is_given(sample_rate)
            else TTS_DEFAULT_SAMPLE_RATES.get(str(model), TTS_FALLBACK_SAMPLE_RATE)
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=resolved_sample_rate,
            num_channels=NUM_CHANNELS,
        )

        convozen_api_key = api_key or os.environ.get("CONVOZEN_API_KEY")
        if not convozen_api_key:
            raise ValueError(
                "ConvoZen API key is required. "
                "Provide it directly or set the CONVOZEN_API_KEY environment variable."
            )

        self._opts = _TTSOptions(
            api_key=convozen_api_key,
            base_url=base_url or os.environ.get("CONVOZEN_BASE_URL") or DEFAULT_BASE_URL,
            model=model,
            voice=str(voice).strip(),
            language=language,
            sample_rate=resolved_sample_rate,
            speed=speed,
            stream_response=stream_response,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "ConvoZen"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options for subsequent requests.

        ``sample_rate`` is deliberately absent: it is fixed for the lifetime of the
        instance because the audio pipeline is built around it at construction.
        """
        if is_given(voice):
            if not str(voice).strip():
                raise ValueError("voice cannot be empty")
            self._opts.voice = str(voice).strip()
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            if language not in TTS_LANGUAGES:
                raise ValueError(
                    f"unsupported language: {language!r}. supported: {sorted(TTS_LANGUAGES)}"
                )
            self._opts.language = language
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a chunk of text through the Ragini bytes endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                url=self._opts.tts_url(),
                data=self._opts.to_form(self._input_text),
                headers={"x-api-key": self._opts.api_key},
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:_MAX_ERROR_BODY]
                    raise APIStatusError(
                        message=f"ConvoZen Ragini returned {resp.status}: {body}",
                        status_code=resp.status,
                        request_id=None,
                        body=body,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    # Ragini returns a RIFF/WAV container, not bare samples. The
                    # emitter hands anything that isn't audio/pcm to
                    # AudioStreamDecoder, which is why this package depends on
                    # livekit-agents[codecs].
                    mime_type="audio/wav",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except APIStatusError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # `from None`, not `from e`: aiohttp embeds the RequestInfo — URL and
            # request headers, which carry the API key — in its exception chain.
            raise APIConnectionError(type(e).__name__) from None
