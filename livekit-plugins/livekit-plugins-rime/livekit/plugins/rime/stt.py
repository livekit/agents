# Copyright 2026 LiveKit, Inc.
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

from collections.abc import Sequence
from typing import Literal

import numpy as np

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given


class STT(stt.STT):
    """Recognize complete utterances using the Rime Python SDK.

    Requires Python 3.11+ and the ``livekit-plugins-rime[stt]`` extra.
    Supply a VAD to ``AgentSession`` to use this recognizer with live audio;
    LiveKit's ``stt.StreamAdapter`` sends each completed utterance to Rime.
    Only final transcripts are emitted.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        mode: Literal["written", "verbatim"] = "written",
        context_terms: Sequence[str] = (),
        api_key: str | None = None,
    ) -> None:
        """Create a Rime speech recognizer.

        Args:
            language: Recognition language, passed unchanged to Rime. Defaults to English.
            mode: Prefer written forms (``written``) or spoken forms (``verbatim``).
            context_terms: Words or phrases to bias recognition towards.
            api_key: Rime API key. Defaults to the ``RIME_API_KEY`` environment variable.
        """
        # Keep the SDK optional for TTS-only installations.
        try:
            import rimelabs_sdk
        except ModuleNotFoundError as exc:
            if exc.name != "rimelabs_sdk":
                raise
            raise ImportError("Install livekit-plugins-rime[stt] to use Rime STT") from None

        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False, keyterms=True)
        )
        self._sdk = rimelabs_sdk
        self._language = language
        self._mode = rimelabs_sdk.TranscriptionMode(mode)
        self._context_terms = _copy_context_terms(context_terms)
        self._session_keyterms: tuple[str, ...] = ()
        self._client = rimelabs_sdk.Rime(api_key=api_key)
        if not callable(getattr(self._client.stt, "transcribe", None)):
            raise ImportError(
                "Rime STT requires a rimelabs-sdk build with stt.transcribe() support"
            )

    @property
    def provider(self) -> str:
        return "Rime"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        mode: NotGivenOr[Literal["written", "verbatim"]] = NOT_GIVEN,
        context_terms: NotGivenOr[Sequence[str]] = NOT_GIVEN,
    ) -> None:
        """Update defaults for subsequent recognition requests."""
        if is_given(language):
            self._language = language
        if is_given(mode):
            self._mode = self._sdk.TranscriptionMode(mode)
        if is_given(context_terms):
            self._context_terms = _copy_context_terms(context_terms)

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._session_keyterms = tuple(keyterms)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            audio = _to_wav(buffer)
        except ValueError:
            raise APIError("Invalid audio buffer for Rime STT", retryable=False) from None

        try:
            result = await self._client.stt.transcribe(
                audio,
                language=language if is_given(language) else self._language,
                mode=self._mode,
                context_terms=tuple(dict.fromkeys((*self._context_terms, *self._session_keyterms))),
                timeout=conn_options.timeout,
            )
        except self._sdk.RimeError as exc:
            error: APIError
            if isinstance(exc, self._sdk.RimeTimeoutError):
                error = APITimeoutError("Rime STT request timed out")
            elif isinstance(exc, self._sdk.RimeUnavailableError):
                error = APIConnectionError("Rime STT service unavailable")
            else:
                # The SDK does not expose HTTP status codes. Preserve the request identity
                # without inventing a status or retrying invalid input / resource limits.
                message = "Rime STT request failed"
                if isinstance(exc, self._sdk.RimeAuthenticationError):
                    message = "Rime STT authentication failed"
                elif isinstance(exc, self._sdk.RimePermissionError):
                    message = "Rime STT permission denied"
                elif isinstance(exc, (self._sdk.RimeInputError, self._sdk.RimeAudioFormatError)):
                    message = "Rime STT rejected the input"
                elif isinstance(exc, self._sdk.RimeResourceLimitError):
                    message = "Rime STT resource limit exceeded"
                elif isinstance(exc, self._sdk.RimeCancelledError):
                    message = "Rime STT request cancelled"
                error = APIStatusError(
                    message,
                    request_id=exc.request_id,
                    retryable=False,
                )
            if exc.request_id:
                error.body = {"request_id": exc.request_id}
            # Provider messages can include request contents. Do not chain them into logs.
            raise error from None

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=result.request_id or "",
            alternatives=[stt.SpeechData(text=result.text, language=LanguageCode(result.language))],
        )

    async def aclose(self) -> None:
        """Close the SDK client and cancel outstanding recognition requests."""
        await self._client.close()


def _to_wav(buffer: utils.AudioBuffer) -> bytes:
    """Convert LiveKit PCM16 audio to Rime's portable mono, 16 kHz WAV format."""
    frame = rtc.combine_audio_frames(buffer)
    if frame.samples_per_channel == 0:
        raise ValueError("Audio buffer is empty")
    if frame.num_channels != 1:
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        # Average in floating point to avoid overflowing when mixing loud channels.
        mono = samples.mean(axis=1).astype(np.int16)
        frame = rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=frame.samples_per_channel,
        )
    if frame.sample_rate != 16000:
        resampler = rtc.AudioResampler(frame.sample_rate, 16000, num_channels=1)
        frame = rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])
    return frame.to_wav_bytes()


def _copy_context_terms(terms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)) or any(not isinstance(term, str) for term in terms):
        raise ValueError("context_terms must be a sequence of strings")
    return tuple(terms)
