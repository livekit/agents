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

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

class STT(stt.STT):
    """Qwen3-ASR on a self-hosted vLLM server.

    Batch recognition posts the utterance to ``/v1/audio/transcriptions``.
    ``language`` and ``prompt`` are omitted from the request when unset, so the
    model keeps its own language detection and an empty context.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "Qwen/Qwen3-ASR-1.7B",
        language: str | None = None,
        prompt: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            base_url: OpenAI-compatible root, including ``/v1``.
                Example: ``http://127.0.0.1:8000/v1``.
            model: Name the vLLM server is serving.
            language: BCP-47 or ISO-639-1 code, such as ``"tr"``. ``None`` leaves
                detection to the model.
            prompt: Optional context. vLLM places this in Qwen3-ASR's system turn.
                Hotwords belong in this text; the model has no separate keywords field.
            api_key: Sent as ``Authorization: Bearer`` when set. A server started
                without ``--api-key`` does not need one.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                keyterms=True,
            )
        )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language or None
        self._prompt = prompt or None
        self._api_key = api_key or None
        self._session_keyterms: list[str] = []
        self._http: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Qwen3-ASR"

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        prompt: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Replace the model, language, or context used by later requests."""
        if is_given(model):
            self._model = model
        if is_given(language):
            self._language = language or None
        if is_given(prompt):
            self._prompt = prompt or None

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._session_keyterms = list(keyterms)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._language = language or None

        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self._model)
        if self._language:
            form.add_field("language", self._language)
        context = self._context_prompt()
        if context:
            form.add_field("prompt", context)

        timeout = aiohttp.ClientTimeout(total=conn_options.timeout)
        try:
            async with self._ensure_http().post(
                f"{self._base_url}/audio/transcriptions",
                data=form,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                body = await _read_body(resp)
                if resp.status >= 400:
                    raise APIStatusError(
                        _error_message(body, resp.reason or "transcription failed"),
                        status_code=resp.status,
                        body=body,
                    )
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except TimeoutError as exc:
            raise APITimeoutError() from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to reach Qwen3-ASR") from exc

        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise APIStatusError(
                "Qwen3-ASR response did not include a transcript",
                status_code=500,
                body=body,
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(self._language or ""),
                    text=text,
                )
            ],
        )

    async def aclose(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None

    def _context_prompt(self) -> str | None:
        """User prompt plus LiveKit keyterms, which Qwen only accepts as context text."""
        parts: list[str] = []
        if self._prompt:
            parts.append(self._prompt)
        if self._session_keyterms:
            parts.append("Vocabulary: " + ", ".join(self._session_keyterms))
        text = "\n".join(parts).strip()
        return text or None

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "LiveKit Agents"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http


async def _read_body(resp: aiohttp.ClientResponse) -> object:
    try:
        return await resp.json(content_type=None)
    except (aiohttp.ContentTypeError, ValueError):
        return await resp.text()


def _error_message(body: object, fallback: str) -> str:
    if isinstance(body, dict):
        error = body.get("error", body.get("message"))
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(error, str) and error:
            return error
    if isinstance(body, str) and body:
        return body
    return fallback
