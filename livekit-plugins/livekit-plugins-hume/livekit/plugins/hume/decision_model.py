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

import asyncio
import copy
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import aiohttp
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
    utils,
)
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import (
    Choice,
    Decision,
    DecisionCapabilities,
    DecisionResponse,
    Probability,
    _validate_decisions,
    _validate_response,
)
from livekit.agents.metrics import DecisionMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .version import __version__

DEFAULT_DECISIONS_URL = "https://api.cloud.hume.ai"
DECISIONS_PATH = "/v0/decisions/file"
API_AUTH_HEADER = "X-Hume-Api-Key"
DEFAULT_HEADERS = {
    "X-Hume-Client-Name": "livekit",
    "X-Hume-Client-Version": __version__,
}

# The model hears one channel at 16 kHz, so anything more only adds bytes against the
# 25 MB request limit.
MODEL_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 5.0
MAX_AUDIO_DURATION = 300.0


class _Received(BaseModel):
    input_tokens: int | None = None


class _Response(BaseModel):
    run_id: str = ""
    answers: dict[str, Any]
    received: _Received = Field(default_factory=_Received)


class _ErrorBody(BaseModel):
    code: str = ""
    message: str = ""
    param: str = ""
    request_id: str = ""


class AudioDecisionModel(rtc.EventEmitter[Literal["metrics_collected"]]):
    """Hume's Audio Decisions API: questions answered by listening to a recording.

    Unlike :class:`livekit.agents.llm.DecisionModel`, which reads a transcript, this
    model hears the audio itself, so it can judge tone, frustration, naturalness or
    interruptions that a transcript does not carry. Decisions and results use the
    framework's types: ``Probability`` maps to Hume's ``noul``, ``Choice`` to ``choice``
    and ``Score`` to ``score``.

    Each answer estimates the share of human raters who would give it, not the model's
    certainty. Choose thresholds from audio you have labeled, and send the same
    decisions in the same order when comparing recordings.

    Example::

        model = hume.AudioDecisionModel()
        response = await model.evaluate(
            audio=frames,
            decisions={"frustrated": llm.decisions.Probability("Is the caller frustrated?")},
            context="A support call. The caller speaks first.",
        )

    See https://dev.hume.ai/audio-decisions/docs/overview.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_DECISIONS_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            api_key: Hume API key. Defaults to the HUME_API_KEY environment variable.
            base_url: Base URL of the Audio Decisions API.
            http_session: Optional caller-owned session. Otherwise uses the job's HTTP
                session. Outside a job, use ``async with utils.http_context.open()``.
        """
        super().__init__()
        key = api_key or os.environ.get("HUME_API_KEY")
        if not key:
            raise ValueError("Hume API key is required via api_key or HUME_API_KEY env var")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._session = http_session
        self._capabilities = DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def capabilities(self) -> DecisionCapabilities:
        return self._capabilities

    @property
    def model(self) -> str:
        return "audio-decisions"

    @property
    def provider(self) -> str:
        return "Hume"

    async def evaluate(
        self,
        *,
        audio: rtc.AudioFrame | Sequence[rtc.AudioFrame] | bytes,
        decisions: Mapping[str, Decision],
        context: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DecisionResponse:
        """Evaluate a batch of decisions against one recording.

        Args:
            audio: 5 seconds to 5 minutes of audio. Frames are mixed to mono and
                resampled to 16 kHz before upload. ``bytes`` is sent as-is and must be
                a WAV or MP3 file.
            decisions: Decisions keyed by name. All of them are answered in one
                inference, so an answer can change when others are added or reordered.
            context: Up to 1,000 characters describing the recording, such as the
                setting and who the speakers are. Shared by every decision.
            conn_options: Retry and per-attempt timeout settings.

        Each requested name appears in either ``results`` or ``errors``. ``request_id``
        is Hume's run id. Request failures and malformed responses raise.
        """
        definitions = copy.deepcopy(dict(decisions))
        _validate_decisions(definitions, capabilities=self.capabilities)
        if not definitions:
            return DecisionResponse(results={}, model=self.model, provider=self.provider)

        file, filename, content_type = _prepare_audio(audio)
        request: dict[str, Any] = {
            "questions": {name: _encode_question(d) for name, d in definitions.items()}
        }
        if context:
            request["context"] = context

        started = time.perf_counter()
        for attempt in range(conn_options.max_retry + 1):
            try:
                response = await asyncio.wait_for(
                    self._post(
                        request=request,
                        file=file,
                        filename=filename,
                        content_type=content_type,
                        conn_options=conn_options,
                    ),
                    timeout=conn_options.timeout,
                )
            except (APIError, asyncio.TimeoutError) as error:
                if isinstance(error, APIError) and not error.retryable:
                    raise
                if attempt == conn_options.max_retry:
                    if isinstance(error, asyncio.TimeoutError):
                        raise APITimeoutError("Hume decision request timed out") from error
                    raise
                await asyncio.sleep(_retry_delay(error, conn_options, attempt))
                continue

            self.emit(
                "metrics_collected",
                DecisionMetrics(
                    label=self.label,
                    request_id=response.request_id,
                    timestamp=time.time(),
                    duration=time.perf_counter() - started,
                    input_tokens=response.input_tokens,
                    metadata=Metadata(model_name=self.model, model_provider=self.provider),
                ),
            )
            _validate_response(response, definitions)
            return response
        raise RuntimeError("unreachable")

    async def _post(
        self,
        *,
        request: dict[str, Any],
        file: bytes,
        filename: str,
        content_type: str,
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        # Hume reads the parts in order: `request` first, then `file`.
        form = aiohttp.FormData()
        form.add_field("request", json.dumps(request), content_type="application/json")
        form.add_field("file", file, filename=filename, content_type=content_type)

        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                f"{self._base_url}{DECISIONS_PATH}",
                headers={**DEFAULT_HEADERS, API_AUTH_HEADER: self._api_key},
                data=form,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                request_id = resp.headers.get("X-Request-Id")
                if resp.status >= 400:
                    raise _error_from_response(
                        await resp.text(),
                        status=resp.status,
                        request_id=request_id,
                        retry_after=resp.headers.get("Retry-After"),
                    )
                parsed = _Response.model_validate(await resp.json())
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("Hume decision request timed out") from exc
        except aiohttp.ContentTypeError as exc:
            raise APIError("invalid Hume decision response", retryable=False) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("Hume connection failed") from exc
        except ValueError as exc:
            raise APIError("invalid Hume decision response", retryable=False) from exc

        results, errors = systemone.decode_answers(
            parsed.answers, invalid="invalid Hume decision answer"
        )
        return DecisionResponse(
            results=results,
            errors=errors,
            model=self.model,
            provider=self.provider,
            request_id=parsed.run_id or request_id or "",
            input_tokens=parsed.received.input_tokens,
        )

    async def aclose(self) -> None:
        """Release resources owned by this model, if any."""


def _encode_question(definition: Decision) -> dict[str, Any]:
    question: dict[str, Any] = {"instructions": definition.prompt}
    if isinstance(definition, Probability):
        question["type"] = "noul"
    elif isinstance(definition, Choice):
        # Hume takes choice options as a list; a description is optional but must not be
        # empty when sent.
        question.update(
            type="choice",
            criteria=[
                {"name": name, "description": description} if description else {"name": name}
                for name, description in definition.options.items()
            ],
        )
    else:
        question.update(type="score", criteria=definition.levels)
    return question


def _prepare_audio(
    audio: rtc.AudioFrame | Sequence[rtc.AudioFrame] | bytes,
) -> tuple[bytes, str, str]:
    if isinstance(audio, (bytes, bytearray)):
        if audio[:4] == b"RIFF":
            return bytes(audio), "audio.wav", "audio/wav"
        return bytes(audio), "audio.mp3", "audio/mpeg"

    frames = [audio] if isinstance(audio, rtc.AudioFrame) else list(audio)
    if not frames:
        raise ValueError("Hume audio decisions require audio")
    frame = rtc.combine_audio_frames(frames)
    if not MIN_AUDIO_DURATION <= frame.duration <= MAX_AUDIO_DURATION:
        raise ValueError(
            f"Hume audio decisions take {MIN_AUDIO_DURATION:g} to {MAX_AUDIO_DURATION:g} "
            f"seconds of audio, got {frame.duration:.2f}"
        )
    return _to_model_format(frame).to_wav_bytes(), "audio.wav", "audio/wav"


def _to_model_format(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    if frame.num_channels > 1:
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = samples.mean(axis=1).astype(np.int16)
        frame = rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono),
        )
    if frame.sample_rate == MODEL_SAMPLE_RATE:
        return frame

    resampler = rtc.AudioResampler(
        input_rate=frame.sample_rate, output_rate=MODEL_SAMPLE_RATE, num_channels=1
    )
    return rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])


class _StatusError(APIStatusError):
    def __init__(self, error: APIStatusError, *, retry_after: float | None) -> None:
        super().__init__(
            error.message,
            status_code=error.status_code,
            request_id=error.request_id,
            body=error.body,
        )
        self.retry_after = retry_after
        """Seconds the server asked to wait before retrying (429 and 503)."""


def _retry_delay(error: BaseException, conn_options: APIConnectOptions, attempt: int) -> float:
    if isinstance(error, _StatusError) and error.retry_after is not None:
        return error.retry_after
    return conn_options._interval_for_retry(attempt)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _error_from_response(
    text: str, *, status: int, request_id: str | None, retry_after: str | None
) -> _StatusError:
    body: object = text
    message = "Hume decision request failed"
    try:
        parsed = _ErrorBody.model_validate_json(text)
    except ValidationError:
        pass
    else:
        body = parsed.model_dump(exclude_defaults=True)
        request_id = request_id or parsed.request_id or None
        if parsed.code or parsed.message:
            detail = ": ".join(part for part in (parsed.code, parsed.message) if part)
            message = f"{message}: {detail}"
            if parsed.param:
                message = f"{message} (at {parsed.param})"
    return _StatusError(
        create_api_error_from_http(message, status=status, request_id=request_id, body=body),
        retry_after=_parse_retry_after(retry_after),
    )
