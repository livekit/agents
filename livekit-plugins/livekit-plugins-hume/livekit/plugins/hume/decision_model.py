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
import json
import os
from collections.abc import Mapping
from typing import Any

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
    llm,
    utils,
)
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import (
    Choice,
    Decision,
    DecisionCapabilities,
    DecisionResponse,
    Probability,
)
from livekit.agents.types import APIConnectOptions

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
MAX_CONTEXT_LENGTH = 1000


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


class DecisionModel(llm.DecisionModel):
    """Hume's Audio Decisions API: decisions answered by listening to a recording.

    The model hears the audio itself, so it can judge tone, frustration, naturalness or
    interruptions that a transcript does not carry. Every call needs ``audio``; ``input``
    is optional and becomes Hume's ``context``, a description of the recording such as
    the setting and who the speakers are. ``Probability`` maps to Hume's ``noul``,
    ``Choice`` to ``choice`` and ``Score`` to ``score``.

    Each answer estimates the share of human raters who would give it, not the model's
    certainty. Choose thresholds from audio you have labeled, and send the same
    decisions in the same order when comparing recordings.

    Example::

        model = hume.DecisionModel()
        response = await model.evaluate(
            audio=frames,
            input="A support call. The caller speaks first.",
            decisions={"frustrated": llm.Probability("Is the caller frustrated?")},
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
        super().__init__(
            capabilities=DecisionCapabilities(
                decision_kinds=["probability", "choice", "score"], audio_input=True
            )
        )
        key = api_key or os.environ.get("HUME_API_KEY")
        if not key:
            raise ValueError("Hume API key is required via api_key or HUME_API_KEY env var")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._session = http_session

    @property
    def model(self) -> str:
        return "audio-decisions"

    @property
    def provider(self) -> str:
        return "Hume"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        raise ValueError("Hume decisions require audio: pass evaluate(audio=...)")

    async def _evaluate_audio_impl(
        self,
        *,
        input: str,
        audio: rtc.AudioFrame,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        """``audio`` must be 5 seconds to 5 minutes long. It is mixed to mono and
        resampled to 16 kHz before upload. ``request_id`` is Hume's run id."""
        if not MIN_AUDIO_DURATION <= audio.duration <= MAX_AUDIO_DURATION:
            raise ValueError(
                f"Hume decisions take {MIN_AUDIO_DURATION:g} to {MAX_AUDIO_DURATION:g} "
                f"seconds of audio, got {audio.duration:.2f}"
            )
        if len(input) > MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"Hume decision input is limited to {MAX_CONTEXT_LENGTH} characters, "
                f"got {len(input)}"
            )

        request: dict[str, Any] = {
            "questions": {name: _encode_question(d) for name, d in decisions.items()}
        }
        if input.strip():
            request["context"] = input

        # Hume reads the parts in order: `request` first, then `file`.
        form = aiohttp.FormData()
        form.add_field("request", json.dumps(request), content_type="application/json")
        form.add_field(
            "file",
            _to_model_format(audio).to_wav_bytes(),
            filename="audio.wav",
            content_type="audio/wav",
        )

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
                        await resp.text(), status=resp.status, request_id=request_id
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


def _error_from_response(text: str, *, status: int, request_id: str | None) -> APIStatusError:
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
    return create_api_error_from_http(message, status=status, request_id=request_id, body=body)
