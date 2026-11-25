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

"""Zoom Decisions for LiveKit Agents.

``zoom_ai.DecisionModel()`` answers a batch of probability, choice, and score decisions
with the Zoom AI Services Decisions API (``POST /v2/aiservices/decisions/decide``), one
synchronous request per batch.

Built on the public docs: https://developers.zoom.us/docs/ai-services/decisions/
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    llm,
    utils,
)
from livekit.agents.llm.decisions import (
    Choice,
    ChoiceResult,
    Decision,
    DecisionCapabilities,
    DecisionResponse,
    DecisionResult,
    Probability,
    ProbabilityResult,
    Score,
    ScoreResult,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .stt import _LOOPBACK_HOSTS, USER_AGENT, _tracking_id, _zoom_reason

BASE_URL = "https://api.zoom.us/v2/aiservices/decisions"

# The models that answer caller-defined questions. zoom-guardrails classifies against
# fixed safety categories and takes no questions, so it doesn't fit DecisionModel.
DecisionModels = Literal["zoom-decisions", "zoom-decisions-pro"]


def _retry_after(headers: Any, body: str) -> float | None:
    """The wait Zoom asks for before the next request, in seconds, if it gives one.

    ``Retry-After`` comes with a 429; a 503's JSON body carries it as
    ``metadata.retry_after_seconds``.
    """
    value: Any = headers.get("Retry-After")
    if value is None:
        try:
            value = (json.loads(body).get("metadata") or {}).get("retry_after_seconds")
        except (TypeError, ValueError, AttributeError):  # no body, or not a JSON object
            return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):  # absent, or an HTTP date
        return None
    return seconds if seconds > 0 else None


class DecisionModel(llm.DecisionModel):
    """Zoom Decisions: a batch of probability, choice, and score decisions in one request.

    Example::

        model = zoom_ai.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller asks for a human.")},
        )
    """

    def __init__(
        self,
        *,
        model: DecisionModels | str = "zoom-decisions",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Zoom Decisions model.

        Args:
            model: ``"zoom-decisions"`` (the default, tuned for latency and cost) or
                ``"zoom-decisions-pro"`` (for higher accuracy on complex decisions).
            api_key: Zoom AI Services API key (from https://www.zoom.ai/). Defaults to
                ``$ZOOM_AI_API_KEY``.
            base_url: Decisions API base URL; requests go to ``<base>/decide``.
            http_session: Optional aiohttp session; the job's session is used otherwise.
        """
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        if model == "zoom-guardrails":
            raise ValueError("zoom-guardrails takes no questions, so it can't be a DecisionModel")
        api_key = api_key if is_given(api_key) else os.environ.get("ZOOM_AI_API_KEY", "")
        if not api_key:
            raise ValueError("Zoom AI API key is required: pass api_key or set ZOOM_AI_API_KEY")
        self._api_key = api_key
        self._model = model
        base = base_url.rstrip("/")
        scheme, sep, _ = base.partition("://")
        if not sep or scheme.lower() not in ("http", "https"):
            raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
        if scheme.lower() == "http" and urlparse(base).hostname not in _LOOPBACK_HOSTS:
            logger.warning(
                "base_url is not encrypted (http://); the API key and text are sent in plaintext"
            )
        self._url = base + "/decide"
        self._session = http_session
        # Monotonic time before which no request is sent: the Retry-After Zoom last gave.
        self._not_before = 0.0

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        """The provider name, ``"Zoom"``."""
        return "Zoom"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
        extra_headers: Mapping[str, str],
    ) -> DecisionResponse:
        # Zoom asks clients to wait out Retry-After before trying again; a request sooner
        # than that is refused, and would use up the retries before the wait is over.
        if (wait := self._not_before - time.monotonic()) > 0:
            if wait >= conn_options.timeout:
                # longer than an attempt may take: stop the retries
                raise APIConnectionError(
                    f"Zoom Decisions asked to wait {wait:.1f}s before the next request",
                    retryable=False,
                )
            await asyncio.sleep(wait)

        # Each request has its own tracking ID, which Zoom support uses to find it.
        tracking_id = _tracking_id()
        headers = {
            **extra_headers,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-zm-trackingid": tracking_id,
        }
        payload = {
            "model": self._model,
            "input": input,
            "questions": [_encode_question(name, d) for name, d in decisions.items()],
        }
        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                self._url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    if (retry_after := _retry_after(resp.headers, text)) is not None:
                        self._not_before = max(self._not_before, time.monotonic() + retry_after)
                    # Zoom explains a refusal (a bad API key or question, say) in the body.
                    reason = _zoom_reason(text)
                    raise APIStatusError(
                        f"Zoom Decisions error (HTTP {resp.status})"
                        + (f": {reason}" if reason else ""),
                        status_code=resp.status,
                        request_id=tracking_id,
                        body=text,
                    )
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Zoom Decisions request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Zoom Decisions request failed") from e

        try:
            body = json.loads(text)
            answers = body["answers"]
            if not isinstance(answers, list):
                raise TypeError("answers is not a list")
        except (ValueError, KeyError, TypeError) as e:
            raise APIError("invalid Zoom Decisions response", retryable=False) from e

        results, errors = _decode_answers(answers, decisions)
        usage = body.get("usage") or {}
        # input_units counts input tokens; output_units counts decisions, not tokens.
        input_tokens = usage.get("input_units") if usage.get("unit_type") == "tokens" else None
        return DecisionResponse(
            results=results,
            errors=errors,
            model=body.get("model") or self._model,
            provider=self.provider,
            request_id=body.get("request_id") or tracking_id,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        )


def _encode_question(name: str, decision: Decision) -> dict[str, Any]:
    if isinstance(decision, Probability):
        return {"type": "predicate", "name": name, "instructions": decision.prompt}
    if isinstance(decision, Choice):
        return {
            "type": "choice",
            "name": name,
            "instructions": decision.prompt,
            "choices": [
                {"value": value, **({"description": description} if description else {})}
                for value, description in decision.options.items()
            ],
        }
    return {
        "type": "score",
        "name": name,
        "instructions": decision.prompt,
        "levels": [{"label": level} for level in decision.levels],
    }


def _decode_answers(
    answers: list[Any], decisions: Mapping[str, Decision]
) -> tuple[dict[str, DecisionResult], dict[str, str]]:
    """Pair each answer with its decision by the name Zoom repeats in it.

    A decision without an answer is reported missing by ``evaluate``.
    """
    results: dict[str, DecisionResult] = {}
    errors: dict[str, str] = {}
    for answer in answers:
        name = answer.get("name") if isinstance(answer, dict) else None
        if not isinstance(name, str) or not name:
            raise APIError("decision answer does not name a decision", retryable=False)
        if name in results or name in errors:
            raise APIError("decision response answers a decision twice", retryable=False)
        if name not in decisions:
            raise APIError("decision response contains unknown decisions", retryable=False)
        try:
            results[name] = _decode_answer(answer, decisions[name])
        except (KeyError, TypeError, ValueError):
            errors[name] = "invalid Zoom Decisions answer"
        except APIError as error:
            errors[name] = error.message
    return results, errors


def _decode_answer(answer: dict[str, Any], decision: Decision) -> DecisionResult:
    kind = answer.get("type")
    if kind == "refusal":
        raise APIError("the model declined this decision", retryable=False)
    if kind == "predicate" and isinstance(decision, Probability):
        return ProbabilityResult(value=answer["probability"])
    if kind == "choice" and isinstance(decision, Choice):
        # Only string values are sent, so a boolean choice names no option.
        if not isinstance(answer["choice"], str):
            raise APIError("choice answer is not one of the requested options", retryable=False)
        return ChoiceResult(
            value=answer["choice"],
            probabilities=_probabilities(answer, list(decision.options)),
            provider_data=_confidence(answer),
        )
    if kind == "score" and isinstance(decision, Score):
        positions = [str(index) for index in range(len(decision.levels))]
        by_position = _probabilities(answer, positions)
        # Zoom returns level indices without their labels; the levels are the ones sent.
        return ScoreResult(
            value=answer["score"],
            probabilities={int(position): p for position, p in by_position.items()},
            levels=list(decision.levels),
            provider_data=_confidence(answer),
        )
    raise APIError(f"unexpected answer type {kind!r}", retryable=False)


def _probabilities(answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
    """The probability Zoom gave each key, 0 for any it left out.

    Zoom lists probabilities as ``{"value", "probability"}`` items, where the value is a
    choice's option or a level's index.
    """
    given = {str(item["value"]): float(item["probability"]) for item in answer["probabilities"]}
    if given.keys() - set(keys):
        raise APIError("distribution names values that weren't asked about", retryable=False)
    return {key: given.get(key, 0.0) for key in keys}


def _confidence(answer: dict[str, Any]) -> dict[str, Any]:
    return {"confidence": answer["confidence"]} if "confidence" in answer else {}
