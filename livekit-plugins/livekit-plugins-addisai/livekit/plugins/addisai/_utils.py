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

import json
from typing import Any, cast

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError

from .models import Language

API_BASE_URL = "https://api.addisassistant.com"


def validate_language(language: str) -> Language:
    if language not in ("am", "om"):
        raise ValueError("language must be either 'am' (Amharic) or 'om' (Afaan Oromo)")
    return cast(Language, language)


def response_request_id(response: aiohttp.ClientResponse) -> str | None:
    return response.headers.get("x-request-id") or response.headers.get("cf-ray")


def unwrap_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise APIConnectionError("AddisAI returned an unexpected response payload")
    return data


async def parse_json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
    raw_body = await response.text()
    body: object
    try:
        body = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        body = raw_body

    if response.status >= 400:
        raise APIStatusError(
            _error_message(body, response.reason),
            status_code=response.status,
            request_id=response_request_id(response),
            body=body,
        )

    if not isinstance(body, dict):
        raise APIConnectionError("AddisAI returned a non-JSON response")

    return body


async def raise_for_audio_response(response: aiohttp.ClientResponse) -> None:
    if response.status < 400:
        return

    raw_body = await response.text()
    body: object
    try:
        body = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        body = raw_body

    raise APIStatusError(
        _error_message(body, response.reason),
        status_code=response.status,
        request_id=response_request_id(response),
        body=body,
    )


def _error_message(body: object, fallback: str | None) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        elif isinstance(error, str) and error:
            return error

        message = body.get("message")
        if isinstance(message, str) and message:
            return message

    if isinstance(body, str) and body:
        return body[:500]

    return fallback or "AddisAI request failed"
