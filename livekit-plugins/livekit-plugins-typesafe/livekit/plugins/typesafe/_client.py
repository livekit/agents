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

import asyncio
import os
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, utils

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"

DEFAULT_MODEL = "jev-1.13.0"
"""Pinned rather than the ``jev-latest`` alias.

The thresholds in :func:`~livekit.plugins.typesafe.default_checks` were measured
against this version. An alias moves on TypeSafe's schedule, and a new version
can score the same reply differently, which would leave those thresholds quietly
mis-set. Pass ``model="jev-latest"`` to track the alias instead, and re-measure
the thresholds when it moves.
"""


class SystemOneClient:
    """Async client for TypeSafe's System One evaluation endpoint.

    The whole API surface we need is a single ``POST /v1/systemone`` with a JSON
    body, so this talks to it directly over the agent's shared aiohttp session
    rather than depending on ``typesafe-sdk`` (which would pull in a second HTTP
    stack alongside the one ``livekit-agents`` already ships).

    There are deliberately no retries. This runs on a live call, and by the time a
    backoff completes the turn it was judging is over, so a 429 or 529 is dropped like
    any other failure and the caller falls through unchecked.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 2.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError(
                "TypeSafe API key is required, either as argument or set "
                "TYPESAFE_API_KEY environment variable"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def evaluate(
        self, state: Any, questions: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        """Evaluate ``state`` against ``questions`` and return the parsed response body.

        Every question in the map is answered against a single ingestion of
        ``state``, in parallel, within this one request.
        """
        payload = {"state": state, "model": model or self._model, "questions": questions}
        try:
            async with self._ensure_session().post(
                f"{self._base_url}/systemone",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    body: Any
                    try:
                        body = await resp.json()
                    except Exception:
                        body = await resp.text()
                    raise APIStatusError(
                        f"typesafe: system one request failed with status {resp.status}",
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=body,
                    )
                data: dict[str, Any] = await resp.json()
                return data
        except asyncio.TimeoutError as e:
            raise APITimeoutError(f"typesafe: request timed out after {self._timeout}s") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"typesafe: {e}") from e
