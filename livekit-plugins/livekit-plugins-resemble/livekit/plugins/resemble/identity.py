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

import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import aiohttp

from livekit.agents import APIStatusError, utils

RESEMBLE_IDENTITY_API_URL = "https://app.resemble.ai/api/v2"

DEFAULT_MATCH_THRESHOLD = 70.0
"""Similarity score (0-100) at or above which the best match counts as verified."""


class AudioHost(Protocol):
    """Hook that makes one audio clip publicly reachable and returns its URL.

    Resemble Identity's ``/search`` endpoint fetches audio from a URL instead of
    accepting a file upload, so callers who start from raw bytes must host the clip
    somewhere Resemble's backend can reach — an S3 presigned URL, a public bucket,
    or an app endpoint. Any async callable with this shape works:

    .. code-block:: python

        async def host_audio(audio: bytes, filename: str) -> str:
            key = f"identity/{filename}"
            await s3.put_object(Bucket="clips", Key=key, Body=audio)
            return f"https://clips.example.com/{key}"
    """

    async def __call__(self, audio: bytes, filename: str) -> str:
        """Host ``audio`` and return a publicly reachable URL for it."""
        ...


class IdentityTransport(Protocol):
    """Transport used by :class:`ResembleIdentity` to call Resemble Identity."""

    async def search(self, url: str, *, request_timeout: float) -> dict[str, Any]:
        """Search enrolled identities against hosted audio; return the ``item`` payload."""


@dataclass
class IdentityMatch:
    """One enrolled speaker compared against the submitted audio."""

    uuid: str
    name: str
    score: float | None
    """Similarity to the submitted audio (0-100, higher is closer); None if unscored."""
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable payload for this enrolled-speaker match."""
        return {
            "uuid": self.uuid,
            "name": self.name,
            "score": self.score,
        }


@dataclass
class IdentityResult:
    """Outcome of searching enrolled identities with Resemble Identity."""

    matches: list[IdentityMatch]
    """All scored enrollments, sorted best-first (unscored entries last)."""
    threshold: float
    """Similarity at or above which :attr:`matched` is True."""
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def best(self) -> IdentityMatch | None:
        """The closest enrolled speaker, or None when nothing is enrolled."""
        return self.matches[0] if self.matches else None

    @property
    def matched(self) -> bool:
        """Whether the best match clears the configured similarity threshold."""
        best = self.best
        return bool(best and best.score is not None and best.score >= self.threshold)

    @property
    def name(self) -> str | None:
        """Name of the closest enrolled speaker, or None."""
        return self.best.name if self.best else None

    @property
    def score(self) -> float | None:
        """Similarity of the closest enrolled speaker, or None."""
        return self.best.score if self.best else None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable payload suitable for app events or dashboards."""
        return {
            "matched": self.matched,
            "name": self.name,
            "score": self.score,
            "threshold": self.threshold,
            "matches": [match.to_dict() for match in self.matches],
        }


class ResembleIdentity:
    """Speaker verification against enrolled voiceprints, powered by Resemble Identity.

    Identity completes the Resemble security trio: Detect answers "is this media
    synthetic?", Signal answers "does this content match a fraud/scam pattern?", and
    Identity answers "is this the enrolled speaker?".

    Because the underlying ``/search`` endpoint fetches audio from a URL rather than
    accepting an upload, raw-bytes searches need an :class:`AudioHost` hook that hosts
    the clip publicly. Apps without hosting infrastructure can skip Identity entirely
    (Detect and Signal keep working) or search pre-hosted clips with
    :meth:`search_url`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = RESEMBLE_IDENTITY_API_URL,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        audio_host: AudioHost | None = None,
        http_session: aiohttp.ClientSession | None = None,
        transport: IdentityTransport | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        """Create an Identity client.

        Args:
            api_key (str, optional): Resemble API key. If omitted, ``RESEMBLE_API_KEY`` is
                read from the environment. Not required when ``transport`` is provided.
                Pass a full ``"Bearer ..."`` value to override the default bearer header.
            base_url (str, optional): Override the Resemble REST Identity API base URL.
            threshold (float, optional): Similarity score (0-100) at or above which the
                best match counts as verified. Defaults to 70.
            audio_host (AudioHost, optional): Async hook that hosts audio bytes at a
                publicly reachable URL. Required only for :meth:`search`;
                :meth:`search_url` works without it.
            http_session (aiohttp.ClientSession, optional): Existing session for the
                default REST transport.
            transport (IdentityTransport, optional): Custom transport, useful for
                gateways or tests.
            request_timeout (float, optional): Per-request timeout in seconds.
        """
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        _validate_threshold(threshold)

        if transport is None:
            api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Resemble API key is required, either as argument or set RESEMBLE_API_KEY"
                    " environment variable"
                )
            transport = RestIdentityTransport(
                api_key=api_key,
                base_url=base_url,
                http_session=http_session,
            )

        self._transport = transport
        self._threshold = threshold
        self._audio_host = audio_host
        self._request_timeout = request_timeout

    @property
    def threshold(self) -> float:
        """Configured similarity threshold."""
        return self._threshold

    async def search(
        self,
        audio: bytes,
        *,
        filename: str = "identity-search.wav",
        threshold: float | None = None,
        request_timeout: float | None = None,
    ) -> IdentityResult:
        """Verify raw audio bytes against the account's enrolled identities.

        Hosts the clip through the configured ``audio_host`` hook, then searches the
        resulting URL. Raises ValueError when no ``audio_host`` was provided.
        """
        if not audio:
            raise ValueError("audio is required")
        if self._audio_host is None:
            raise ValueError(
                "searching raw audio requires an audio_host hook: Resemble Identity's"
                " /search endpoint fetches audio from a public URL instead of accepting"
                " an upload. Provide audio_host=..., or host the clip yourself and call"
                " search_url()."
            )
        url = await self._audio_host(audio, filename)
        return await self.search_url(
            url,
            threshold=threshold,
            request_timeout=request_timeout,
        )

    async def search_url(
        self,
        url: str,
        *,
        threshold: float | None = None,
        request_timeout: float | None = None,
    ) -> IdentityResult:
        """Verify audio already hosted at a publicly reachable URL."""
        url = url.strip()
        if not url:
            raise ValueError("url is required")
        # the URL is fetched by Resemble's backend — never forward non-web schemes
        # (file://, internal metadata endpoints, ...) from untrusted input
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("url must be an absolute http(s) URL")
        if threshold is not None:
            _validate_threshold(threshold)
        item = await self._transport.search(
            url,
            request_timeout=request_timeout or self._request_timeout,
        )
        return _parse_identity_result(
            item,
            threshold=threshold if threshold is not None else self._threshold,
        )

    async def aclose(self) -> None:
        """Close transport resources when the transport exposes a close method."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable


class RestIdentityTransport:
    """Default transport for Resemble Identity's REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = RESEMBLE_IDENTITY_API_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = http_session

    async def search(self, url: str, *, request_timeout: float) -> dict[str, Any]:
        async with self._ensure_session().post(
            f"{self._base_url}/identity/search",
            headers={
                "Authorization": _authorization_value(self._api_key),
                "Content-Type": "application/json",
            },
            json={"url": url},
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            if resp.status < 200 or resp.status >= 300:
                body = await resp.text()
                raise APIStatusError(
                    message="resemble identity request failed",
                    status_code=resp.status,
                    request_id=None,
                    body=body[:500],
                )
            payload: object = await resp.json()
            if not isinstance(payload, dict):
                raise APIStatusError(
                    message="resemble identity response was not a JSON object",
                    status_code=resp.status,
                    request_id=None,
                    body=str(payload)[:500],
                )

        item = payload.get("item")
        if item is None:
            return {}
        if not isinstance(item, Mapping):
            raise ValueError(f"Resemble Identity response has malformed item: {payload}")
        return dict(cast("Mapping[str, Any]", item))

    def _ensure_session(self) -> aiohttp.ClientSession:
        session = self._session
        if session is None:
            session = utils.http_context.http_session()
            self._session = session

        return session


def _parse_identity_result(item: Mapping[str, Any], *, threshold: float) -> IdentityResult:
    matches = []
    for uuid, info in item.items():
        if not isinstance(info, Mapping):
            continue
        matches.append(
            IdentityMatch(
                uuid=str(uuid),
                name=str(info.get("name") or "Unknown"),
                score=_opt_float(info.get("distance")),
                raw=dict(info),
            )
        )
    matches.sort(key=lambda match: (match.score is None, -(match.score or 0.0)))
    return IdentityResult(matches=matches, threshold=threshold, raw=dict(item))


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 100.0:
        raise ValueError("threshold must be between 0 and 100")


def _authorization_value(api_key: str) -> str:
    return api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
