from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any, NoReturn, get_args

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.utils import shortuuid

from .log import logger
from .models import (
    CLUSTER_HTTP_BASE_URLS,
    CLUSTER_WS_BASE_URLS,
    DEFAULT_CLUSTER,
    GLOBAL_TTS_VOICES,
    OPENAI_COMPAT_SUFFIX,
    Cluster,
    TTSVoices,
)

_KNOWN_TTS_VOICES = frozenset(get_args(TTSVoices))


def resolve_api_key(api_key: str | None) -> str:
    """Explicit argument wins, then the environment variable.

    Note that the two clusters issue separate credentials: a mainland-China key
    will not authenticate against the global endpoint, or vice versa.
    """
    key = api_key or os.environ.get("STEPFUN_API_KEY")
    if not key:
        raise ValueError(
            "StepFun API key is required. "
            "Pass api_key=... or set the STEPFUN_API_KEY environment variable."
        )
    return key


def resolve_cluster(cluster: Cluster | None) -> Cluster:
    value = cluster if cluster is not None else os.environ.get("STEPFUN_CLUSTER")
    if not value:
        return DEFAULT_CLUSTER
    if value == "global":
        return "global"
    if value == "china":
        return "china"
    raise ValueError(f"STEPFUN_CLUSTER must be 'global' or 'china', got {value!r}")


def resolve_ws_base_url(base_url: str | None, cluster: Cluster) -> str:
    return (base_url or CLUSTER_WS_BASE_URLS[cluster]).rstrip("/")


def resolve_http_base_url(base_url: str | None, cluster: Cluster) -> str:
    return (base_url or CLUSTER_HTTP_BASE_URLS[cluster]).rstrip("/")


def resolve_llm_base_url(base_url: str | None, cluster: Cluster) -> str:
    """Append `/v1` unless the explicit OpenAI-compatible base URL already has it."""
    resolved = resolve_http_base_url(base_url, cluster)
    if resolved.endswith(OPENAI_COMPAT_SUFFIX):
        return resolved
    return f"{resolved}{OPENAI_COMPAT_SUFFIX}"


def auth_headers(api_key: str) -> dict[str, str]:
    """The protocol requires the API key in a WebSocket header."""
    return {"Authorization": f"Bearer {api_key}"}


def warn_if_voice_unavailable(voice: TTSVoices | str, cluster: Cluster) -> None:
    """Warn when a voice is known to be mainland-China only but used globally."""
    if cluster != "global":
        return
    if voice in GLOBAL_TTS_VOICES:
        return
    if voice not in _KNOWN_TTS_VOICES:
        # Unrecognized entirely — most likely a cloned voice, which is
        # account-specific and legitimately absent from any static list.
        return

    logger.warning(
        "voice is not published for the global cluster; the request will "
        "likely fail with 400 voice_id_invalid. Use TTS.list_voices() to see "
        "what is available.",
        extra={"voice": voice, "cluster": cluster},
    )


def event_id() -> str:
    """Every client frame on the ASR WebSocket carries its own event_id."""
    return shortuuid()


def pcm_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def b64_to_pcm(data: str) -> bytes:
    return base64.b64decode(data)


async def iter_sse_events(resp: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON payloads from an SSE response.

    Only the `data:` field is of interest here; the batch ASR route carries the
    event type inside the JSON rather than in an SSE `event:` line. A literal
    `[DONE]` terminates the stream, following the convention the TTS SSE route
    uses.
    """
    async for raw in resp.content:
        line = raw.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            if payload == "[DONE]":
                return
            continue

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("stepfun: failed to decode sse payload")


_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_QUOTA_ERROR_IDENTIFIERS = (
    "insufficient_credit",
    "project_credit_limit_exceeded",
    "member_project_credit_limit_exceeded",
)

#: WebSocket error codes on the streaming ASR endpoint that are worth retrying.
#: Both timeout codes mean the connection went stale, which reconnecting fixes.
_RETRYABLE_WS_CODES = frozenset({"internal_error", "max_idle_timeout", "pong_timeout"})


def _is_quota_exhaustion(status_code: int, body: Any) -> bool:
    if status_code != 429 or not body:
        return False
    text = body if isinstance(body, str) else str(body)
    return any(marker in text for marker in _QUOTA_ERROR_IDENTIFIERS)


class _SafeAPIStatusError(APIStatusError):
    """Retain provider details for callers without exposing them in retry logs."""

    def __str__(self) -> str:
        return (
            f"message={self.message!r}, status_code={self.status_code}, "
            f"retryable={self.retryable}, request_id={self.request_id}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


def status_error(
    *,
    status_code: int,
    message: str,
    request_id: str | None = None,
    body: Any = None,
) -> APIStatusError:
    retryable = status_code in _RETRYABLE_STATUS and not _is_quota_exhaustion(status_code, body)
    return _SafeAPIStatusError(
        message,
        status_code=status_code,
        request_id=request_id,
        body=body,
        retryable=retryable,
    )


async def raise_for_status(resp: aiohttp.ClientResponse) -> None:
    """Single error exit point for the HTTP paths."""
    if resp.status < 400:
        return

    request_id = resp.headers.get("x-request-id")
    try:
        body = await resp.text()
    except Exception:
        body = None

    raise status_error(
        status_code=resp.status,
        message=f"StepFun API error: {resp.status}",
        request_id=request_id,
        body=body,
    )


def translate_exception(exc: BaseException, *, context: str) -> BaseException:
    """Translate aiohttp/asyncio exceptions into LiveKit's standard exceptions.

    Already-translated errors pass straight through, so wrapping a call site
    that itself called `raise_for_status` does not downgrade a precise
    APIStatusError into a generic connection error. Raise the result with
    `raise_translated`, not `raise ... from exc`: on the pass-through path
    that would make the exception its own cause.
    """
    if isinstance(exc, (APIStatusError, APIConnectionError, APITimeoutError)):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return APITimeoutError(f"{context}: timed out")
    if isinstance(exc, aiohttp.ClientResponseError):
        return status_error(
            status_code=exc.status,
            message=f"{context}: {exc.message}",
            request_id=None,
        )
    if isinstance(exc, aiohttp.ClientError):
        return APIConnectionError(f"{context}: {exc}")
    return exc


def raise_translated(exc: BaseException, *, context: str) -> NoReturn:
    """Raise `exc` translated, chained to the original only when it changed."""
    translated = translate_exception(exc, context=context)
    if translated is exc:
        raise exc
    raise translated from exc


def translate_asr_ws_error(frame: dict[str, Any], *, context: str) -> BaseException:
    """Translate an `error` frame from the streaming ASR WebSocket."""
    err = frame.get("error") or {}
    code = err.get("code")
    meta = frame.get("meta") or {}

    logger.warning(
        "stepfun asr: server error frame",
        extra={
            "context": context,
            "code": code,
            "error_type": err.get("type"),
            "param": err.get("param"),
            "session_id": meta.get("session_id"),
        },
    )

    return _SafeAPIStatusError(
        f"{context}: server returned an ASR error",
        status_code=-1,
        request_id=meta.get("session_id"),
        body=frame,
        retryable=code in _RETRYABLE_WS_CODES,
    )


def translate_tts_ws_error(frame: dict[str, Any], *, context: str) -> BaseException:
    """Translate a `tts.response.error` frame from the streaming TTS WebSocket."""
    data = frame.get("data") or {}
    raw_code = data.get("code")

    try:
        status_code = int(str(raw_code))
    except (TypeError, ValueError):
        status_code = -1

    logger.warning(
        "stepfun tts: server error frame",
        extra={
            "context": context,
            "code": raw_code,
            "session_id": data.get("session_id"),
        },
    )

    if status_code < 0:
        # An unparseable code carries no retry semantics. Treat it as terminal so
        # a FallbackAdapter moves on rather than retrying an unknown failure.
        return _SafeAPIStatusError(
            f"{context}: server returned a TTS error",
            status_code=-1,
            request_id=data.get("session_id"),
            body=frame,
            retryable=False,
        )

    return status_error(
        status_code=status_code,
        message=f"{context}: server returned a TTS error",
        request_id=data.get("session_id"),
        body=frame,
    )
