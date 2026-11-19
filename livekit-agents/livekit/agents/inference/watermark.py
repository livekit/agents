from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass
from typing import BinaryIO, Literal

import aiohttp
from pydantic import BaseModel, Field, ValidationError

from .._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
)
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import http_context, is_given
from ._utils import (
    HEADER_REQUEST_ID,
    _terminal_denial,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)

WatermarkProvider = Literal["cartesia", "inworld"]

# The gateway names this route provenance; it is the watermark check endpoint.
_CHECK_PATH = "/provenance"

# The gateway reads the file's own header, so the part needs no real name or type.
# A caller's file name can identify a person and is never sent.
_UPLOAD_FILE_NAME = "audio"
_UPLOAD_CONTENT_TYPE = "application/octet-stream"

# Bounds one attempt, upload and answer together, above the gateway's worst case of
# about 185 s (120 s to receive the upload, 5 s for a provider slot, 60 s for the
# provider). conn_options.timeout bounds the connect only.
_ATTEMPT_TIMEOUT = 190.0

# A Retry-After longer than this is not waited out; the error goes to the caller.
_MAX_RETRY_AFTER = 60.0

# Longest stretch of a plain-text error body quoted in the exception message.
_MAX_TEXT_IN_MESSAGE = 200

_INVALID_FILE = "file must be a path, bytes, or a file object opened in binary mode"


@dataclass(frozen=True)
class WatermarkResult:
    """One signal a provider looked for in a file."""

    type: str
    """The kind of signal: ``"watermark"``, ``"classifier"``, or a value added later."""

    outcome: str | None = None
    """The provider's verdict, when it returns one.

    ``"detected"`` means the provider's signal was found in the file.
    ``"not_detected"`` means this provider's signal was not found in this file. It
    does not mean the audio is not AI-generated: another provider may have produced
    it, or the signal may not have survived editing or re-encoding. A value added
    later is passed through unchanged.
    """

    probability: float | None = None
    """The provider's score from 0 to 1, when it returns one, exactly as the gateway
    sent it. No threshold is applied. ``None`` together with a ``None`` outcome means
    the provider could not score the file."""


@dataclass(frozen=True)
class WatermarkCheck:
    """A provider's answer for one file."""

    provider: str
    results: list[WatermarkResult]
    request_id: str | None = None
    """The gateway's id for the request, when it sent one."""


class _GatewayError(BaseModel):
    type: str = ""
    error: str = ""
    hint: str = ""


class _ResultBody(BaseModel):
    type: str
    outcome: str | None = None
    probability: float | None = None


class _CheckBody(BaseModel):
    provider: str
    # An empty list would read as "nothing detected", which no provider said.
    results: list[_ResultBody] = Field(min_length=1)


@dataclass(frozen=True)
class _Reply:
    status: int
    body: bytes
    request_id: str | None
    retry_after: float | None
    terminal: bool


class WatermarkDetector:
    """Checks audio for a TTS provider's AI-generation signal through LiveKit Inference.

    Each check sends one file to one provider's detector and returns that provider's
    answer unchanged: a watermark verdict from Cartesia, a classifier score from
    Inworld.

    Example::

        detector = inference.WatermarkDetector()
        check = await detector.check("reply.wav", provider="cartesia")
        for result in check.results:
            print(result.type, result.outcome, result.probability)

    The check is off for a project until LiveKit enables it; until then every call
    raises a non-retryable 403 ``not_enabled``.

    Data residency: Cartesia has no EU endpoint. An EU-pinned project gets a 403
    ``inference_region_restricted`` for it, and an EU-tagged request a 400
    ``not_available`` whose ``hint`` lists the residency tags the provider supports.
    Inworld serves the EU.

    Credentials resolve from the arguments, then ``LIVEKIT_INFERENCE_API_KEY`` /
    ``LIVEKIT_INFERENCE_API_SECRET``, then ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``.
    """

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            base_url: Inference gateway base URL. Defaults to the environment's
                gateway (see ``get_default_inference_url``).
            api_key: Gateway API key (see the class docstring for resolution).
            api_secret: Gateway API secret (see the class docstring).
            http_session: Optional caller-owned session. Otherwise uses the job's
                HTTP session. Outside a job, use ``async with utils.http_context.open()``.
            conn_options: Retry settings. ``timeout`` bounds the connect, not the upload.
        """
        self._base_url = base_url if is_given(base_url) else get_default_inference_url()
        self._api_key, self._api_secret = resolve_credentials(
            api_key if is_given(api_key) else None,
            api_secret if is_given(api_secret) else None,
        )
        self._session = http_session
        self._conn_options = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS

    async def check(
        self,
        file: str | os.PathLike[str] | bytes | BinaryIO,
        *,
        provider: WatermarkProvider | str,
    ) -> WatermarkCheck:
        """Ask one provider whether a file still carries its AI-generation signal.

        Args:
            file: The audio, as a path, bytes, or a binary file object (read from its
                current position and left open). It needs a container header: WAV, MP3,
                OGG, FLAC, M4A or WebM. Raw PCM is refused. The file's name is not sent.
            provider: The provider whose signal to look for. One per call.

        Returns:
            The provider's results, at least one. See :class:`WatermarkResult` for what
            an outcome means: ``"not_detected"`` is not proof of human speech.

        Raises:
            APIStatusError: The gateway refused the request. ``body`` holds its ``type``,
                ``error`` and ``hint``, or the raw text when the body is not JSON (a 401,
                a suspended project's 403, a draining gateway's 503). Every request that
                reaches the gateway counts against the project's daily checks, so only
                two answers are retried: a 503 (provider at capacity, gateway busy or
                draining) and a 429 that sends ``Retry-After``. Each waits the gateway's
                ``Retry-After``, and one over 60 s is raised instead. A 503 that sends
                no usable ``Retry-After`` waits the ``conn_options`` retry interval. A
                502 or 504 is raised at once: the provider failed on this file and
                would again.
            APITimeoutError: The attempt connected and did not finish within 190 s.
                Not retried.
            APIConnectionError: The gateway could not be reached. A failed connect, a
                connect that timed out included, is retried, since the gateway never
                saw it; a dropped request is not.
            APIError: The gateway answered 200 with a body that is not a check.
            ValueError: The file is empty.
            TypeError: ``file`` is not a path, bytes, or a binary file object.
        """
        audio = await _read_audio(file)
        session = self._session or http_context.http_session()
        for attempt in range(self._conn_options.max_retry + 1):
            answer, retry_after = await self._attempt(session, audio, provider)
            if isinstance(answer, WatermarkCheck):
                return answer
            delay = self._retry_delay(answer, retry_after, attempt)
            if delay is None:
                raise answer
            _log_retry(answer, provider=provider, attempt=attempt, delay=delay)
            await self._sleep(delay)
        raise RuntimeError("unreachable")

    async def _sleep(self, delay: float) -> None:
        # The retry wait, on the instance so a test can replace it without patching asyncio.
        await asyncio.sleep(delay)

    async def _attempt(
        self, session: aiohttp.ClientSession, audio: bytes, provider: str
    ) -> tuple[WatermarkCheck | APIError, float | None]:
        try:
            reply = await self._post(session, audio, provider)
        except APIError as error:
            return error, None
        if reply.status >= 400:
            return _error_from_reply(reply), reply.retry_after
        try:
            return _check_from_reply(reply), None
        except APIError as error:
            return error, None

    async def _post(self, session: aiohttp.ClientSession, audio: bytes, provider: str) -> _Reply:
        try:
            async with session.post(
                f"{self._base_url.rstrip('/')}{_CHECK_PATH}",
                headers=self._headers(),
                data=_form(audio, provider),
                timeout=aiohttp.ClientTimeout(
                    total=_ATTEMPT_TIMEOUT, sock_connect=self._conn_options.timeout
                ),
            ) as response:
                return _Reply(
                    status=response.status,
                    body=await response.read(),
                    request_id=response.headers.get(HEADER_REQUEST_ID),
                    retry_after=_parse_retry_after(response.headers.get("Retry-After")),
                    terminal=_terminal_denial(response.headers),
                )
        except (aiohttp.ConnectionTimeoutError, aiohttp.ClientConnectorError) as exc:
            # Listed before TimeoutError: a connect timeout is one too, and it never
            # reached the gateway.
            raise APIConnectionError("inference gateway connection failed") from exc
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("inference watermark check timed out", retryable=False) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError(
                "inference gateway connection failed", retryable=False
            ) from exc

    def _headers(self) -> dict[str, str]:
        # Minted per request: a retry must not reuse a token that may have expired.
        return {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._api_key, self._api_secret)}",
        }

    def _retry_delay(
        self, failure: APIError, retry_after: float | None, attempt: int
    ) -> float | None:
        """Seconds to wait before the next attempt, or None to give up."""
        if not failure.retryable or attempt >= self._conn_options.max_retry:
            return None
        if retry_after is None:
            return self._conn_options._interval_for_retry(attempt)
        if retry_after > _MAX_RETRY_AFTER:
            return None
        return retry_after


async def _read_audio(file: str | os.PathLike[str] | bytes | BinaryIO) -> bytes:
    if isinstance(file, (bytes, bytearray, memoryview)):
        audio = bytes(file)
    elif isinstance(file, (str, os.PathLike)):
        audio = await asyncio.to_thread(_read_path, file)
    else:
        audio = await _read_file_object(file)
    if not audio:
        raise ValueError("file is empty")
    return audio


async def _read_file_object(file: object) -> bytes:
    read = getattr(file, "read", None)
    # A text-mode file is refused before the read, which could fail to decode the audio.
    if isinstance(file, io.TextIOBase) or not callable(read):
        raise TypeError(_INVALID_FILE)
    data = await asyncio.to_thread(read)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(_INVALID_FILE)
    return bytes(data)


def _read_path(path: str | os.PathLike[str]) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _form(audio: bytes, provider: str) -> aiohttp.FormData:
    # Built per attempt: aiohttp sends a FormData once.
    form = aiohttp.FormData()
    form.add_field("provider", provider)
    form.add_field(
        "file",
        io.BytesIO(audio),
        filename=_UPLOAD_FILE_NAME,
        content_type=_UPLOAD_CONTENT_TYPE,
    )
    return form


def _parse_retry_after(value: str | None) -> float | None:
    """Whole seconds from a Retry-After header, or None when absent or unusable.

    Zero and anything that is not a positive whole number count as absent: a retry
    never goes out with no delay.
    """
    digits = (value or "").strip()
    if not (digits.isascii() and digits.isdigit()):
        return None
    return float(digits) or None


def _is_retryable(reply: _Reply) -> bool:
    if reply.terminal:
        return False
    if reply.status == 503:
        return True
    return reply.status == 429 and reply.retry_after is not None


def _check_from_reply(reply: _Reply) -> WatermarkCheck:
    try:
        parsed = _CheckBody.model_validate_json(reply.body)
    except ValidationError as exc:
        raise APIStatusError(
            "invalid inference watermark check response",
            status_code=reply.status,
            request_id=reply.request_id,
            retryable=False,
        ) from exc
    return WatermarkCheck(
        provider=parsed.provider,
        results=[
            WatermarkResult(type=r.type, outcome=r.outcome, probability=r.probability)
            for r in parsed.results
        ],
        request_id=reply.request_id,
    )


def _error_from_reply(reply: _Reply) -> APIError:
    body, detail = _error_body(reply.body)
    message = "inference gateway returned an error"
    if detail:
        message = f"{message}: {detail}"
    error = create_api_error_from_http(
        message, status=reply.status, request_id=reply.request_id, body=body
    )
    error.retryable = _is_retryable(reply)
    return error


def _error_body(raw: bytes) -> tuple[object, str]:
    """The error body to attach and a short detail for the message.

    The gateway's JSON error object when it sent one, else the text: a 401, a
    suspended project's 403 and a draining gateway's 503 are plain text, and no body
    is trusted to be UTF-8.
    """
    parsed = _gateway_error(raw)
    if parsed is not None:
        return parsed.model_dump(), parsed.error or parsed.type
    text = raw.decode("utf-8", errors="replace")
    return text, text.strip()[:_MAX_TEXT_IN_MESSAGE]


def _gateway_error(body: bytes) -> _GatewayError | None:
    """The gateway's error object, or None when the body is not one."""
    try:
        parsed = _GatewayError.model_validate_json(body)
    except ValidationError:
        return None
    return parsed if parsed.type or parsed.error else None


def _log_retry(failure: APIError, *, provider: str, attempt: int, delay: float) -> None:
    extra: dict[str, object] = {"provider": provider, "attempt": attempt, "delay": delay}
    if isinstance(failure, APIStatusError):
        extra["status_code"] = failure.status_code
        extra["request_id"] = failure.request_id
    logger.warning("watermark check failed, retrying", extra=extra)
